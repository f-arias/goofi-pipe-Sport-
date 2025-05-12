from copy import deepcopy

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam

information_dynamics_metrics = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Copy": ["xtx", "yty"],
    "Transfer": ["xty", "ytx"],
    "Erasure": ["rtx", "rty"],
    "Downward causation": ["sty", "stx", "str"],
    "Upward causation": ["xts", "yts", "rts"],
}

IIT_metrics = {
    "Information storage": ["xtx", "yty", "rtr", "sts"],
    "Transfer entropy": ["xty", "xtr", "str", "sty"],
    "Causal density": ["xtr", "ytr", "sty", "str", "str", "xty", "ytx", "stx"],
    "Integrated information": ["rts", "xts", "sts", "sty", "str", "yts", "ytx", "stx", "xty"],
}


class PhiID(Node):

    def config_input_slots():
        return {"matrix": DataType.ARRAY}

    def config_output_slots():
        return {"PhiID": DataType.ARRAY, "inf_dyn": DataType.ARRAY, "IIT": DataType.ARRAY}

    def config_params():
        return {
            "PhiID": {
                "tau": FloatParam(5, 1, 100, doc="Time lag for the PhiID algorithm"),
                "kind": StringParam(
                    "gaussian",
                    options=["gaussian", "discrete"],
                    doc="Kind of data (continuous Gaussian or discrete-binarized)",
                ),
                "redudancy": StringParam("MMI", options=["MMI", "CCS"], doc="Redundancy measure to use"),
            }
        }

    def setup(self):
        try:
            from phyid.calculate import calc_PhiID
        except ImportError:
            raise ImportError(
                "The phyid package is not installed. Please install it with the following command:\n"
                "pip install git+https://github.com/Imperial-MIND-lab/integrated-info-decomp.git"
            )

        self.calc_PhiID = calc_PhiID

    def process(self, matrix: Data):
        # If no input, do nothing
        if matrix is None or matrix.data is None:
            return None
        # Ensure data is a 2D array: channels x timepoints
        data = np.asarray(matrix.data, dtype=float)

        n_channels, n_time = data.shape

        # Read parameters

        tau = int(self.params["PhiID"]["tau"].value)
        kind = self.params["PhiID"]["kind"].value
        redundancy = self.params["PhiID"]["redudancy"].value

        # List of atom names in fixed order
        atom_names = [
            "rtr",
            "rtx",
            "rty",
            "rts",
            "xtr",
            "xtx",
            "xty",
            "xts",
            "ytr",
            "ytx",
            "yty",
            "yts",
            "str",
            "stx",
            "sty",
            "sts",
        ]
        n_atoms = len(atom_names)

        # Prepare output array: one row per channel, one col per atom
        PhiID_vals = np.zeros((n_channels, n_atoms), dtype=np.float32)
        inf_dyn_vals = np.zeros((n_channels, len(information_dynamics_metrics)), dtype=np.float32)
        IIT_vals = np.zeros((n_channels, len(IIT_metrics)), dtype=np.float32)
        # Compute PhiID for each channel vs. the mean of all other channels
        for i in range(n_channels):
            src = data[i]
            if n_channels > 1:
                # target is average of all other channels
                trg = np.mean(data[np.arange(n_channels) != i], axis=0)
            else:
                # only one channel: create trg as the timelagged version of src
                trg = np.roll(src, tau)
                # TODO

            # Run the PhiID calculation
            atoms_res, _ = self.calc_PhiID(src, trg, tau, kind=kind, redundancy=redundancy)
            # add 'str', 'stx', 'sty', 'sts' together

            # Each atoms_res[name] is a vector length n_time - tau
            # We average over time to get a single scalar per atom
            for j, name in enumerate(atom_names):
                PhiID_vals[i, j] = float(np.mean(atoms_res[name]))
            for j, name in enumerate(information_dynamics_metrics):
                # Get the indices of the atoms in the information_dynamics_metrics dict
                atom_indices = [atom_names.index(atom) for atom in information_dynamics_metrics[name]]
                # Sum the values of the atoms and average over time
                inf_dyn_vals[i, j] = float(np.mean(np.sum(PhiID_vals[i, atom_indices], axis=0)))
            for j, name in enumerate(IIT_metrics):
                # Get the indices of the atoms in the IIT_metrics dict
                atom_indices = [atom_names.index(atom) for atom in IIT_metrics[name]]
                # Sum the values of the atoms and average over time
                IIT_vals[i, j] = float(np.mean(np.sum(PhiID_vals[i, atom_indices], axis=0)))
                if name == "Integrated information":
                    # Subtract rtr
                    IIT_vals[i, j] -= float(np.mean(atoms_res["rtr"]))

        # Build metadata for output
        # Copy original metadata but replace channel dims
        out_meta = {}
        if matrix.meta is not None:
            out_meta = deepcopy(matrix.meta)
        # Overwrite channels info
        channel_labels = None
        if matrix.meta and "channels" in matrix.meta and "dim0" in matrix.meta["channels"]:
            channel_labels = matrix.meta["channels"]["dim0"]
        else:
            channel_labels = [f"ch{i}" for i in range(n_channels)]
        out_meta["channels"] = {"dim0": channel_labels, "dim1": atom_names}
        out_phi = {}
        out_phi["channels"] = {"dim0": channel_labels, "dim1": list(information_dynamics_metrics.keys())}
        out_IIT = {}
        out_IIT["channels"] = {"dim0": channel_labels, "dim1": list(IIT_metrics.keys())}

        return {"PhiID": (PhiID_vals, out_meta), "inf_dyn": (inf_dyn_vals, out_phi), "IIT": (IIT_vals, out_IIT)}
