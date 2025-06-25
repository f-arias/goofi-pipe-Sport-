import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam


class Autocorrelation(Node):
    def config_input_slots():
        return {"signal": DataType.ARRAY}

    def config_output_slots():
        return {"autocorr": DataType.ARRAY}

    def config_params():
        return {
            "autocorrelation": {
                "normalize": BoolParam(True, doc="Normalize the autocorrelation result"),
                "biased": BoolParam(False, doc="Use biased (divide by N) or unbiased (divide by N-lag) estimator"),
                "cutoff": IntParam(-1, -5000, -1, doc="Cut off the autocorrelation result at this lag (use -1 for no cutoff)"),
            },
        }

    def setup(self):
        pass

    def process(self, signal: Data):
        x = signal.data
        if x is None or len(x) == 0:
            return None

        N = len(x)
        autocorr = np.correlate(x, x, mode="full")
        autocorr = autocorr[autocorr.size // 2 :]  # keep non-negative lags

        # Apply unbiased or biased normalization
        if self.params.autocorrelation.biased.value:
            autocorr = autocorr / N
        else:
            lags = np.arange(N, 0, -1)
            autocorr = autocorr / lags

        # Normalize if requested
        if self.params.autocorrelation.normalize.value:
            autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr

        # Apply cutoff if set
        cutoff = self.params.autocorrelation.cutoff.value
        if cutoff != -1:
            autocorr = autocorr[:cutoff]

        return {"autocorr": (autocorr, signal.meta)}
