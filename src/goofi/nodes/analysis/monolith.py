import os
from os.path import join

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam


class Monolith(Node):

    def config_params():
        return {
            "monolith": {
                "toto": BoolParam(False, doc="Whether to include Toto embedings"),
                "pyspi": BoolParam(False, doc="Whether to compute SPI features"),
            }
        }

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"features": DataType.ARRAY}

    def setup(self):
        pass

    def process(self, data: Data):
        assert "sfreq" in data.meta, "Data must have a 'sfreq' (sampling frequency) in its metadata."
        sfreq = data.meta["sfreq"]
        data_arr = data.data

        feat_fns = FEAT_FNS.copy()
        if not self.params.monolith.pyspi.value:
            feat_fns.pop("spi", None)
        if not self.params.monolith.toto.value:
            feat_fns.pop("toto", None)

        # Preprocess the data
        data_arr = preprocess(data_arr, sfreq)

        # compute channel-wise features
        features = []
        for feat_name, feat_fn in feat_fns.items():
            if feat_name in NON_CHANNEL_WISE_FEATS:
                ft = feat_fn(data_arr, sfreq)
                if isinstance(ft, np.ndarray) or isinstance(ft, list):
                    features.append(ft)
                else:
                    features.append([ft])
            else:
                for channel_data in data_arr:
                    ft = feat_fn(channel_data, sfreq)
                    if isinstance(ft, np.ndarray) or isinstance(ft, list):
                        features.append(ft)
                    else:
                        features.append([ft])

        features = np.concatenate(features, axis=-1)
        meta = data.meta.copy()
        del meta["channels"]
        return {"features": (features, meta)}


from scipy.signal import butter, filtfilt, iirnotch, welch


def preprocess(data: np.ndarray, sfreq: float):
    # Apply bandpass filter (3-30 Hz)
    nyquist = sfreq / 2
    low_cut = 3 / nyquist
    high_cut = 30 / nyquist
    b, a = butter(4, [low_cut, high_cut], btype="band")
    data = filtfilt(b, a, data, axis=-1)

    # Apply notch filters for 50Hz and harmonics
    for freq in np.arange(50, 101, 50):
        if freq < nyquist:
            b, a = iirnotch(freq, Q=30, fs=sfreq)
            data = filtfilt(b, a, data, axis=-1)

    # Clip values
    data = np.clip(data, -150, 150)

    # Standardize per channel
    data = (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)

    return data


import tempfile

import neurokit2 as nk2
import numpy as np
from antropy import detrended_fluctuation, higuchi_fd, lziv_complexity, petrosian_fd
from fooof import FOOOF
from phyid.calculate import calc_PhiID
from pyspi.calculator import Calculator
from scipy.stats import kurtosis, skew
from toto.inference.embedding import embed as embed_toto


def mean_amplitude(x, sfreq):
    return np.mean(x)


def std_amplitude(x, sfreq):
    return np.std(x)


def skewness(x, sfreq):
    return skew(x)


def kurt(x, sfreq):
    return kurtosis(x)


def zero_crossings(x, sfreq):
    return ((x[:-1] * x[1:]) < 0).sum()


def hjorth_activity(x, sfreq):
    return np.var(x)


def hjorth_mobility(x, sfreq):
    return np.sqrt(np.var(np.diff(x)) / np.var(x))


def hjorth_complexity(x, sfreq):
    diff1 = np.diff(x)
    diff2 = np.diff(diff1)
    return (np.sqrt(np.var(diff2) / np.var(diff1))) / (np.sqrt(np.var(diff1) / np.var(x)))


def spectral(x, sfreq):
    f, Pxx = welch(x, sfreq)
    mean_frequency = np.sum(f * Pxx) / np.sum(Pxx)
    delta = np.trapezoid(Pxx[(f >= 0) & (f <= 4)], f[(f >= 0) & (f <= 4)])
    theta = np.trapezoid(Pxx[(f > 4) & (f <= 8)], f[(f > 4) & (f <= 8)])
    alpha = np.trapezoid(Pxx[(f > 8) & (f <= 12)], f[(f > 8) & (f <= 12)])
    beta = np.trapezoid(Pxx[(f > 12) & (f <= 30)], f[(f > 12) & (f <= 30)])
    gamma = np.trapezoid(Pxx[(f > 30) & (f <= 45)], f[(f > 30) & (f <= 45)])

    model = FOOOF(peak_width_limits=(2, 12))
    model.fit(f, Pxx, freq_range=(3, 30))

    return np.array([delta, theta, alpha, beta, gamma, mean_frequency] + list(model.aperiodic_params_))


def binarize_by_mean(x):
    return (x > np.mean(x)).astype(int)


def entropy_shannon(x, sfreq):
    bin_ts = binarize_by_mean(x)
    return nk2.entropy_shannon(bin_ts, base=2)[0]


def entropy_renyi(x, sfreq):
    bin_ts = binarize_by_mean(x)
    return nk2.entropy_renyi(bin_ts, alpha=2)[0]


def entropy_approximate(x, sfreq):
    return nk2.entropy_approximate(x, delay=1, dimension=2, tolerance="sd", Corrected=True)[0]


def entropy_sample(x, sfreq):
    return nk2.entropy_sample(x, delay=1, dimension=2, tolerance="sd")[0]


def entropy_rate(x, sfreq):
    return nk2.entropy_rate(x, kmax=10, symbolize="mean")[0]


def entropy_permutation(x, sfreq):
    return nk2.entropy_permutation(x, delay=1, dimension=2, corrected=True, weighted=False, conditional=False)[0]


def entropy_permutation_weighted(x, sfreq):
    return nk2.entropy_permutation(x, delay=1, dimension=2, corrected=True, weighted=True, conditional=False)[0]


def entropy_permutation_conditional(x, sfreq):
    return nk2.entropy_permutation(x, delay=1, dimension=2, corrected=True, weighted=False, conditional=True)[0]


def entropy_permutation_weighted_conditional(x, sfreq):
    return nk2.entropy_permutation(x, delay=1, dimension=2, corrected=True, weighted=True, conditional=True)[0]


def entropy_multiscale(x, sfreq):
    return nk2.entropy_multiscale(x, dimension=2, tolerance="sd", method="MSPEn")[0]


def entropy_bubble(x, sfreq):
    return nk2.entropy_bubble(x, delay=1, dimension=2, alpha=2, tolerance="sd")[0]


def entropy_svd(x, sfreq):
    return nk2.entropy_svd(x, delay=1, dimension=2)[0]


def entropy_attention(x, sfreq):
    return nk2.entropy_attention(x)[0]


def entropy_dispersion(x, sfreq):
    return nk2.entropy_dispersion(x, delay=1, dimension=2, c=6, symbolize="NCDF")[0]


def compute_spi_features(data, subset="fast"):
    # Create a temporary file with the config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(PYSPI_FAST_CONFIG)
        pyspi_fast_config = f.name

    calc = Calculator(data, subset=subset, configfile=pyspi_fast_config)
    calc.compute()
    s = calc.table.stack(list(range(calc.table.columns.nlevels))).dropna()
    print(s, s.values)
    return s.values


def phiid_pairwise_metrics(x, sfreq):
    """
    Compute all PhiID atom metrics for all pairs of channels (asymmetric, src->tgt and tgt->src).
    Returns a flattened array of all atom means for each channel pair and direction.
    """
    data = np.atleast_2d(x)
    n_channels, n_time = data.shape
    tau = 5
    kind = "gaussian"
    redundancy = "MMI"
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
    # For each ordered pair (i, j), i != j
    results = []
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue
            src = data[i]
            trg = data[j]
            atoms_res, _ = calc_PhiID(src, trg, tau, kind=kind, redundancy=redundancy)
            atom_means = [float(np.mean(atoms_res[name])) for name in atom_names]
            results.append(atom_means)
    return np.array(results, dtype=np.float32).flatten()


FEAT_FNS = {
    "detrended_fluctuation": lambda x, sfreq: detrended_fluctuation(x),
    "higuchi_fd": lambda x, sfreq: higuchi_fd(x),
    "lziv_complexity": lambda x, sfreq: lziv_complexity(x > x.mean(), normalize=True),
    "petrosian_fd": lambda x, sfreq: petrosian_fd(x),
    "mean_amplitude": mean_amplitude,
    "std_amplitude": std_amplitude,
    "skewness": skewness,
    "kurtosis": kurt,
    "zero_crossings": zero_crossings,
    "hjorth_activity": hjorth_activity,
    "hjorth_mobility": hjorth_mobility,
    "hjorth_complexity": hjorth_complexity,
    "spectral": spectral,
    "entropy_shannon": entropy_shannon,
    "entropy_renyi": entropy_renyi,
    # "entropy_approximate": entropy_approximate,
    # "entropy_sample": entropy_sample,
    "entropy_rate": entropy_rate,
    # "entropy_permutation": entropy_permutation,
    # "entropy_permutation_weighted": entropy_permutation_weighted,
    # "entropy_permutation_conditional": entropy_permutation_conditional,
    # "entropy_permutation_weighted_conditional": entropy_permutation_weighted_conditional,
    # "entropy_multiscale": entropy_multiscale,
    "entropy_bubble": entropy_bubble,
    "entropy_svd": entropy_svd,
    "entropy_attention": entropy_attention,
    "entropy_dispersion": entropy_dispersion,
    "spi": compute_spi_features,
    "phiid": phiid_pairwise_metrics,
    "toto": lambda x, sfreq: embed_toto(x, global_average=True),
}

BATCHED_FEATS = ["toto"]
NON_CHANNEL_WISE_FEATS = ["spi", "phiid", "toto"]

PYSPI_FAST_CONFIG = """
.statistics.basic:
  Covariance:
    labels:
    - undirected
    - linear
    - signed
    - multivariate
    - contemporaneous
    dependencies: null
    configs:
    - estimator: ShrunkCovariance
      squared: true
  Precision:
    labels:
    - undirected
    - linear
    - signed
    - multivariate
    - contemporaneous
    dependencies: null
    configs:
    - estimator: ShrunkCovariance
      squared: true
  SpearmanR:
    labels:
    - undirected
    - nonlinear
    - signed
    - bivariate
    - contemporaneous
    dependencies: null
    configs:
    - squared: false
  KendallTau:
    labels:
    - undirected
    - nonlinear
    - signed
    - bivariate
    - contemporaneous
    dependencies: null
    configs:
    - squared: false
  CrossCorrelation:
    labels:
    - undirected
    - linear
    - signed/unsigned
    - bivariate
    - time-dependent
    dependencies: null
    configs:
    - statistic: mean
      squared: true
      sigonly: false
.statistics.distance:
  PairwiseDistance:
    labels:
    - unsigned
    - unordered
    - nonlinear
    - undirected
    dependencies: null
    configs:
    - metric: braycurtis
  DynamicTimeWarping:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - time-dependent
    dependencies: null
    configs:
    - global_constraint: sakoe_chiba
  LongestCommonSubsequence:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - time-dependent
    dependencies: null
    configs:
    - global_constraint: sakoe_chiba
.statistics.infotheory:
  JointEntropy:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - contemporaneous
    dependencies:
    - java
    configs:
    - estimator: kernel
  ConditionalEntropy:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - contemporaneous
    dependencies:
    - java
    configs:
    - estimator: kernel
  CrossmapEntropy:
    labels:
    - unsigned
    - directed
    - time-dependent
    - bivariate
    dependencies:
    - java
    configs:
    - estimator: gaussian
      history_length: 10
  StochasticInteraction:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - time-dependent
    dependencies:
    - java
    configs:
    - estimator: gaussian
  MutualInfo:
    labels:
    - undirected
    - nonlinear
    - unsigned
    - bivariate
    - contemporaneous
    dependencies:
    - java
    configs:
    - estimator: gaussian
  TimeLaggedMutualInfo:
    labels:
    - directed
    - nonlinear
    - unsigned
    - bivariate
    - time-dependent
    dependencies:
    - java
    configs:
    - estimator: gaussian
.statistics.spectral:
  CoherencePhase:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  CoherenceMagnitude:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  ImaginaryCoherence:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  PhaseSlopeIndex:
    labels:
    - directed
    - linear/nonlinear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
  PhaseLockingValue:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  PhaseLagIndex:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  WeightedPhaseLagIndex:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  DebiasedSquaredPhaseLagIndex:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  DebiasedSquaredWeightedPhaseLagIndex:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  PairwisePhaseConsistency:
    labels:
    - undirected
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: max
  GroupDelay:
    labels:
    - directed
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      statistic: delay
  SpectralGrangerCausality:
    labels:
    - directed
    - linear
    - unsigned
    - bivariate
    - frequency-dependent
    dependencies: null
    configs:
    - fmin: 0.25
      fmax: 0.5
      order: 20
      method: parametric
      statistic: max
"""
