import time
from collections import deque

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class DreamInceptor(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY, "start": DataType.ARRAY, "reset": DataType.ARRAY}

    def config_output_slots():
        return {
            "trigger": DataType.ARRAY,
            "z_theta_alpha": DataType.ARRAY,
            "z_lempel_ziv": DataType.ARRAY,
            "baseline_stats": DataType.TABLE,
        }

    def config_params():
        return {
            "control": {
                "start": BoolParam(False, trigger=True, doc="Start the dream inception process"),
                "reset": BoolParam(False, trigger=True, doc="Reset and stop the process"),
                "wait_time": IntParam(100, 15, 300, doc="Waiting time after trigger in seconds"),
            },
            "baseline": {
                "n_seconds": IntParam(30, 15, 300, doc="Baseline duration in seconds"),
                "method": StringParam("mean", options=["mean", "quantile"], doc="Baseline computation method"),
            },
            "features": {
                "n_features": IntParam(100, 5, 500, doc="Number of feature values to accumulate"),
                "lz_binarization": StringParam("mean", options=["mean", "median"], doc="LZ binarization method"),
            },
            "detection": {
                "threshold": FloatParam(2.0, 0.5, 5.0, doc="Theta/alpha z-score threshold"),
                "n_windows": IntParam(20, 5, 100, doc="Number of successive windows required"),
            },
        }

    def setup(self):
        # Import required libraries
        from antropy import lziv_complexity
        from scipy import signal

        self.compute_lzc = lziv_complexity
        self.signal = signal
        self.last_trigger_time = None

        # Initialize state variables
        self.reset_state()

        # read the start value once to avoid starting before the button is pressed
        # TODO: for some reason the start param is True initially, but other trigger params are not
        self.params.control.start.value

    def reset_state(self):
        """Reset all internal state variables"""
        self.is_running = False
        self.last_trigger_time = None
        self.baseline_data = []
        self.baseline_computed = False
        self.baseline_stats = {}
        self.feature_buffer = deque(maxlen=self.params.features.n_features.value if hasattr(self, "params") else 10)
        self.successive_count = 0
        self.time_origin = None

    def process(self, data: Data, start: Data = None, reset: Data = None):
        if start is not None:
            self.params.control.start.value = True
            self.input_slots["start"].clear()
        if reset is not None:
            self.params.control.reset.value = True
            self.input_slots["reset"].clear()

        if data is None or data.data is None:
            return None

        # Handle control parameters
        if self.params.control.reset.value:
            self.reset_state()
            return {"trigger": None, "z_theta_alpha": None, "z_lempel_ziv": None, "baseline_stats": None}

        if self.params.control.start.value and not self.is_running:
            self.is_running = True
            self.time_origin = time.time()
            self.baseline_data = []
            self.baseline_computed = False

        if not self.is_running:
            return {"trigger": None, "z_theta_alpha": None, "z_lempel_ziv": None, "baseline_stats": None}

        eeg_signal = np.asarray(data.data)
        assert eeg_signal.ndim == 1, "Expected 1d time series"

        send_trigger = None

        # Phase 1: Baseline computation (first minute)
        if not self.baseline_computed:
            elapsed_time = time.time() - self.time_origin

            if elapsed_time < self.params.baseline.n_seconds.value:
                # Still collecting baseline data
                self.baseline_data.extend(eeg_signal)
                return {
                    "trigger": None,
                    "z_theta_alpha": None,
                    "z_lempel_ziv": None,
                    "baseline_stats": None,
                }
            else:
                # Compute baseline statistics
                self._compute_baseline_stats()
                self.baseline_computed = True
                send_trigger = np.array(0), data.meta # 0 means baseline finished

        # Phase 2: Feature extraction and detection
        if self.baseline_computed and len(self.baseline_data) > 0:
            # Extract features from current window
            lz_complexity = self._compute_lempel_ziv(eeg_signal)
            theta_alpha_ratio = self._compute_theta_alpha_ratio(eeg_signal, data.meta)

            # Compute z-scores using baseline
            lz_zscore = self._compute_zscore(lz_complexity, "lz")
            ta_zscore = self._compute_zscore(theta_alpha_ratio, "theta_alpha")

            # Add features to buffer
            self.feature_buffer.append({"lz_zscore": lz_zscore, "ta_zscore": ta_zscore})

            # Check detection criteria
            detected = self._check_detection_criteria()

            # --- Trigger cooldown logic ---
            wait_time = self.params.control.wait_time.value
            now = time.time()
            if detected:
                if (self.last_trigger_time is None) or ((now - self.last_trigger_time) >= wait_time):
                    send_trigger = np.array(1), data.meta # 1 means incubation triggered
                    self.last_trigger_time = now  # reset cooldown
                else:
                    send_trigger = None  # within cooldown window

            baseline_stats_table = {
                "lz_mean": Data(DataType.ARRAY, np.array([self.baseline_stats["lz"]["mean"]]), {}),
                "lz_std": Data(DataType.ARRAY, np.array([self.baseline_stats["lz"]["std"]]), {}),
                "ta_mean": Data(DataType.ARRAY, np.array([self.baseline_stats["theta_alpha"]["mean"]]), {}),
                "ta_std": Data(DataType.ARRAY, np.array([self.baseline_stats["theta_alpha"]["std"]]), {}),
            }

            return {
                "trigger": send_trigger,
                "z_theta_alpha": (np.array([ta_zscore]), data.meta),
                "z_lempel_ziv": (np.array([lz_zscore]), data.meta),
                "baseline_stats": (baseline_stats_table, data.meta),
            }

    def _compute_baseline_stats(self):
        """Compute baseline statistics for z-score normalization"""
        baseline_array = np.array(self.baseline_data)

        # Compute Lempel-Ziv complexity for baseline
        lz_values = []
        window_size = min(1000, len(baseline_array) // 10)  # Adaptive window size

        for i in range(0, len(baseline_array) - window_size, window_size // 2):
            window = baseline_array[i : i + window_size]
            lz_val = self._compute_lempel_ziv(window)
            lz_values.append(lz_val)

        # Compute theta/alpha ratios for baseline
        ta_values = []
        for i in range(0, len(baseline_array) - window_size, window_size // 2):
            window = baseline_array[i : i + window_size]
            ta_val = self._compute_theta_alpha_ratio(window, {})
            ta_values.append(ta_val)

        # Store baseline statistics
        if self.params.baseline.method.value == "mean":
            self.baseline_stats = {
                "lz": {"mean": np.nanmean(lz_values), "std": np.nanstd(lz_values)},
                "theta_alpha": {"mean": np.nanmean(ta_values), "std": np.nanstd(ta_values)},
            }
        else:  # quantile method
            self.baseline_stats = {
                "lz": {"q25": np.percentile(lz_values, 25), "q75": np.percentile(lz_values, 75)},
                "theta_alpha": {"q25": np.percentile(ta_values, 25), "q75": np.percentile(ta_values, 75)},
            }

    def _compute_lempel_ziv(self, signal_data):
        """Compute Lempel-Ziv complexity"""
        if len(signal_data) == 0:
            return 0.0

        # Binarize signal
        if self.params.features.lz_binarization.value == "mean":
            binarized = signal_data > np.nanmean(signal_data)
        else:  # median
            binarized = signal_data > np.nanmedian(signal_data)

        # Compute LZ complexity
        try:
            lzc = self.compute_lzc(binarized, normalize=True)
            return float(lzc)
        except:
            return 0.0

    def _compute_theta_alpha_ratio(self, signal_data, meta):
        """Compute theta/alpha power ratio"""
        if len(signal_data) < 100:  # Need minimum samples for FFT
            print("Data too short")
            return 0.0

        signal_data = signal_data[~np.isnan(signal_data)]
        # Get sampling frequency from metadata or use default
        fs = meta.get("sfreq", 256.0) if isinstance(meta, dict) else 256.0
        # Compute power spectral density
        freqs, psd = self.signal.welch(
            signal_data, fs=fs, nperseg=min(512, len(signal_data) // 4), noverlap=min(400, len(signal_data) // 5)
        )
        # Define frequency bands
        theta_band = (4, 8)
        alpha_band = (8, 12)

        # Extract power in each band
        theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])

        theta_power = np.nansum(psd[theta_mask])
        alpha_power = np.nansum(psd[alpha_mask])
        # Compute ratio (avoid division by zero)
        if alpha_power > 1e-10:
            return theta_power / alpha_power
        else:
            return 0.0

    def _compute_zscore(self, value, feature_type):
        """Compute z-score using baseline statistics"""
        if feature_type not in self.baseline_stats:
            return 0.0

        stats = self.baseline_stats[feature_type]

        if self.params.baseline.method.value == "mean":
            mean = stats["mean"]
            std = stats["std"]
            return (value - mean) / (std + 1e-8)
        else:  # quantile method
            q25, q75 = stats["q25"], stats["q75"]
            iqr = q75 - q25
            median = (q25 + q75) / 2
            return (value - median) / (iqr + 1e-8)

    def _check_detection_criteria(self):
        """Check if detection criteria are met"""
        if len(self.feature_buffer) < self.params.detection.n_windows.value:
            return 0

        # Check last n_windows for threshold crossing
        recent_features = list(self.feature_buffer)[-self.params.detection.n_windows.value :]
        threshold = self.params.detection.threshold.value

        # Count successive windows above threshold
        successive_above = 0
        for features in reversed(recent_features):
            if features["ta_zscore"] > threshold:
                successive_above += 1
            else:
                break

        if successive_above >= self.params.detection.n_windows.value:
            return 1
        else:
            return 0
