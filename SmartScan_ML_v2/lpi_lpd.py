"""
LPI / LPD - definitions and receiver-side features (shared by step 6 and the dashboard).

LABELS (ground truth, used ONLY to train and to score - never as a model input)
  LPD  (Low Probability of Detection):
       a real transmitter whose pulses reach the IRSA receiver above the detection
       threshold (SNR >= 10 dB, i.e. amplitude >= -85 dBm) less than 10 % of the time.
  LPI  (Low Probability of Intercept):
       low peak power (power_w < LPI_POWER_W) AND an agile waveform
       (frequency hopping / random frequency, OR jittered / staggered / sliding PRI).

MODEL INPUTS (what the receiver measures, per transmitter it has intercepted)
  Pulse descriptors of the pulses the receiver detected: time of arrival, frequency,
  pulse width, amplitude/SNR. From these: SNR level and margin above threshold,
  frequency agility, pulse-width behaviour, PRI (pulse interval) regularity.
"""
import numpy as np

from smartscan_config import NOISE_FLOOR_DBM, DETECTION_THRESHOLD_DB

LPI_POWER_W = 10_000.0          # "low peak power" threshold (change here if needed)
LPD_DETECT_FRACTION = 0.10      # detectable pulse fraction below this -> LPD
MIN_PULSES = 3                  # the receiver needs at least this many pulses to classify
AGILE_FREQ_MODES = {"HoppingLinear", "HoppingSawtooth", "RandomRange", "RandomFixed"}
AGILE_PRI_MODES = {"Jittered", "Staggered", "Sliding"}

LPI_FEATURE_NAMES = [
    "log_pulses", "snr_mean", "snr_max", "snr_min", "snr_std", "frac_near_threshold",
    "freq_std_mhz", "freq_range_mhz", "freq_unique_ratio", "freq_rel_std",
    "pw_mean_us", "pw_rel_std", "pri_median_log_us", "pri_cv", "pri_irregular_frac",
]


def label_lpi(power_w, frequency_mode, pri_mode):
    agile = (frequency_mode in AGILE_FREQ_MODES) or (pri_mode in AGILE_PRI_MODES)
    return int(float(power_w) < LPI_POWER_W and agile)


def label_lpd(detectable_fraction):
    return int(detectable_fraction < LPD_DETECT_FRACTION)


def pulse_features(toa_s, freq_mhz, pw_us, amp_dbm):
    """Features from the pulses the receiver detected from ONE transmitter.
    Arrays must have >= MIN_PULSES elements. Returns a float32 vector (LPI_FEATURE_NAMES)."""
    toa_s = np.asarray(toa_s, float)
    order = np.argsort(toa_s)
    toa_s = toa_s[order]
    f = np.asarray(freq_mhz, float)[order]
    pw = np.asarray(pw_us, float)[order]
    snr = np.asarray(amp_dbm, float)[order] - NOISE_FLOOR_DBM
    n = len(toa_s)
    fm = max(abs(f.mean()), 1e-6)
    pri = np.diff(toa_s) * 1e6                           # microseconds
    pri = pri[pri > 0]
    if len(pri) >= 2:
        med = np.median(pri)
        pri_cv = float(np.std(pri) / max(np.mean(pri), 1e-9))
        irregular = float(np.mean(np.abs(pri - med) > 0.05 * med))
        pri_med_log = float(np.log1p(med))
    else:
        pri_cv, irregular, pri_med_log = 0.0, 0.0, float(np.log1p(pri[0])) if len(pri) else 0.0
    return np.array([
        np.log1p(n), snr.mean(), snr.max(), snr.min(), snr.std(),
        np.mean(snr < DETECTION_THRESHOLD_DB + 3.0),
        f.std(), f.max() - f.min(), len(np.unique(np.round(f, 0))) / n, f.std() / fm,
        pw.mean(), pw.std() / max(abs(pw.mean()), 1e-9),
        pri_med_log, pri_cv, irregular,
    ], dtype=np.float32)
