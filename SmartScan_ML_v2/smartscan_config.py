"""
Shared settings for the Smart Scan ML pipeline (v2).

One global band grid is used everywhere (training, runtime, Scilab commands):
    band b covers [50*b, 50*b + 50) MHz, centre = 50*b + 25 MHz
    b = 0 .. 321  ->  0 .. 16100 MHz  (covers every configuration, 5 MHz .. 16069 MHz)
"""
from pathlib import Path
import numpy as np

BAND_WIDTH_MHZ = 50.0
N_BANDS = 322                      # 0 .. 16100 MHz
BAND_LOWER_MHZ = np.arange(N_BANDS) * BAND_WIDTH_MHZ
BAND_CENTRE_MHZ = BAND_LOWER_MHZ + BAND_WIDTH_MHZ / 2.0
DWELL_S = 0.010

# Receiver detection rule used by the Scilab IRSA stations:
# SNR = amplitude - (-95 dBm) >= 10 dB  ->  amplitude >= -85 dBm
NOISE_FLOOR_DBM = -95.0
DETECTION_THRESHOLD_DB = 10.0

# Default configuration split (configuration-level, never row-level).
DEFAULT_TRAIN_CONFIGS = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_TEST_CONFIGS = [7, 8]

HERE = Path(__file__).resolve().parent
PREPARED_DIR = HERE / "prepared"
MODELS_DIR = HERE / "models"
REPORTS_DIR = HERE / "reports"


def band_of_centre(centre_mhz):
    """Global band index for a receiver window centre (vectorised)."""
    b = np.rint((np.asarray(centre_mhz, dtype=float) - BAND_WIDTH_MHZ / 2.0) / BAND_WIDTH_MHZ)
    return b.astype(int)


def band_of_frequency(freq_mhz):
    """Global band index containing a frequency (vectorised)."""
    b = np.floor(np.asarray(freq_mhz, dtype=float) / BAND_WIDTH_MHZ)
    return np.clip(b, 0, N_BANDS - 1).astype(int)


def parse_config_list(text):
    """'0-6' or '0,1,2' or '7 8' -> [ints]"""
    out = []
    for part in str(text).replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))

# ---------------------------------------------------------------------------
# Adaptive dwell - CUSTOM MODEL ONLY (Whittle and Round Robin always use 10 ms):
#   dwell = clip(DWELL_BASE_S * (1 + DWELL_K * H(P)), DWELL_MIN_S, DWELL_MAX_S)
#   H(P) = binary entropy (bits, 0..1) of the model's P(target) for the chosen band.
# Short look (5 ms) when the model is sure, up to 15 ms when it is unsure. Chosen on development
# configuration 6 (live Scilab) by: most transmitters found, then fastest first detection.
# The Scilab receiver (main_closed_loop.sce) applies the commanded dwell.
SPECTRA_ADAPTIVE_DWELL = True            # False = Spectra also uses the fixed 10 ms dwell. Test on unseen configs 7-11:
                                         # adaptive -> 0 false alarms; fixed 10 ms -> more detections and pulses
DWELL_BASE_S = 0.005
DWELL_K = 2.0
DWELL_MIN_S = 0.005
DWELL_MAX_S = 0.015
FIXED_DWELL_S = 0.010                    # baselines and the recorded sweep
# LPD-aware dwell (Spectra only): if the band's last detection was weak (SNR within
# LPD_WEAK_MARGIN_DB of the 10-dB detection threshold), the dwell is multiplied by (1 + LPD_DWELL_GAIN),
# still capped at DWELL_MAX_S. Weak, near-threshold signals are typical of LPD emitters.
# Physics-based dwell (Spectra only). Detectable pulses arrive at a measured rate r per second,
# so P(detect in a dwell T) = 1 - exp(-r*T). A longer dwell catches more, but costs scans, so the
# dwell is chosen to maximise discovery PER SECOND:
#     value_rate(T) = P(target) * (1 - exp(-r*T)) / (T + RETUNE_OVERHEAD_S)
# evaluated over DWELL_GRID_S. DWELL_MODEL = "rate" uses this; "entropy" is the older heuristic.
DWELL_MODEL = "entropy"
RETUNE_OVERHEAD_S = 0.0005               # receiver settling time charged to every scan
DWELL_GRID_S = (0.005, 0.0075, 0.010, 0.0125, 0.015)
LPD_WEAK_MARGIN_DB = 3.0
LPD_DWELL_GAIN = 0.0            # tested on development config 6: no gain in LPD found -> off (set e.g. 0.5 to enable)

# Failover (Task 4): PPO custom -> Whittle index
PPO_TIMEOUT_S = 0.050                    # hard per-decision limit for the PPO forward pass
PPO_TIMEOUT_ENV = "SPECTRA_PPO_TIMEOUT_MS"  # optional test setting, e.g. 1 -> real PPO timeouts -> Whittle failover
FORCE_TIER_ENV = "SPECTRA_FORCE_TIER"    # set to 1 (PPO custom) or 2 (Whittle) to force a tier (testing)
