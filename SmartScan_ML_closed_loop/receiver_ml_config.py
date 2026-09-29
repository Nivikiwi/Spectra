
from pathlib import Path
import numpy as np

# Match the IRSA receiver: 50 MHz instantaneous bandwidth.
RF_MIN_MHZ = 100.0
RF_MAX_MHZ = 12000.0
WINDOW_BW_MHZ = 50.0

# Window lower edges: 100..11950, centres: 125..11975.
WINDOW_LOWER_MHZ = np.arange(
    RF_MIN_MHZ,
    RF_MAX_MHZ - WINDOW_BW_MHZ + 1e-9,
    WINDOW_BW_MHZ,
)
WINDOW_CENTRES_MHZ = WINDOW_LOWER_MHZ + WINDOW_BW_MHZ / 2.0
N_BANDS = len(WINDOW_CENTRES_MHZ)

DEFAULT_DWELL_S = 0.010

def band_index(center_mhz: float) -> int:
    return int(np.argmin(np.abs(WINDOW_CENTRES_MHZ - float(center_mhz))))

def band_bounds(index: int):
    lo = float(WINDOW_LOWER_MHZ[index])
    return lo, lo + WINDOW_BW_MHZ, float(WINDOW_CENTRES_MHZ[index])

def normalize(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)
