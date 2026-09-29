"""
D - Frequency mode of an observed emitter, from RECEIVER OBSERVATIONS ONLY (rule R3).

Input: the carrier frequencies (and arrival times) of the pulse descriptors the receiver has
intercepted from one transmitter. No emitter truth is read here.

Modes
  FIXED              one carrier, jitter only
  FIXED_MULTI        several carriers emitted at the same time (simultaneous pulses)
  FREQUENCY_HOPPING  a small set of carriers revisited (a hop pattern)
  SWEEP_CHIRP        carrier moves monotonically, a ramp
  AGILE_RANDOM       carrier changes widely with no repeating set and no ramp
  UNKNOWN            fewer than MIN_PULSES pulses - not enough evidence yet

Confidence is 0..1 and grows with the number of pulses seen, so an early guess is
reported as an early guess.
"""
import numpy as np

MIN_PULSES = 6                 # below this: UNKNOWN
FIXED_REL_SPREAD = 0.010       # spread within 1 % of the mean -> one carrier with jitter
CONTINUUM_REL_SPREAD = 0.05    # single group wider than 5 % of the mean = a continuum, not a carrier
SIMULTANEOUS_S = 2e-6          # pulses closer than this are treated as emitted together
SIMULTANEOUS_SHARE = 0.15      # share of such pairs on DIFFERENT channels -> multi-carrier
HOP_TOLERANCE_MHZ = 5.0        # carriers within this are treated as the same hop channel
MAX_HOP_CHANNELS = 12          # more distinct channels than this -> not a hop set
SWEEP_MONOTONIC = 0.75         # share of steps in one direction that counts as a ramp
MODES = ("FIXED", "FIXED_MULTI", "FREQUENCY_HOPPING", "SWEEP_CHIRP", "AGILE_RANDOM", "UNKNOWN")


def _channels(freqs):
    """Group carriers into channels: sorted values within HOP_TOLERANCE_MHZ of each other."""
    s = np.sort(freqs)
    cuts = np.flatnonzero(np.diff(s) > HOP_TOLERANCE_MHZ)
    groups = np.split(s, cuts + 1)
    return np.array([g.mean() for g in groups]), np.array([len(g) for g in groups])


def classify(freq_mhz, toa_s=None):
    """Return (mode, confidence, evidence dict). freq_mhz: carriers of the intercepted pulses."""
    f = np.asarray([x for x in np.asarray(freq_mhz, dtype=float).ravel() if np.isfinite(x)])
    n = len(f)
    ev = {"pulses": int(n)}
    if n < MIN_PULSES:
        return "UNKNOWN", 0.0, ev

    t = None
    if toa_s is not None:
        t = np.asarray(toa_s, dtype=float).ravel()
        if len(t) == n:
            order = np.argsort(t)
            f, t = f[order], t[order]                   # chronological, needed for the ramp test
        else:
            t = None

    mean = float(np.mean(f))
    spread = float(np.max(f) - np.min(f))
    rel = spread / max(abs(mean), 1e-9)
    ev.update({"mean_mhz": round(mean, 3), "span_mhz": round(spread, 3), "rel_spread": round(rel, 6)})
    # more pulses -> more confidence, saturating around 40 pulses
    base = float(np.clip(0.45 + 0.55 * (n - MIN_PULSES) / 34.0, 0.45, 1.0))

    if rel <= FIXED_REL_SPREAD:
        return "FIXED", base, ev

    centres, counts = _channels(f)
    k = len(centres)
    if k == 1:
        # one group of carriers: a jittery fixed emitter if the span is modest, but a
        # continuous spread over a wide range is an agile emitter, not a carrier.
        if rel <= CONTINUUM_REL_SPREAD:
            return "FIXED", round(base * 0.9, 3), ev
        return "AGILE_RANDOM", round(base * 0.8, 3), ev

    # Several carriers present at the SAME instant = a multi-carrier fixed emitter,
    # not a hopper (a hopper visits its channels one after another).
    if t is not None and n >= 4:
        dt = np.diff(t)
        together = dt <= SIMULTANEOUS_S
        if together.any():
            diff_channel = together & (np.abs(np.diff(f)) > HOP_TOLERANCE_MHZ)
            share = float(diff_channel.sum()) / float(together.sum())
            ev["simultaneous_diff_channel"] = round(share, 3)
            if share >= SIMULTANEOUS_SHARE:
                return "FIXED_MULTI", round(base * 0.9, 3), ev
    revisited = int(np.sum(counts > 1))
    ev.update({"channels": int(k), "revisited_channels": revisited})

    steps = np.diff(f)
    moving = steps[np.abs(steps) > HOP_TOLERANCE_MHZ]
    monotonic = float(np.mean(moving > 0)) if len(moving) else 0.0
    monotonic = max(monotonic, 1.0 - monotonic)         # a ramp in either direction
    ev["monotonic_share"] = round(monotonic, 3)

    if len(moving) >= 3 and monotonic >= SWEEP_MONOTONIC and k > MAX_HOP_CHANNELS:
        return "SWEEP_CHIRP", round(base * monotonic, 3), ev
    if 2 <= k <= MAX_HOP_CHANNELS and revisited >= 1:
        # a small set of channels, used more than once each = a hop pattern
        return "FREQUENCY_HOPPING", round(base * min(1.0, revisited / max(k - 1, 1)), 3), ev
    return "AGILE_RANDOM", round(base * 0.8, 3), ev


class FreqModeTracker:
    """Keeps the intercepted carriers per transmitter and re-classifies as evidence arrives (D2)."""

    MAX_KEEP = 400

    def __init__(self):
        self.pulses = {}
        self.result = {}

    def reset(self):
        self.pulses.clear()
        self.result.clear()

    def add(self, key, freqs, toas=None):
        f, t = self.pulses.setdefault(key, ([], []))
        f.extend(float(x) for x in np.asarray(freqs).ravel())
        if toas is not None:
            t.extend(float(x) for x in np.asarray(toas).ravel())
        del f[:-self.MAX_KEEP]
        del t[:-self.MAX_KEEP]
        self.result[key] = classify(f, t if len(t) == len(f) else None)
        return self.result[key]

    def get(self, key):
        return self.result.get(key, ("UNKNOWN", 0.0, {"pulses": 0}))

    def table(self):
        return [{"key": k, "mode": v[0], "confidence": v[1], "pulses": v[2].get("pulses", 0)}
                for k, v in sorted(self.result.items())]
