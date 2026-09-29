"""
Scoring of scan decisions with ANY dwell length (fixed 10 ms or adaptive).

A scan = (start time, dwell length, band, receiver hit, PDWs intercepted).
Ground truth comes from the pulse list stored by step 1 (prepared/config_<id>.npz):
  * truly active   : a real transmitter (id > 0) sent a pulse in that band during the scan
  * detectable     : that pulse reached the receiver above threshold (SNR >= 10 dB)
All time-based metrics use simulation seconds, so strategies with different dwell
lengths are compared on the same clock. Ground truth is used for scoring only.
"""
import numpy as np

from smartscan_config import NOISE_FLOOR_DBM, DETECTION_THRESHOLD_DB, BAND_WIDTH_MHZ

THRESH_DBM = NOISE_FLOOR_DBM + DETECTION_THRESHOLD_DB


class ConfigTruth:
    def __init__(self, grid):
        g = grid
        order = np.argsort(g["ev_time"], kind="stable")
        self.time = g["ev_time"][order]
        self.band = g["ev_band"][order].astype(np.int64)
        self.tx = g["ev_tx"][order].astype(np.int64)
        self.amp = g["ev_amp"][order]
        self.freq = g["ev_freq"][order] if "ev_freq" in g else None
        self.pw = g["ev_pw"][order] if "ev_pw" in g else None
        self.end_s = float(self.time[-1]) if len(self.time) else 0.0
        det = (self.tx > 0) & (self.amp >= THRESH_DBM)
        self.first_detectable = {}
        for t, x in zip(self.time[det], self.tx[det]):
            self.first_detectable.setdefault(int(x), float(t))
        self.lpi, self.lpd = {}, {}
        if "tx_ids" in g:
            self.lpi = {int(t): int(v) for t, v in zip(g["tx_ids"], g["tx_lpi"])}
            self.lpd = {int(t): int(v) for t, v in zip(g["tx_ids"], g["tx_lpd"])}

    def detectable_pulses(self, start_s, dwell_s, band):
        """Indices (into this object's arrays) of pulses the receiver can detect in this scan."""
        i0 = np.searchsorted(self.time, start_s - 5e-10, side="left")
        i1 = np.searchsorted(self.time, start_s + dwell_s + 5e-10, side="right")
        idx = np.arange(i0, i1)
        return idx[(self.band[i0:i1] == band) & (self.tx[i0:i1] > 0) & (self.amp[i0:i1] >= THRESH_DBM)]

    def window(self, start_s, dwell_s, band):
        """(truly_active, set of transmitters detectable in this scan window)."""
        i0 = np.searchsorted(self.time, start_s - 5e-10, side="left")
        i1 = np.searchsorted(self.time, start_s + dwell_s + 5e-10, side="right")
        sel = self.band[i0:i1] == band
        tx = self.tx[i0:i1][sel]
        amp = self.amp[i0:i1][sel]
        active = bool((tx > 0).any())
        det = set(int(x) for x in tx[(tx > 0) & (amp >= THRESH_DBM)])
        return active, det


def score_config(truth, scans):
    """scans: iterable of (start_s, dwell_s, band, hit, pdws). Returns a dict of raw counts for one config."""
    found = {}
    n = hits = tp = fp = tn = fn = 0
    pdws = 0.0
    listen = 0.0
    for start, dwell, band, hit, p in scans:
        n += 1
        hits += int(hit)
        pdws += float(p) if np.isfinite(p) else 0.0
        listen += float(dwell)
        active, det = truth.window(start, dwell, int(band))
        if active and hit:
            tp += 1
        elif active:
            fn += 1
        elif hit:
            fp += 1
        else:
            tn += 1
        if hit:
            for x in det:
                found.setdefault(x, float(start))
    delays = [(found[x] if x in found else truth.end_s) - t0 for x, t0 in truth.first_detectable.items()]
    lpi_all = [x for x in truth.first_detectable if truth.lpi.get(x) == 1]
    lpd_all = [x for x in truth.first_detectable if truth.lpd.get(x) == 1]
    return {"scans": n, "hits": hits, "pdws": pdws, "listen_s": listen, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "found": len(found), "detectable": len(truth.first_detectable), "delays": delays,
            "lpi_found": sum(1 for x in lpi_all if x in found), "lpi_detectable": len(lpi_all),
            "lpd_found": sum(1 for x in lpd_all if x in found), "lpd_detectable": len(lpd_all),
            "sim_s": truth.end_s}


def combine(parts):
    """Sum per-config counts and compute the comparison metrics."""
    tot = {k: 0 for k in ("scans", "hits", "pdws", "listen_s", "tp", "fp", "tn", "fn", "found", "detectable",
                          "lpi_found", "lpi_detectable", "lpd_found", "lpd_detectable", "sim_s")}
    delays = []
    for p in parts:
        for k in tot:
            tot[k] += p[k]
        delays += p["delays"]
    n = max(tot["scans"], 1)
    return {
        "scans": tot["scans"], "detections": tot["hits"],
        "detection_rate_pct": 100.0 * tot["hits"] / n,
        "detections_per_s": tot["hits"] / max(tot["sim_s"], 1e-9),
        "pdws_intercepted": tot["pdws"],
        "transmitters_found": tot["found"], "transmitters_detectable": tot["detectable"],
        "coverage_pct": 100.0 * tot["found"] / max(tot["detectable"], 1),
        "time_to_first_detection_s": float(np.mean(delays)) if delays else 0.0,
        "pd_pct": 100.0 * tot["tp"] / max(tot["tp"] + tot["fn"], 1),
        "pfa_pct": 100.0 * tot["fp"] / max(tot["fp"] + tot["tn"], 1),
        "lpi_found": tot["lpi_found"], "lpi_detectable": tot["lpi_detectable"],
        "lpd_found": tot["lpd_found"], "lpd_detectable": tot["lpd_detectable"],
        "mean_dwell_ms": 1000.0 * tot["listen_s"] / n,
    }
