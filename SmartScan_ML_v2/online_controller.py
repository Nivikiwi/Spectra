"""
LIVE CLOSED LOOP with Scilab (main_closed_loop.sce).

    Scilab receiver --observation--> ml_observation_closed_loop.csv
         ^                                   |
         |                                   v
    ml_scan_command.txt <--decision-- this controller (custom | whittle | round_robin | greedy)

Start Scilab main_closed_loop.sce FIRST, then this script.
At the end it writes reports/live_history_<policy>_config<id>.csv and
reports/live_audit_<policy>_config<id>.xlsx
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from smartscan_config import (MODELS_DIR, REPORTS_DIR, PREPARED_DIR, BAND_CENTRE_MHZ, DWELL_S,
                              band_of_centre)
from smartscan_core import Controller, load_grid
from decision_tiers import TieredDecider, binary_entropy_bits
from smartscan_config import FIXED_DWELL_S

POLL_S = 0.002


def read_status(path):
    try:
        return path.read_text().strip().split()
    except (FileNotFoundError, PermissionError, OSError):
        return []


def write_command(path, step, band, dwell_s=DWELL_S):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(f"{step} {BAND_CENTRE_MHZ[band]:.6f} {dwell_s:.6f}\n")
    for _ in range(500):                      # Windows: retry if Scilab is reading the file
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.002)
    raise RuntimeError(f"Could not write {path}")


class CsvFollower:
    """Reads only NEW complete lines of a growing CSV (never re-reads the whole file)."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.buffer = ""
        self.header = None

    def new_rows(self):
        try:
            with open(self.path, "rb") as f:          # binary: exact byte offsets on Windows
                f.seek(self.offset)
                chunk = f.read()
        except (FileNotFoundError, PermissionError):
            return []
        # only consume complete lines; keep a partial last line for the next read
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return []
        self.offset += cut + 1
        chunk = chunk[:cut + 1].decode("utf-8", errors="replace")
        self.buffer += chunk
        lines = self.buffer.split("\n")
        self.buffer = lines.pop()             # incomplete last line stays in the buffer
        rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if self.header is None:
                self.header = line.split(",")
                continue
            vals = line.split(",")
            if len(vals) != len(self.header):
                continue
            rows.append(dict(zip(self.header, vals)))
        return rows


def to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


# The live history file has exactly these 10 columns.
HISTORY_COLUMNS = ["config_id", "dwell", "time_s", "dwell_s", "centre_mhz", "decision", "p_target", "confidence",
                   "receiver_hit", "pdws"]


def decide_with_telemetry(ctrl, policy, t, now_s=None):
    """Scheduling step: choose the band; adaptive dwell only for the custom model (Spectra)."""
    try:
        band, score = ctrl.decide(t, now_s)
    except TypeError:
        band, score = ctrl.decide(t)
    info = getattr(ctrl, "last_info", None)
    if info is not None:                               # custom (tiered decider)
        return band, score, {k: info[k] for k in ("p_target", "confidence", "dwell_s", "decision")}
    p = score if (score is not None and np.isfinite(score)) else float("nan")
    h = binary_entropy_bits(p)
    return band, score, {"p_target": p, "confidence": (1.0 - h) if np.isfinite(h) else float("nan"),
                         "dwell_s": float(FIXED_DWELL_S), "decision": f"POLICY_{policy.upper()}"}


def live_audit(hist, policy, cfg, out_xlsx):
    """Audit of the live run (works for fixed and adaptive dwell). Ground truth from
    prepared/config_<id>.npz is used for scoring only."""
    from scan_metrics import ConfigTruth, score_config, combine
    summary = {"policy": policy, "config_id": cfg, "scans": len(hist),
               "detections": int(hist.receiver_hit.sum()),
               "detection_rate_pct": round(float(100 * hist.receiver_hit.mean()), 2) if len(hist) else 0.0}
    npz = PREPARED_DIR / f"config_{cfg}.npz"
    if npz.exists() and len(hist):
        truth = ConfigTruth(load_grid(npz))
        bands = band_of_centre(hist.centre_mhz.to_numpy())
        m = combine([score_config(truth, zip(hist.time_s, hist.dwell_s, bands, hist.receiver_hit, hist.pdws))])
        summary.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    band_stats = (hist.groupby("centre_mhz").agg(scans=("receiver_hit", "size"), detections=("receiver_hit", "sum"),
                  mean_dwell_ms=("dwell_s", lambda x: 1000 * x.mean())).reset_index())
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        pd.DataFrame([summary]).to_excel(w, sheet_name="Summary", index=False)
        hist.to_excel(w, sheet_name="Scan_History", index=False)
        band_stats.to_excel(w, sheet_name="Band_Stats", index=False)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Live Smart Scan controller for Scilab")
    ap.add_argument("--scilab-dir", required=True, help="folder containing main_closed_loop.sce")
    ap.add_argument("--policy", default="custom", choices=list(Controller.POLICIES))
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument("--wait-s", type=float, default=600, help="how long to wait for Scilab READY")
    ap.add_argument("--force-tier", default=None,
                    help="custom policy only: force tier 1 (PPO custom) or 2 (Whittle) for testing; "
                         "default: environment variable SPECTRA_FORCE_TIER, else automatic failover")
    args = ap.parse_args()

    sdir = Path(args.scilab_dir)
    obs_path = sdir / "ml_observation_closed_loop.csv"
    cmd_path = sdir / "ml_scan_command.txt"
    status_path = sdir / "ml_closed_loop_status.txt"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.policy == "custom":
        ctrl = TieredDecider(args.models_dir, force_tier=args.force_tier)
        print(f"Custom policy with failover (PPO custom -> Whittle), adaptive dwell"
              f"{'; FORCED tier ' + str(ctrl.force_tier) if ctrl.force_tier else ''}.")
    else:
        ctrl = Controller(args.policy, args.models_dir)
    print(f"Policy: {args.policy}. Waiting for Scilab READY in {status_path} ...")
    t0 = time.time()
    while True:
        st = read_status(status_path)
        if st and st[0] == "READY":
            break
        if time.time() - t0 > args.wait_s:
            raise SystemExit("Scilab did not become READY. Start main_closed_loop.sce first.")
        time.sleep(0.2)
    cfg = int(st[1]) if len(st) > 1 else -1
    print(f"Scilab READY (configuration {cfg}). Running closed loop ...")

    follower = CsvFollower(obs_path)
    history = []
    pending = {}                              # step -> (band, score, telemetry)
    next_step = 1
    band, score, info = decide_with_telemetry(ctrl, args.policy, 0)
    pending[1] = (band, score, info)
    write_command(cmd_path, 1, band, info["dwell_s"])
    last_activity = time.time()
    done_seen = False

    while True:
        rows = follower.new_rows()
        for r in rows:
            step = int(to_float(r["step"]))
            if step not in pending:
                continue
            band_chosen, score, info = pending.pop(step)
            band_seen = int(band_of_centre(to_float(r["scan_center_mhz"])))
            if band_seen != band_chosen:
                print(f"WARNING: step {step}: Scilab scanned band {band_seen}, expected {band_chosen}")
            pdws = to_float(r["total_pdws"])
            hit = int(pdws > 0 or to_float(r["hit"]) > 0)
            snr = to_float(r["mean_snr_db"])
            t = step - 1
            scan_t = to_float(r["scan_time_s"])
            scan_d = to_float(r["dwell_s"])
            try:
                ctrl.observe(t, band_seen, hit, snr, pdws, scan_t)
            except TypeError:
                ctrl.observe(t, band_seen, hit, snr, pdws)
            history.append({"config_id": cfg, "dwell": t, "time_s": to_float(r["scan_time_s"]),
                            "dwell_s": to_float(r["dwell_s"]) or info["dwell_s"],
                            "centre_mhz": float(BAND_CENTRE_MHZ[band_seen]), "decision": info["decision"],
                            "p_target": info["p_target"], "confidence": info["confidence"],
                            "receiver_hit": hit, "pdws": pdws})
            # decide the next dwell immediately
            next_step = step + 1
            band, score, info = decide_with_telemetry(ctrl, args.policy, next_step - 1, scan_t + scan_d)
            pending[next_step] = (band, score, info)
            write_command(cmd_path, next_step, band, info["dwell_s"])
            last_activity = time.time()
            if step % 100 == 0:
                h = sum(x["receiver_hit"] for x in history)
                print(f"  dwell {step}: detections {h} ({100 * h / len(history):.1f}%)")
        if not rows:
            if done_seen:
                break                          # DONE seen and file fully drained
            st = read_status(status_path)
            if st and st[0] == "DONE":
                done_seen = True               # read the file once more before stopping
                continue
            if time.time() - last_activity > 120:
                print("No new observation for 120 s - stopping.")
                break
            time.sleep(POLL_S)

    hist = pd.DataFrame(history, columns=HISTORY_COLUMNS)
    tag = f"{args.policy}_config{cfg}"
    if isinstance(ctrl, TieredDecider) and ctrl.force_tier:
        tag += f"_forced_tier{ctrl.force_tier}"         # do not overwrite the normal run's history
    if isinstance(ctrl, TieredDecider):
        print(f"Tier usage: " + ", ".join(f"tier {k}: {v}" for k, v in ctrl.tier_counts.items()))
    hist_path = REPORTS_DIR / f"live_history_{tag}.csv"
    hist.to_csv(hist_path, index=False)
    if len(hist):
        summ = live_audit(hist, args.policy, cfg, REPORTS_DIR / f"live_audit_{tag}.xlsx")
        print("\nLIVE RUN SUMMARY:", summ)
        print(f"Saved {hist_path} and {REPORTS_DIR / f'live_audit_{tag}.xlsx'}")
    else:
        print("No observations were received.")


if __name__ == "__main__":
    main()
