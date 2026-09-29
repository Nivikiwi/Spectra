
"""
Build training examples for receiver-driven LightGBM.

INPUT:
  receiver_training_observations.csv
  receiver_training_ground_truth.csv

Both files must contain:
  config_id, step, scan_center_mhz, ...
The observation file is the same receiver schema as ml_observation_live.csv,
with config_id and step added.

Ground truth must contain one row per config_id/step and one binary column
active_<band_index> for every 50-MHz candidate window.

IMPORTANT:
  Ground truth is used only to create training targets. It is NEVER passed
  to the online scheduler.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from receiver_ml_config import N_BANDS, WINDOW_CENTRES_MHZ, normalize

OBS_REQUIRED = [
    "config_id", "step", "scan_time_s", "scan_center_mhz",
    "scan_lower_mhz", "scan_upper_mhz", "dwell_s", "total_pdws",
    "station_hit_count", "track_count",
    "rs1_pdws", "rs1_mean_snr_db", "rs1_mean_aoa_deg",
    "rs2_pdws", "rs2_mean_snr_db", "rs2_mean_aoa_deg",
    "rs3_pdws", "rs3_mean_snr_db", "rs3_mean_aoa_deg",
    "mean_frequency_mhz", "mean_amplitude_dbm", "max_amplitude_dbm",
    "mean_snr_db", "max_snr_db", "mean_pulse_width_us",
    "mean_aoa_deg", "hit",
]

def build_history_features(group):
    group = group.sort_values("step")
    per_band = {}
    rows = []
    for _, r in group.iterrows():
        t = float(r.scan_time_s)
        global_hits = [v["hit"] for v in per_band.values()]
        global_hit_rate = float(np.mean(global_hits)) if global_hits else 0.0
        for b in range(N_BANDS):
            c = float(WINDOW_CENTRES_MHZ[b])
            h = per_band.get(b, {
                "scans": 0, "hits": 0, "last_t": np.nan,
                "last_snr": np.nan, "last_max_snr": np.nan,
                "last_pdws": 0.0, "last_freq": np.nan,
            })
            if np.isnan(h["last_t"]):
                age = 1e6
            else:
                age = max(0.0, t - h["last_t"])
            rows.append({
                "config_id": r.config_id,
                "step": int(r.step),
                "candidate_band": b,
                "candidate_center_mhz": c,
                "candidate_lower_mhz": c - 25.0,
                "candidate_upper_mhz": c + 25.0,
                "time_s": t,
                "band_scan_count": h["scans"],
                "band_hit_count": h["hits"],
                "band_miss_count": h["scans"] - h["hits"],
                "band_hit_rate": h["hits"] / h["scans"] if h["scans"] else 0.0,
                "time_since_scan_s": age,
                "last_snr_db": 0.0 if np.isnan(h["last_snr"]) else h["last_snr"],
                "last_max_snr_db": 0.0 if np.isnan(h["last_max_snr"]) else h["last_max_snr"],
                "last_pdws": h["last_pdws"],
                "last_frequency_mhz": c if np.isnan(h["last_freq"]) else h["last_freq"],
                "global_recent_hit_rate": global_hit_rate,
            })
        b = int(np.argmin(np.abs(WINDOW_CENTRES_MHZ - float(r.scan_center_mhz))))
        d = per_band.setdefault(b, {
            "scans": 0, "hits": 0, "last_t": np.nan,
            "last_snr": np.nan, "last_max_snr": np.nan,
            "last_pdws": 0.0, "last_freq": np.nan,
        })
        d["scans"] += 1
        d["hits"] += int(r.hit)
        d["last_t"] = t
        d["last_snr"] = float(r.mean_snr_db) if pd.notna(r.mean_snr_db) else np.nan
        d["last_max_snr"] = float(r.max_snr_db) if pd.notna(r.max_snr_db) else np.nan
        d["last_pdws"] = float(r.total_pdws)
        d["last_freq"] = float(r.mean_frequency_mhz) if pd.notna(r.mean_frequency_mhz) else np.nan
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--output", default="receiver_lgbm_training.csv")
    args = ap.parse_args()

    obs = pd.read_csv(args.observations)
    gt = pd.read_csv(args.ground_truth)
    missing = [c for c in OBS_REQUIRED if c not in obs.columns]
    if missing:
        raise ValueError(f"Missing observation columns: {missing}")
    gt_cols = [f"active_{i}" for i in range(N_BANDS)]
    missing_gt = [c for c in ["config_id", "step"] + gt_cols if c not in gt.columns]
    if missing_gt:
        raise ValueError(f"Missing ground-truth columns: {missing_gt}")

    parts = []
    for cfg, g in obs.groupby("config_id"):
        parts.append(build_history_features(g))
    X = pd.concat(parts, ignore_index=True)

    gt_key = gt.set_index(["config_id", "step"])[gt_cols]
    y = gt_key.reindex(pd.MultiIndex.from_frame(X[["config_id", "step"]]))
    y.index = X.index
    y = y.fillna(0.0)

    out = pd.concat([X, y.reset_index(drop=True)], axis=1)
    out.to_csv(args.output, index=False)
    print(f"Rows: {len(out):,}")
    print(f"Configs: {out.config_id.nunique():,}")
    print(f"Candidate bands: {N_BANDS}")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
