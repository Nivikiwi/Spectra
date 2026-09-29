"""
Prepare a receiver-only LightGBM training set for Smart Scan.

Inputs
------
1) ml_observation_live.csv produced by Scilab. It must contain config_id,
   step and the receiver observation columns.
2) whole_dataset_1.24m.csv, the simulator event dataset. It is used ONLY to
   create offline training/evaluation targets.

Outputs
-------
receiver_training_observations.csv
    Clean copy of the receiver observations.

receiver_training_ground_truth.csv
    One row per receiver observation with the true activity of the BAND THAT
    WAS SCANNED on that observation. This is an offline target only.

receiver_lgbm_training.csv
    One row per receiver observation/band scanned. Features are constructed
    from receiver history only; target_active is the offline simulator label.

Design note
-----------
The older package attempted to create 238 candidate rows for every
observation and 238 target columns. With all 12 configurations this can
become hundreds of millions of rows. This script intentionally avoids that
explosion: each baseline observation trains the model for the band that was
actually scanned. At runtime, the same feature builder is used to score all
238 candidate bands from receiver history.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from receiver_ml_config import N_BANDS, WINDOW_CENTRES_MHZ, WINDOW_LOWER_MHZ, WINDOW_BW_MHZ

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


def read_original_dataset(path: Path) -> pd.DataFrame:
    use = ["config_id", "ToA", "Frequency", "transmitter_id"]
    df = pd.read_csv(path, usecols=use)
    df = df.rename(columns={
        "ToA": "toa_us",
        "Frequency": "frequency_mhz",
        "transmitter_id": "tx_id",
    })
    for c in use:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["config_id", "toa_us", "frequency_mhz", "tx_id"])
    df["config_id"] = df["config_id"].astype(int)
    df["toa_s"] = df["toa_us"] * 1e-6
    # Match the receiver's positive-emitter convention.
    df = df[df["tx_id"] > 0].copy()
    return df


def band_index(center_mhz: float) -> int:
    return int(np.argmin(np.abs(WINDOW_CENTRES_MHZ - float(center_mhz))))


def build_truth(obs: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Create one target per observation using the observation's time/frequency window."""
    records = []
    events_by_cfg = {int(k): g.sort_values("toa_s") for k, g in events.groupby("config_id")}

    for cfg, gobs in obs.groupby("config_id", sort=True):
        ev = events_by_cfg.get(int(cfg))
        if ev is None or ev.empty:
            for _, r in gobs.iterrows():
                records.append({
                    "config_id": int(cfg),
                    "step": int(r.step),
                    "candidate_band": band_index(r.scan_center_mhz),
                    "active": 0,
                })
            continue

        times = ev["toa_s"].to_numpy(float)
        freqs = ev["frequency_mhz"].to_numpy(float)
        for _, r in gobs.iterrows():
            t0 = float(r.scan_time_s)
            dwell = float(r.dwell_s)
            t1 = t0 + dwell
            lo = float(r.scan_lower_mhz)
            hi = float(r.scan_upper_mhz)
            i0 = np.searchsorted(times, t0, side="left")
            i1 = np.searchsorted(times, t1, side="right")
            active = int(np.any((freqs[i0:i1] >= lo) & (freqs[i0:i1] <= hi)))
            records.append({
                "config_id": int(cfg),
                "step": int(r.step),
                "candidate_band": band_index(r.scan_center_mhz),
                "active": active,
            })

    return pd.DataFrame.from_records(records)


def build_receiver_features(obs: pd.DataFrame) -> pd.DataFrame:
    """Build candidate-band features for the band actually scanned on each row."""
    parts = []
    for cfg, group in obs.groupby("config_id", sort=True):
        g = group.sort_values("step").copy()
        history = {}
        rows = []

        for _, r in g.iterrows():
            b = band_index(r.scan_center_mhz)
            h = history.get(b, {
                "scans": 0, "hits": 0, "last_t": np.nan,
                "last_snr": np.nan, "last_max_snr": np.nan,
                "last_pdws": 0.0, "last_freq": np.nan,
            })

            t = float(r.scan_time_s)
            age = 1e6 if np.isnan(h["last_t"]) else max(0.0, t - h["last_t"])

            rows.append({
                "config_id": int(cfg),
                "step": int(r.step),
                "candidate_band": b,
                "candidate_center_mhz": float(WINDOW_CENTRES_MHZ[b]),
                "candidate_lower_mhz": float(WINDOW_LOWER_MHZ[b]),
                "candidate_upper_mhz": float(WINDOW_LOWER_MHZ[b] + WINDOW_BW_MHZ),
                "time_s": t,
                "band_scan_count": h["scans"],
                "band_hit_count": h["hits"],
                "band_miss_count": h["scans"] - h["hits"],
                "band_hit_rate": h["hits"] / h["scans"] if h["scans"] else 0.0,
                "time_since_scan_s": age,
                "last_snr_db": 0.0 if np.isnan(h["last_snr"]) else h["last_snr"],
                "last_max_snr_db": 0.0 if np.isnan(h["last_max_snr"]) else h["last_max_snr"],
                "last_pdws": h["last_pdws"],
                "last_frequency_mhz": float(WINDOW_CENTRES_MHZ[b]) if np.isnan(h["last_freq"]) else h["last_freq"],
                "global_recent_hit_rate": float(g.loc[g["step"] < r.step, "hit"].mean()) if (g["step"] < r.step).any() else 0.0,
                # Current receiver observation: these are valid observable inputs.
                "current_total_pdws": float(r.total_pdws),
                "current_station_hit_count": float(r.station_hit_count),
                "current_track_count": float(r.track_count),
                "current_mean_snr_db": float(r.mean_snr_db) if pd.notna(r.mean_snr_db) else 0.0,
                "current_max_snr_db": float(r.max_snr_db) if pd.notna(r.max_snr_db) else 0.0,
                "current_mean_frequency_mhz": float(r.mean_frequency_mhz) if pd.notna(r.mean_frequency_mhz) else float(WINDOW_CENTRES_MHZ[b]),
                "current_mean_amplitude_dbm": float(r.mean_amplitude_dbm) if pd.notna(r.mean_amplitude_dbm) else -95.0,
                "current_mean_pulse_width_us": float(r.mean_pulse_width_us) if pd.notna(r.mean_pulse_width_us) else 0.0,
                "current_mean_aoa_deg": float(r.mean_aoa_deg) if pd.notna(r.mean_aoa_deg) else 0.0,
            })

            history[b] = {
                "scans": h["scans"] + 1,
                "hits": h["hits"] + (0 if pd.isna(r.hit) else int(r.hit)),
                "last_t": t,
                "last_snr": float(r.mean_snr_db) if pd.notna(r.mean_snr_db) else np.nan,
                "last_max_snr": float(r.max_snr_db) if pd.notna(r.max_snr_db) else np.nan,
                "last_pdws": float(r.total_pdws),
                "last_freq": float(r.mean_frequency_mhz) if pd.notna(r.mean_frequency_mhz) else np.nan,
            }

        parts.append(pd.DataFrame(rows))

    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", required=True, help="Path to ml_observation_live.csv")
    ap.add_argument("--dataset", required=True, help="Path to whole_dataset_1.24m.csv")
    ap.add_argument("--output-dir", default="training_data")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    obs = pd.read_csv(args.observations)
    missing = [c for c in OBS_REQUIRED if c not in obs.columns]
    if missing:
        raise ValueError("Observation file is missing columns: " + ", ".join(missing))

    obs["config_id"] = pd.to_numeric(obs["config_id"], errors="raise").astype(int)
    obs["step"] = pd.to_numeric(obs["step"], errors="raise").astype(int)
    obs = obs.sort_values(["config_id", "step"]).reset_index(drop=True)

    events = read_original_dataset(Path(args.dataset))
    truth = build_truth(obs, events)
    features = build_receiver_features(obs)

    truth_path = outdir / "receiver_training_ground_truth.csv"
    obs_path = outdir / "receiver_training_observations.csv"
    train_path = outdir / "receiver_lgbm_training.csv"

    obs.to_csv(obs_path, index=False)
    truth.to_csv(truth_path, index=False)

    train = features.merge(
        truth,
        on=["config_id", "step", "candidate_band"],
        how="left",
        validate="one_to_one",
    )
    if train["active"].isna().any():
        raise RuntimeError("Some observation rows could not be aligned with ground truth.")
    train["target_active"] = train.pop("active").astype(int)
    train.to_csv(train_path, index=False)

    print("Preparation complete.")
    print(f"Observations: {len(obs):,}")
    print(f"Configurations: {obs.config_id.nunique():,}")
    print(f"Training rows: {len(train):,}")
    print(f"Positive targets: {int(train.target_active.sum()):,}")
    print(f"Candidate bands: {N_BANDS}")
    print(f"Saved: {obs_path}")
    print(f"Saved: {truth_path}")
    print(f"Saved: {train_path}")


if __name__ == "__main__":
    main()
