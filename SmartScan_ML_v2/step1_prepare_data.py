"""
STEP 1 - Prepare training data from the Scilab receiver output.

Reads
  * ml_observation_live.csv   (written by Scilab main.sce, baseline sweep)
  * whole_dataset_1.24m.csv   (only used for OFFLINE ground truth / evaluation)

Writes  prepared/config_<id>.npz  with, per configuration:
  dwell_time_s  (D,)        start time of every 10-ms receiver dwell
  hit           (D, 322)    1 if the receiver detected something in that band (observable)
  snr           (D, 322)    receiver mean SNR in that band (0 if nothing)
  pdws          (D, 322)    receiver PDW count in that band
  truth_active  (D, 322)    1 if a real transmitter (id > 0) was in that band  (offline only)
  ev_time, ev_band, ev_tx, ev_amp   pulse list for evaluation (offline only)

Handles the known Scilab issue where empty-window rows have no 'hit' value
(26 fields instead of 27): missing hit is treated as 0.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from smartscan_config import (N_BANDS, PREPARED_DIR, NOISE_FLOOR_DBM, DETECTION_THRESHOLD_DB,
                              band_of_centre, band_of_frequency, parse_config_list)
from lpi_lpd import label_lpi, label_lpd

OBS_COLUMNS = ["config_id", "scan_time_s", "scan_center_mhz", "dwell_s",
               "total_pdws", "mean_snr_db", "hit"]


def read_observations(path):
    print(f"Reading receiver observations: {path}")
    t0 = time.time()
    df = pd.read_csv(path, usecols=OBS_COLUMNS, na_values=["Nan", "NaN", "nan", "-Nan", "Inf", "-Inf"],
                     dtype={"config_id": "float64", "scan_time_s": "float64", "scan_center_mhz": "float64",
                            "dwell_s": "float64", "total_pdws": "float64", "mean_snr_db": "float64",
                            "hit": "float64"})
    df = df.dropna(subset=["config_id", "scan_time_s", "scan_center_mhz"])
    df["config_id"] = df["config_id"].astype(int)
    df["total_pdws"] = df["total_pdws"].fillna(0.0)
    df["hit"] = df["hit"].fillna(0.0)
    df["mean_snr_db"] = df["mean_snr_db"].fillna(0.0)
    print(f"  rows: {len(df):,}   ({time.time() - t0:.1f} s)")
    return df


def read_dataset(path, configs):
    print(f"Reading dataset (ground truth only): {path}")
    t0 = time.time()
    use = ["config_id", "ToA", "Frequency", "Amplitude", "transmitter_id", "PulseWidth"]
    ds = pd.read_csv(path, usecols=use + ["power_w", "frequency_mode", "pri_mode"])
    for c in use + ["power_w"]:
        ds[c] = pd.to_numeric(ds[c], errors="coerce")
    ds = ds.dropna(subset=use)
    ds["config_id"] = ds["config_id"].astype(int)
    ds = ds[ds["config_id"].isin(configs)]
    print(f"  rows for selected configs: {len(ds):,}   ({time.time() - t0:.1f} s)")
    return ds


def build_config(cfg, obs_cfg, ds_cfg):
    # ---------------- dwell index ----------------
    times = np.sort(obs_cfg["scan_time_s"].unique())
    dwell_idx = np.searchsorted(times, obs_cfg["scan_time_s"].to_numpy())
    rows_per_dwell = np.bincount(dwell_idx, minlength=len(times))
    expected = int(np.median(rows_per_dwell))
    # The last dwell can be incomplete if Scilab was stopped mid-dwell.
    n_keep = len(times)
    while n_keep > 0 and rows_per_dwell[n_keep - 1] < expected:
        n_keep -= 1
    dropped = len(times) - n_keep
    keep = dwell_idx < n_keep
    obs_cfg = obs_cfg[keep]
    dwell_idx = dwell_idx[keep]
    times = times[:n_keep]
    D = n_keep

    bands = band_of_centre(obs_cfg["scan_center_mhz"].to_numpy())
    if bands.min() < 0 or bands.max() >= N_BANDS:
        raise ValueError(f"Config {cfg}: scan centre outside 0..16100 MHz grid.")

    hit = np.zeros((D, N_BANDS), dtype=np.int8)
    snr = np.zeros((D, N_BANDS), dtype=np.float32)
    pdws = np.zeros((D, N_BANDS), dtype=np.int16)
    detected = (obs_cfg["total_pdws"].to_numpy() > 0) | (obs_cfg["hit"].to_numpy() > 0)
    hit[dwell_idx, bands] = detected.astype(np.int8)
    snr[dwell_idx, bands] = obs_cfg["mean_snr_db"].to_numpy(np.float32)
    pdws[dwell_idx, bands] = np.clip(obs_cfg["total_pdws"].to_numpy(), 0, 32000).astype(np.int16)

    # ---------------- ground truth from dataset ----------------
    ds_cfg = ds_cfg.sort_values("ToA")
    ev_time = ds_cfg["ToA"].to_numpy(float) * 1e-6
    ev_band = band_of_frequency(ds_cfg["Frequency"].to_numpy(float))
    ev_tx = ds_cfg["transmitter_id"].to_numpy(float).astype(np.int32)
    ev_amp = ds_cfg["Amplitude"].to_numpy(float).astype(np.float32)
    ev_freq = ds_cfg["Frequency"].to_numpy(float)
    ev_pw = ds_cfg["PulseWidth"].to_numpy(float).astype(np.float32)
    # True LPI / LPD label of every transmitter (offline scoring only)
    tx_ids, tx_lpi, tx_lpd = [], [], []
    thr = NOISE_FLOOR_DBM + DETECTION_THRESHOLD_DB
    for tid, gtx in ds_cfg[ds_cfg["transmitter_id"] > 0].groupby("transmitter_id"):
        tx_ids.append(int(tid))
        tx_lpi.append(label_lpi(gtx["power_w"].iloc[0], str(gtx["frequency_mode"].iloc[0]), str(gtx["pri_mode"].iloc[0])))
        tx_lpd.append(label_lpd(float((gtx["Amplitude"] >= thr).mean())))

    dwell_s = float(obs_cfg["dwell_s"].median())
    lo = np.searchsorted(ev_time, times - 5e-10, side="left")
    hi = np.searchsorted(ev_time, times + dwell_s + 5e-10, side="right")
    ev_dwell = np.full(len(ev_time), -1, dtype=np.int64)
    for d in range(D):
        ev_dwell[lo[d]:hi[d]] = d
    m = ev_dwell >= 0

    truth_active = np.zeros((D, N_BANDS), dtype=np.int8)
    sel = m & (ev_tx > 0)
    truth_active[ev_dwell[sel], ev_band[sel]] = 1

    # Receiver rule check (any pulse >= threshold -> detection expected).
    expect = np.zeros((D, N_BANDS), dtype=np.int8)
    sel = m & ((ev_amp - NOISE_FLOOR_DBM) >= DETECTION_THRESHOLD_DB)
    expect[ev_dwell[sel], ev_band[sel]] = 1
    agree = float((expect == hit).mean())
    hit_cells = int(hit.sum())
    exp_cells = int(expect.sum())

    out = PREPARED_DIR / f"config_{cfg}.npz"
    np.savez_compressed(out, dwell_time_s=times, hit=hit, snr=snr, pdws=pdws,
                        truth_active=truth_active,
                        ev_time=ev_time[m], ev_dwell=ev_dwell[m].astype(np.int32),
                        ev_band=ev_band[m].astype(np.int16), ev_tx=ev_tx[m], ev_amp=ev_amp[m],
                        ev_freq=ev_freq[m], ev_pw=ev_pw[m],
                        tx_ids=np.array(tx_ids, np.int32), tx_lpi=np.array(tx_lpi, np.int8),
                        tx_lpd=np.array(tx_lpd, np.int8))
    info = {
        "config_id": int(cfg), "dwells": int(D), "dropped_incomplete_dwells": int(dropped),
        "windows_per_dwell": expected, "receiver_hit_cells": hit_cells,
        "dataset_rule_hit_cells": exp_cells, "grid_agreement": round(agree, 6),
        "truth_active_cells": int(truth_active.sum()),
        "dwells_with_any_hit": int((hit.sum(axis=1) > 0).sum()),
        "simulated_seconds": round(float(times[-1] - times[0] + dwell_s), 3) if D else 0.0,
    }
    return info


def main():
    ap = argparse.ArgumentParser(description="Step 1: prepare per-configuration grids")
    ap.add_argument("--observations", required=True, nargs="+",
                    help="ml_observation_live.csv (and ml_observation_sweep.csv for configs 9-11)")
    ap.add_argument("--dataset", required=True, help="path to whole_dataset_1.24m.csv")
    ap.add_argument("--configs", default="all", help="e.g. 0-8 (default: every config in the CSV)")
    args = ap.parse_args()

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    parts = [read_observations(p) for p in args.observations]
    obs = pd.concat(parts, ignore_index=True)
    # a configuration present in several files: keep the file listed LAST (newest sweep)
    for i, p in enumerate(parts[:-1]):
        later = set().union(*[set(q["config_id"].unique()) for q in parts[i + 1:]])
        dup = set(p["config_id"].unique()) & later
        if dup:
            print(f"  configs {sorted(dup)} appear in more than one file - using the later file")
    if len(parts) > 1:
        keep = []
        seen = set()
        for p in reversed(parts):
            cfgs = set(p["config_id"].unique()) - seen
            keep.append(p[p["config_id"].isin(cfgs)])
            seen |= cfgs
        obs = pd.concat(keep, ignore_index=True)
    available = sorted(obs["config_id"].unique().tolist())
    configs = available if args.configs == "all" else [c for c in parse_config_list(args.configs) if c in available]
    missing = [] if args.configs == "all" else [c for c in parse_config_list(args.configs) if c not in available]
    if missing:
        print(f"WARNING: configs not present in the observation CSV (skipped): {missing}")
    if not configs:
        raise SystemExit("No configurations to prepare.")
    print(f"Configurations: {configs}")

    ds = read_dataset(args.dataset, configs)
    summary = []
    for cfg in configs:
        info = build_config(cfg, obs[obs["config_id"] == cfg], ds[ds["config_id"] == cfg])
        summary.append(info)
        print(f"  config {cfg}: dwells={info['dwells']:,} (dropped {info['dropped_incomplete_dwells']}), "
              f"hits={info['receiver_hit_cells']:,}, receiver-vs-dataset agreement={info['grid_agreement']:.4%}")
    (PREPARED_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSTEP 1 DONE. Saved {len(configs)} configs to {PREPARED_DIR}")


if __name__ == "__main__":
    main()
