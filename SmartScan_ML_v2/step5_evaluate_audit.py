"""
STEP 5 - Evaluate on the held-out TEST configurations and write the audit report.

For every test configuration and every policy, the recorded receiver sweep is
replayed dwell by dwell. The policy picks ONE band per 10-ms dwell, receives the
receiver result for that band only, updates its history, and picks again.

Policies
  custom       LightGBM -> DBN -> PPO           (your model)
  whittle      Whittle index                    (baseline)
  round_robin  Round Robin                      (baseline)
  greedy       LightGBM -> DBN, no PPO          (ablation: shows what PPO adds)

Output: reports/smart_scan_audit.xlsx
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from smartscan_config import (PREPARED_DIR, MODELS_DIR, REPORTS_DIR, BAND_CENTRE_MHZ, DWELL_S,
                              NOISE_FLOOR_DBM, DETECTION_THRESHOLD_DB, parse_config_list)
from smartscan_core import Controller, load_grid

POLICY_LABEL = {"custom": "Custom: LightGBM -> DBN -> PPO", "whittle": "Baseline: Whittle index",
                "round_robin": "Baseline: Round Robin", "greedy": "Ablation: LightGBM -> DBN (no PPO)"}


def emitter_cells(g):
    """(dwell, band) -> set of transmitter ids the receiver can detect there."""
    ok = (g["ev_tx"] > 0) & ((g["ev_amp"] - NOISE_FLOOR_DBM) >= DETECTION_THRESHOLD_DB)
    cells = defaultdict(set)
    first = {}
    for d, b, tx in zip(g["ev_dwell"][ok], g["ev_band"][ok], g["ev_tx"][ok]):
        cells[(int(d), int(b))].add(int(tx))
        if int(tx) not in first or d < first[int(tx)]:
            first[int(tx)] = int(d)
    return cells, first


def run_policy(ctrl, g, cfg, policy, cells):
    hit, snr, pdws, truth = g["hit"], g["snr"], g["pdws"], g["truth_active"]
    ctrl.reset()
    rows = []
    first_detect = {}
    for t in range(hit.shape[0]):
        now = float(g["dwell_time_s"][t])
        band, score = ctrl.decide(t, now)
        h = int(hit[t, band])
        ctrl.observe(t, band, h, float(snr[t, band]), float(pdws[t, band]), now)
        found = cells.get((t, band), set())
        for tx in found:
            first_detect.setdefault(tx, t)
        rows.append({"config_id": cfg, "policy": policy, "dwell": t,
                     "time_s": float(g["dwell_time_s"][t]), "band": band,
                     "centre_mhz": float(BAND_CENTRE_MHZ[band]), "model_score": score,
                     "receiver_hit": h, "truth_active": int(truth[t, band]),
                     "emitters_detected": len(found), "any_hit_available": int(hit[t].sum() > 0),
                     "mean_snr_db": float(snr[t, band]) if h else np.nan})
    return rows, first_detect


def summarise(df, first_by_key, detectable_first):
    out = []
    for (cfg, pol), d in df.groupby(["config_id", "policy"], sort=False):
        det_first = detectable_first[cfg]
        found = first_by_key[(cfg, pol)]
        delays = [(found[tx] - det_first[tx]) * DWELL_S * 1000 for tx in found]
        out.append({"config_id": cfg, "policy": pol, "scans": len(d),
                    "detections": int(d.receiver_hit.sum()),
                    "detection_rate_pct": 100 * d.receiver_hit.mean(),
                    "oracle_max_detections": int(d.any_hit_available.sum()),
                    "efficiency_vs_oracle_pct": 100 * d.receiver_hit.sum() / max(d.any_hit_available.sum(), 1),
                    "scans_on_truly_active_band_pct": 100 * d.truth_active.mean(),
                    "emitters_detectable": len(det_first), "emitters_detected": len(found),
                    "emitter_coverage_pct": 100 * len(found) / max(len(det_first), 1),
                    "mean_first_detection_delay_ms": float(np.mean(delays)) if delays else np.nan})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="Step 5: evaluate + audit")
    rep = json.loads((MODELS_DIR / "lightgbm_report.json").read_text())
    ap.add_argument("--test-configs", default=",".join(map(str, rep.get("test_configs", []))))
    ap.add_argument("--policies", default="custom,whittle,round_robin,greedy")
    ap.add_argument("--output", default=str(REPORTS_DIR / "smart_scan_audit.xlsx"))
    args = ap.parse_args()
    test_cfg = parse_config_list(args.test_configs)
    if not test_cfg:
        raise SystemExit("No test configurations given.")
    overlap = set(test_cfg) & set(rep["train_configs"])
    if overlap:
        print(f"WARNING: configs {sorted(overlap)} were used for training - results on them are optimistic.")
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    controllers = {p: Controller(p, MODELS_DIR) for p in policies}
    all_rows, first_by_key, detectable_first = [], {}, {}
    for cfg in test_cfg:
        g = load_grid(PREPARED_DIR / f"config_{cfg}.npz")
        cells, det_first = emitter_cells(g)
        detectable_first[cfg] = det_first
        for p in policies:
            rows, found = run_policy(controllers[p], g, cfg, p, cells)
            all_rows.extend(rows)
            first_by_key[(cfg, p)] = found
            print(f"  config {cfg} | {p:12s} | detections {sum(r['receiver_hit'] for r in rows):5d} "
                  f"/ {len(rows)} scans")

    hist = pd.DataFrame(all_rows)
    per_cfg = summarise(hist, first_by_key, detectable_first)
    tot = per_cfg.groupby("policy", sort=False).agg(
        scans=("scans", "sum"), detections=("detections", "sum"),
        oracle_max_detections=("oracle_max_detections", "sum"),
        emitters_detectable=("emitters_detectable", "sum"), emitters_detected=("emitters_detected", "sum")
    ).reset_index()
    tot["detection_rate_pct"] = 100 * tot.detections / tot.scans
    tot["efficiency_vs_oracle_pct"] = 100 * tot.detections / tot.oracle_max_detections.clip(lower=1)
    tot["emitter_coverage_pct"] = 100 * tot.emitters_detected / tot.emitters_detectable.clip(lower=1)
    act = hist.groupby("policy", sort=False).truth_active.mean() * 100
    tot["scans_on_truly_active_band_pct"] = tot.policy.map(act)
    delay = per_cfg.groupby("policy", sort=False).mean_first_detection_delay_ms.mean()
    tot["mean_first_detection_delay_ms"] = tot.policy.map(delay)
    tot.insert(1, "description", tot.policy.map(POLICY_LABEL))
    tot = tot.round(2)
    per_cfg = per_cfg.round(2)

    bands = (hist.groupby(["policy", "band", "centre_mhz"]).agg(scans=("receiver_hit", "size"),
             detections=("receiver_hit", "sum")).reset_index())
    bands["hit_rate_pct"] = (100 * bands.detections / bands.scans).round(2)
    notes = pd.DataFrame({"item": [
        "test_configs", "train_configs", "dwell", "detection_rate_pct", "efficiency_vs_oracle_pct",
        "scans_on_truly_active_band_pct", "emitter_coverage_pct", "mean_first_detection_delay_ms",
        "ground truth use"], "meaning": [
        str(test_cfg), str(rep["train_configs"]), f"{DWELL_S * 1000:.0f} ms, one 50-MHz band per dwell",
        "share of scans where the receiver detected a signal",
        "detections / dwells where at least one band had a detectable signal (the best any 1-band scanner could do)",
        "share of scans on a band where a real transmitter was active (includes signals too weak to detect)",
        "distinct transmitters detected at least once / distinct transmitters that were detectable",
        "average delay between a transmitter first becoming detectable and the policy first detecting it",
        "used ONLY here for scoring; the policies never see it"]})

    out = args.output
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        tot.to_excel(w, sheet_name="Summary", index=False)
        per_cfg.to_excel(w, sheet_name="Per_Config", index=False)
        hist.to_excel(w, sheet_name="Scan_History", index=False)
        bands.to_excel(w, sheet_name="Band_Stats", index=False)
        notes.to_excel(w, sheet_name="Notes", index=False)
    hist.to_csv(REPORTS_DIR / "scan_history_offline.csv", index=False)
    tot.to_csv(REPORTS_DIR / "summary_offline.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + tot[["policy", "detections", "scans", "detection_rate_pct", "efficiency_vs_oracle_pct",
                      "emitter_coverage_pct", "mean_first_detection_delay_ms"]].to_string(index=False))
    print(f"\nSTEP 5 DONE ({time.time() - t0:.0f} s). Audit: {out}")


if __name__ == "__main__":
    main()
