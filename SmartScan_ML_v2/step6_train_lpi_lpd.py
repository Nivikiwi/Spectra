"""
STEP 6 - Train the LPI and LPD classifiers (LightGBM), from receiver pulse data.

For every real transmitter in every configuration:
  * receiver pulses = its pulses that reach the receiver above the detection
    threshold (what the IRSA receiver can measure)
  * labels (LPI, LPD) from the definitions in lpi_lpd.py (ground truth, training only)
  * training rows = features from random subsets of 3 .. all of its pulses, so the
    model also works early, when the receiver has only caught a few pulses.

Evaluation: 5-fold cross-validation GROUPED BY TRANSMITTER SIGNATURE. Many configurations
contain identical copies of the same transmitter (same power, modes, PRIs, pulse widths,
frequencies); all copies are always in the same fold, so the model is never tested on a
transmitter it has already seen. Transmitters never intercepted (< 3 pulses above threshold)
cannot be classified from receiver data; they are counted separately.

Final models are trained WITHOUT the demo configurations (default 7 and 8) and without any
identical copy of their transmitters, so the dashboard shows predictions on unseen transmitters.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from smartscan_config import MODELS_DIR, NOISE_FLOOR_DBM, DETECTION_THRESHOLD_DB, parse_config_list
from lpi_lpd import (LPI_FEATURE_NAMES, MIN_PULSES, LPI_POWER_W, LPD_DETECT_FRACTION,
                     label_lpi, label_lpd, pulse_features)

SUBSET_SIZES = [3, 5, 10, 20, 50, 100, 300]
SIGNATURE_COLS = ["power_w", "frequency_mode", "pri_mode", "pris_us", "pulse_widths_us", "frequencies_mhz",
                  "gain", "scan_rate_rpm"]
N_FOLDS = 5
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, seed=7, num_threads=0)
ROUNDS = 200


def build_transmitters(dataset_path):
    use = ["config_id", "ToA", "Frequency", "PulseWidth", "Amplitude", "transmitter_id",
           "power_w", "frequency_mode", "pri_mode"] + SIGNATURE_COLS[3:]
    ds = pd.read_csv(dataset_path, usecols=use)
    ds = ds[pd.to_numeric(ds["transmitter_id"], errors="coerce") > 0]
    thr = NOISE_FLOOR_DBM + DETECTION_THRESHOLD_DB
    txs = []
    for (cfg, tx), g in ds.groupby(["config_id", "transmitter_id"], sort=True):
        det = g[g["Amplitude"] >= thr]
        frac = len(det) / max(len(g), 1)
        txs.append({"config_id": int(cfg), "tx_id": int(tx), "n_pulses": len(g), "n_detectable": len(det),
                    "detectable_fraction": frac,
                    "lpi": label_lpi(g["power_w"].iloc[0], str(g["frequency_mode"].iloc[0]),
                                     str(g["pri_mode"].iloc[0])),
                    "lpd": label_lpd(frac),
                    "signature": "|".join(str(g[c].iloc[0]) for c in SIGNATURE_COLS),
                    "toa_s": det["ToA"].to_numpy(float) * 1e-6, "freq": det["Frequency"].to_numpy(float),
                    "pw": det["PulseWidth"].to_numpy(float), "amp": det["Amplitude"].to_numpy(float)})
    return txs


def samples_for(tx, rng, full_only=False, k_only=None):
    n = tx["n_detectable"]
    if n < MIN_PULSES:
        return []
    if full_only:
        ks = [n]
    elif k_only is not None:
        ks = [min(k_only, n)]
    else:
        ks = sorted(set([k for k in SUBSET_SIZES if k < n] + [n]))
    out = []
    for k in ks:
        reps = 1 if k == n else 2
        for _ in range(reps):
            idx = np.sort(rng.choice(n, size=k, replace=False)) if k < n else np.arange(n)
            out.append(pulse_features(tx["toa_s"][idx], tx["freq"][idx], tx["pw"][idx], tx["amp"][idx]))
    return out


def make_set(txs, label, rng):
    X, y, w = [], [], []
    for tx in txs:
        rows = samples_for(tx, rng)
        for r in rows:
            X.append(r)
            y.append(tx[label])
            w.append(1.0 / len(rows))                      # every transmitter counts equally
    return np.array(X, np.float32), np.array(y), np.array(w)


def train(X, y, w):
    import lightgbm as lgb
    pos = max(y.sum(), 1)
    neg = max(len(y) - y.sum(), 1)
    ww = w * np.where(y == 1, neg / pos, 1.0)            # balance the rare positive class
    return lgb.train(PARAMS, lgb.Dataset(X, y, weight=ww, feature_name=LPI_FEATURE_NAMES), num_boost_round=ROUNDS)


def scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"transmitters": int(len(y)), "positives": int(y.sum()),
            "roc_auc": round(float(roc_auc_score(y, p)), 3) if 0 < y.sum() < len(y) else None,
            "accuracy_pct": round(100 * (tp + tn) / max(len(y), 1), 1),
            "precision_pct": round(100 * tp / max(tp + fp, 1), 1),
            "recall_pct": round(100 * tp / max(tp + fn, 1), 1),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "always_no_accuracy_pct": round(float(100 * (len(y) - y.sum()) / max(len(y), 1)), 1)}


def main():
    ap = argparse.ArgumentParser(description="Step 6: LPI / LPD classifiers")
    ap.add_argument("--dataset", required=True, help="path to whole_dataset_1.24m.csv")
    ap.add_argument("--demo-configs", default="7,8", help="left out of the final models (dashboard demo)")
    args = ap.parse_args()
    demo = parse_config_list(args.demo_configs)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    txs = build_transmitters(args.dataset)
    seen = [t for t in txs if t["n_detectable"] >= MIN_PULSES]
    configs = sorted({t["config_id"] for t in txs})
    print(f"Transmitters: {len(txs)} in configs {configs}; intercepted (>= {MIN_PULSES} pulses): {len(seen)}; "
          f"never intercepted: {len(txs) - len(seen)}  ({time.time() - t0:.0f} s)")
    print(f"  LPI among intercepted: {sum(t['lpi'] for t in seen)} | LPD among intercepted: {sum(t['lpd'] for t in seen)}")

    report = {"definitions": {"LPI": f"power_w < {LPI_POWER_W:g} W AND agile frequency or PRI",
                              "LPD": f"detectable pulse fraction < {LPD_DETECT_FRACTION:.0%} "
                                     f"(SNR >= {DETECTION_THRESHOLD_DB:g} dB at the receiver)",
                              "min_pulses_to_classify": MIN_PULSES},
              "transmitters_total": len(txs), "intercepted": len(seen),
              "never_intercepted": len(txs) - len(seen),
              "never_intercepted_that_are_LPD": sum(t["lpd"] for t in txs if t["n_detectable"] < MIN_PULSES)}

    sigs = sorted({t["signature"] for t in seen})
    fold_of = {s: i % N_FOLDS for i, s in enumerate(np.random.default_rng(3).permutation(sigs))}
    demo_sigs = {t["signature"] for t in txs if t["config_id"] in demo}
    report["identical_transmitter_copies_across_configs"] = int(sum(
        1 for t in txs if sum(1 for u in txs if u["signature"] == t["signature"]) > 1))
    for label in ("lpi", "lpd"):
        y_all, p_all, y_10, p_10 = [], [], [], []
        for c in range(N_FOLDS):
            tr = [t for t in seen if fold_of[t["signature"]] != c]
            te = [t for t in seen if fold_of[t["signature"]] == c]
            if not te:
                continue
            rng = np.random.default_rng(100 + c)
            X, y, w = make_set(tr, label, rng)
            if y.sum() == 0:
                continue
            m = train(X, y, w)
            for t in te:
                y_all.append(t[label]); p_all.append(float(m.predict(np.array(samples_for(t, rng, full_only=True)))[0]))
                y_10.append(t[label]); p_10.append(float(m.predict(np.array(samples_for(t, rng, k_only=10)))[0]))
        report[f"{label}_cv_all_pulses"] = scores(y_all, p_all)
        report[f"{label}_cv_first_10_pulses"] = scores(y_10, p_10)
        print(f"\n{label.upper()} grouped 5-fold CV (all intercepted pulses): {report[f'{label}_cv_all_pulses']}")
        print(f"{label.upper()} grouped 5-fold CV (only 10 pulses):        {report[f'{label}_cv_first_10_pulses']}")

        final_tr = [t for t in seen if t["config_id"] not in demo and t["signature"] not in demo_sigs]
        X, y, w = make_set(final_tr, label, np.random.default_rng(1))
        train(X, y, w).save_model(str(MODELS_DIR / f"{label}_model.txt"))
    report["final_models_trained_without_configs"] = demo
    report["final_models_training_transmitters"] = len([t for t in seen if t["config_id"] not in demo
                                                        and t["signature"] not in demo_sigs])
    (MODELS_DIR / "lpi_lpd_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSTEP 6 DONE ({time.time() - t0:.0f} s). Saved lpi_model.txt, lpd_model.txt, lpi_lpd_report.json")


if __name__ == "__main__":
    main()
