"""
STEP 2 - Train LightGBM:  P(receiver detects a signal in band b at dwell t).

How training rows are made (no leakage, no train/runtime mismatch):
  The Scilab baseline sweep recorded what the receiver saw in EVERY band at
  EVERY dwell. We replay that record while pretending to be a scanner that can
  look at only ONE band per dwell (like the real Smart Scan). The scanner's
  knowledge is kept by the same FeatureTracker used at runtime. At each dwell we
  log the features of all 322 bands and the label "would the receiver detect
  something in this band now?" (taken from the recorded sweep).
  Features only ever describe PAST scans, never the scan being predicted.

Configuration-level split: the test configurations are never used here.
"""
import argparse
import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from smartscan_config import (N_BANDS, PREPARED_DIR, MODELS_DIR, DEFAULT_TRAIN_CONFIGS,
                              DEFAULT_TEST_CONFIGS, parse_config_list)
from smartscan_core import FeatureTracker, FEATURE_NAMES, load_grid

NEG_KEEP = 0.05          # keep 5 % of negative rows ...
NEG_WEIGHT = 1.0 / NEG_KEEP   # ... and weight them x20 so probabilities stay calibrated


def behaviour_band(kind, t, tracker, rng, rr_state):
    """Exploration policies used only to generate realistic scan histories."""
    if kind == "random":
        return int(rng.integers(N_BANDS))
    if kind == "round_robin":
        b = rr_state[0]
        rr_state[0] = (b + 1) % N_BANDS
        return b
    # "sticky": re-scan a band that just hit, otherwise favour bands with hit history
    if tracker.recent and tracker.recent[-1] > 0 and rng.random() < 0.8:
        return int(np.argmax(tracker.last_scan))
    if rng.random() < 0.3:
        return int(rng.integers(N_BANDS))
    w = (tracker.hits + 0.02) / (tracker.scans + 1.0) + 0.002
    return int(rng.choice(N_BANDS, p=w / w.sum()))


def replay_rows(grid, kinds, rng):
    hit = grid["hit"]
    snr = grid["snr"]
    pdws = grid["pdws"]
    D = hit.shape[0]
    Xs, ys, ws = [], [], []
    for kind in kinds:
        tr = FeatureTracker()
        rr_state = [int(rng.integers(N_BANDS))]
        for t in range(D):
            X, _ = tr.features(t)
            y = hit[t].astype(np.int8)
            keep = (y == 1) | (rng.random(N_BANDS) < NEG_KEEP)
            Xs.append(X[keep])
            ys.append(y[keep])
            ws.append(np.where(y[keep] == 1, 1.0, NEG_WEIGHT))
            b = behaviour_band(kind, t, tr, rng, rr_state)
            tr.update(t, b, hit[t, b], snr[t, b], pdws[t, b])
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def build_set(configs, kinds, seed):
    rng = np.random.default_rng(seed)
    parts = [replay_rows(load_grid(PREPARED_DIR / f"config_{c}.npz"), kinds, rng) for c in configs]
    return (np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def metrics(y, p, w):
    return {"rows": int(len(y)), "positives": int(y.sum()),
            "roc_auc": round(float(roc_auc_score(y, p, sample_weight=w)), 4),
            "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), sample_weight=w)), 5)}


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser(description="Step 2: train LightGBM")
    ap.add_argument("--train-configs", default=",".join(map(str, DEFAULT_TRAIN_CONFIGS)))
    ap.add_argument("--test-configs", default=",".join(map(str, DEFAULT_TEST_CONFIGS)))
    args = ap.parse_args()
    train_cfg = parse_config_list(args.train_configs)
    test_cfg = parse_config_list(args.test_configs)
    if set(train_cfg) & set(test_cfg):
        raise SystemExit("Train and test configurations must not overlap.")
    for c in train_cfg + test_cfg:
        if not (PREPARED_DIR / f"config_{c}.npz").exists():
            raise SystemExit(f"prepared/config_{c}.npz missing - run step 1 for this configuration.")
    if len(train_cfg) < 2:
        raise SystemExit("Need at least 2 training configurations (one is used for validation).")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    kinds = ["random", "round_robin", "sticky"]
    val_cfg = [train_cfg[-1]]
    fit_cfg = train_cfg[:-1]
    t0 = time.time()
    print(f"Building training rows: fit={fit_cfg} validation={val_cfg} test={test_cfg}")
    Xf, yf, wf = build_set(fit_cfg, kinds, 1)
    Xv, yv, wv = build_set(val_cfg, kinds, 2)
    print(f"  fit rows {len(yf):,} (positives {int(yf.sum()):,}), validation rows {len(yv):,}  "
          f"({time.time() - t0:.1f} s)")

    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbosity=-1, seed=42, num_threads=0)
    dfit = lgb.Dataset(Xf, yf, weight=wf, feature_name=FEATURE_NAMES)
    dval = lgb.Dataset(Xv, yv, weight=wv, feature_name=FEATURE_NAMES, reference=dfit)
    m_val = lgb.train(params, dfit, num_boost_round=1500, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    best = max(50, int(m_val.best_iteration or 300))
    m_val.save_model(str(MODELS_DIR / "lightgbm_validation_model.txt"), num_iteration=best)
    print(f"  best number of trees (early stopping on config {val_cfg[0]}): {best}")

    # Final model on ALL training configurations.
    Xa, ya, wa = np.vstack([Xf, Xv]), np.concatenate([yf, yv]), np.concatenate([wf, wv])
    final = lgb.train(params, lgb.Dataset(Xa, ya, weight=wa, feature_name=FEATURE_NAMES),
                      num_boost_round=best)
    final.save_model(str(MODELS_DIR / "lightgbm_model.txt"))

    # Cross-fitted models for PPO training: for each training configuration, a model
    # that never saw it (so PPO learns with realistic, out-of-sample probabilities).
    per_cfg = {c: build_set([c], kinds, 10 + c) for c in train_cfg}
    for c in train_cfg:
        others = [k for k in train_cfg if k != c]
        Xo = np.vstack([per_cfg[k][0] for k in others])
        yo = np.concatenate([per_cfg[k][1] for k in others])
        wo = np.concatenate([per_cfg[k][2] for k in others])
        m = lgb.train(params, lgb.Dataset(Xo, yo, weight=wo, feature_name=FEATURE_NAMES),
                      num_boost_round=best)
        m.save_model(str(MODELS_DIR / f"lightgbm_crossfit_config{c}.txt"))
    print(f"  cross-fitted models for PPO: {len(train_cfg)}")

    report = {"train_configs": train_cfg, "validation_config": val_cfg[0], "test_configs": test_cfg,
              "trees": best, "features": FEATURE_NAMES,
              "validation": metrics(yv, m_val.predict(Xv, num_iteration=best), wv)}
    if test_cfg:
        Xt, yt, wt = build_set(test_cfg, kinds, 3)
        report["test"] = metrics(yt, final.predict(Xt), wt)
    imp = final.feature_importance(importance_type="gain")
    report["feature_importance_gain"] = {n: round(float(v), 1) for n, v in
                                         sorted(zip(FEATURE_NAMES, imp), key=lambda z: -z[1])}
    (MODELS_DIR / "lightgbm_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("validation", "test") if k in report}, indent=2))
    print(f"\nSTEP 2 DONE ({time.time() - t0:.0f} s). Saved {MODELS_DIR / 'lightgbm_model.txt'}")


if __name__ == "__main__":
    main()
