"""
STEP 3 - Build the DBN (temporal belief), the LightGBM+DBN fusion, and the
Whittle-index baseline tables.

DBN: for every band, a 2-state hidden Markov chain
        hidden state  = "a transmitter is active in this band"
        P(0->1)=p01, P(1->1)=p11             (how emitters switch on/off over dwells)
        P(detect | active)=pd, P(detect | idle)=pfa   (receiver sensitivity)
     Learned OFFLINE from the training configurations (ground truth is used
     only here, never at runtime). Sparse bands are shrunk toward the global average.

Fusion: logistic combination of logit(LightGBM) and logit(DBN) fitted on the
        validation configuration using the validation LightGBM model
        (so the weights are out-of-sample).

Whittle: 2-state Markov model of "receiver detection" per band + numerical
         Whittle index table (baseline only).
"""
import argparse
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from smartscan_config import (N_BANDS, PREPARED_DIR, MODELS_DIR, DEFAULT_TRAIN_CONFIGS,
                              parse_config_list)
from smartscan_core import DBN, FeatureTracker, load_grid, whittle_index_table, _logit
from step2_train_lightgbm import behaviour_band, NEG_KEEP, NEG_WEIGHT


def markov_counts(seq):
    """seq: (D, N_BANDS) 0/1 -> transition counts per band."""
    a, b = seq[:-1].astype(bool), seq[1:].astype(bool)
    return ((~a) & b).sum(0), ((~a) & (~b)).sum(0), (a & b).sum(0), (a & (~b)).sum(0)


def estimate_chain(grids, key, s0=200.0, s1=20.0):
    n01 = n00 = n11 = n10 = 0
    for g in grids:
        c = markov_counts(g[key])
        n01, n00, n11, n10 = n01 + c[0], n00 + c[1], n11 + c[2], n10 + c[3]
    p01g = n01.sum() / max((n01 + n00).sum(), 1)
    p11g = n11.sum() / max((n11 + n10).sum(), 1)
    p01 = (n01 + s0 * p01g) / (n01 + n00 + s0)
    p11 = (n11 + s1 * p11g) / (n11 + n10 + s1)
    return np.clip(p01, 1e-5, 0.5), np.clip(p11, 0.05, 0.9995), float(p01g), float(p11g)


def estimate_dbn(grids):
    p01, p11, p01g, p11g = estimate_chain(grids, "truth_active")
    act = sum(g["truth_active"].sum(0) for g in grids)
    det = sum((g["truth_active"] & g["hit"]).sum(0) for g in grids)
    idle = sum((1 - g["truth_active"]).sum(0) for g in grids)
    fa = sum(((1 - g["truth_active"]) & g["hit"]).sum(0) for g in grids)
    pdg = det.sum() / max(act.sum(), 1)
    pd = np.clip((det + 20.0 * pdg) / (act + 20.0), 0.01, 0.999)
    pfa = float(np.clip(fa.sum() / max(idle.sum(), 1), 1e-6, 0.01))
    return {"p01": p01.tolist(), "p11": p11.tolist(), "pd": pd.tolist(), "pfa": pfa,
            "global": {"p01": p01g, "p11": p11g, "pd": float(pdg), "pfa": pfa}}


def fusion_rows(grid, booster, dbn_params, rng):
    hit, snr, pdws = grid["hit"], grid["snr"], grid["pdws"]
    rows_l, rows_d, ys, ws = [], [], [], []
    for kind in ("sticky", "random"):
        tr = FeatureTracker()
        dbn = DBN(dbn_params)
        rr_state = [0]
        for t in range(hit.shape[0]):
            X, _ = tr.features(t)
            p_l = np.clip(booster.predict(X, num_threads=1), 1e-6, 1 - 1e-6)
            dbn.predict()
            p_d = np.clip(dbn.hit_probability(), 1e-6, 1 - 1e-6)
            y = hit[t]
            keep = (y == 1) | (rng.random(N_BANDS) < NEG_KEEP)
            rows_l.append(p_l[keep]); rows_d.append(p_d[keep]); ys.append(y[keep])
            ws.append(np.where(y[keep] == 1, 1.0, NEG_WEIGHT))
            b = behaviour_band(kind, t, tr, rng, rr_state)
            dbn.update(b, hit[t, b])
            tr.update(t, b, hit[t, b], snr[t, b], pdws[t, b])
    return (np.concatenate(rows_l), np.concatenate(rows_d), np.concatenate(ys), np.concatenate(ws))


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser(description="Step 3: DBN + fusion + Whittle")
    ap.add_argument("--train-configs", default=",".join(map(str, DEFAULT_TRAIN_CONFIGS)))
    args = ap.parse_args()
    train_cfg = parse_config_list(args.train_configs)
    lgb_report = json.loads((MODELS_DIR / "lightgbm_report.json").read_text())
    if sorted(lgb_report["train_configs"]) != train_cfg:
        raise SystemExit(f"Step 2 was trained on {lgb_report['train_configs']}, not {train_cfg}. "
                         "Use the same --train-configs for steps 2 and 3.")
    val_cfg = lgb_report["validation_config"]
    t0 = time.time()
    grids = {c: load_grid(PREPARED_DIR / f"config_{c}.npz") for c in train_cfg}

    # ---------------- DBN ----------------
    dbn_final = estimate_dbn([grids[c] for c in train_cfg])
    (MODELS_DIR / "dbn_params.json").write_text(json.dumps(dbn_final))
    print("DBN (global): " + ", ".join(f"{k}={v:.4f}" for k, v in dbn_final["global"].items()))

    # ---------------- Fusion (out-of-sample on the validation config) ----------------
    dbn_fit = estimate_dbn([grids[c] for c in train_cfg if c != val_cfg])
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_validation_model.txt"))
    pl, pdb, y, w = fusion_rows(grids[val_cfg], booster, dbn_fit, np.random.default_rng(7))
    Z = np.column_stack([_logit(pl), _logit(pdb)])
    lr = LogisticRegression(C=10.0, max_iter=1000)
    lr.fit(Z, y, sample_weight=w)
    fused = lr.predict_proba(Z)[:, 1]
    fusion = {"w_lgbm": float(lr.coef_[0, 0]), "w_dbn": float(lr.coef_[0, 1]),
              "bias": float(lr.intercept_[0]), "fitted_on_config": int(val_cfg),
              "log_loss_lightgbm_only": float(log_loss(y, pl, sample_weight=w)),
              "log_loss_dbn_only": float(log_loss(y, pdb, sample_weight=w)),
              "log_loss_fused": float(log_loss(y, fused, sample_weight=w))}
    (MODELS_DIR / "fusion.json").write_text(json.dumps(fusion, indent=2))
    print("Fusion: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in fusion.items()}))

    # ---------------- Whittle baseline ----------------
    p01, p11, _, _ = estimate_chain([grids[c] for c in train_cfg], "hit")
    key = np.round(np.column_stack([p01, p11]), 4)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    tables = []
    grid = None
    for a, b in uniq:
        grid, W = whittle_index_table(float(a), float(b))
        tables.append(W)
    table = np.asarray(tables)[inv]
    (MODELS_DIR / "whittle_params.json").write_text(json.dumps(
        {"p01": p01.tolist(), "p11": p11.tolist(), "omega_grid": grid.tolist(),
         "index_table": np.round(table, 5).tolist(), "beta": 0.9}))
    print(f"Whittle index tables: {len(uniq)} unique band models")
    print(f"\nSTEP 3 DONE ({time.time() - t0:.0f} s). Saved dbn_params.json, fusion.json, whittle_params.json")


if __name__ == "__main__":
    main()
