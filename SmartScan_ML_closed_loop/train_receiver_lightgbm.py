"""Train the receiver-driven LightGBM probability models.

The training CSV contains one row per receiver scan, with candidate_band equal
 to the band that was actually scanned. One model is trained per band using
only receiver-observable/history features. Simulator ground truth appears
only in target_active and is never used as a feature.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, log_loss
from receiver_ml_config import N_BANDS, normalize

FEATURES = [
    "candidate_center_mhz", "candidate_lower_mhz", "candidate_upper_mhz",
    "time_s", "band_scan_count", "band_hit_count", "band_miss_count",
    "band_hit_rate", "time_since_scan_s", "last_snr_db",
    "last_max_snr_db", "last_pdws", "last_frequency_mhz",
    "global_recent_hit_rate", "current_total_pdws",
    "current_station_hit_count", "current_track_count", "current_mean_snr_db",
    "current_max_snr_db", "current_mean_frequency_mhz",
    "current_mean_amplitude_dbm", "current_mean_pulse_width_us",
    "current_mean_aoa_deg",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-dir", default="receiver_lightgbm_models")
    ap.add_argument("--test-fraction", type=float, default=0.20)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    required = FEATURES + ["config_id", "step", "candidate_band", "target_active"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    configs = np.array(sorted(df.config_id.unique()))
    if len(configs) < 2:
        raise ValueError("Need at least 2 configurations for a configuration-level train/test split.")
    rng = np.random.default_rng(42)
    rng.shuffle(configs)
    n_test = max(1, int(round(len(configs) * args.test_fraction)))
    n_test = min(n_test, len(configs) - 1)
    test_cfg = set(configs[:n_test])
    train_cfg = set(configs[n_test:])

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    test_predictions = []
    per_band = {}

    for b in range(N_BANDS):
        tr = df[(df.config_id.isin(train_cfg)) & (df.candidate_band == b)]
        te = df[(df.config_id.isin(test_cfg)) & (df.candidate_band == b)]
        if tr.empty or te.empty:
            raise ValueError(f"Band {b}: missing train/test rows.")

        ytr = tr.target_active.astype(int)
        yte = te.target_active.astype(int)
        if ytr.nunique() < 2:
            # A constant model is valid for a band with no positive training
            # examples, but still keeps the same runtime interface.
            constant = float(ytr.iloc[0])
            booster_path = model_dir / f"band_{b:03d}.txt"
            # Train a tiny LightGBM model on a duplicated two-class sample so
            # the saved model remains a standard LightGBM booster.
            x0 = tr[FEATURES].iloc[[0]].copy()
            x1 = x0.copy()
            y0 = int(ytr.iloc[0])
            y_alt = 1 - y0
            Xfit = pd.concat([x0, x1], ignore_index=True)
            yfit = pd.Series([y0, y_alt])
            m = LGBMClassifier(
                objective="binary", n_estimators=1, learning_rate=0.1,
                num_leaves=2, min_child_samples=1, min_data_in_leaf=1,
                min_gain_to_split=0.0, verbosity=-1, random_state=42+b,
            )
            m.fit(Xfit, yfit)
            m.booster_.save_model(str(booster_path))
            p = np.full(len(te), constant, dtype=float)
        else:
            m = LGBMClassifier(
                objective="binary", n_estimators=250, learning_rate=0.03,
                num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                verbosity=-1, random_state=42 + b,
            )
            m.fit(tr[FEATURES], ytr)
            m.booster_.save_model(str(model_dir / f"band_{b:03d}.txt"))
            p = m.predict_proba(te[FEATURES])[:, 1]

        test_predictions.append(pd.DataFrame({
            "config_id": te.config_id.to_numpy(),
            "step": te.step.to_numpy(),
            "candidate_band": b,
            "prediction": np.clip(p, 0.0, 1.0),
            "truth": yte.to_numpy(),
        }))

        if yte.nunique() >= 2:
            per_band[str(b)] = {
                "test_rows": int(len(te)),
                "positive_rate": float(yte.mean()),
                "roc_auc": float(roc_auc_score(yte, p)),
                "log_loss": float(log_loss(yte, np.clip(p, 1e-6, 1-1e-6))),
            }
        else:
            per_band[str(b)] = {
                "test_rows": int(len(te)),
                "positive_rate": float(yte.mean()),
                "roc_auc": None,
                "log_loss": None,
            }

    pred = pd.concat(test_predictions, ignore_index=True)
    pred_p = pred.pivot(index=["config_id", "step"], columns="candidate_band", values="prediction")
    pred_y = pred.pivot(index=["config_id", "step"], columns="candidate_band", values="truth")
    P = np.asarray([normalize(r) for r in pred_p.to_numpy(float)])
    Y = pred_y.to_numpy(float)
    top1 = float(np.mean(Y[np.arange(len(Y)), np.argmax(P, axis=1)] > 0))
    top5_idx = np.argpartition(P, -min(5, N_BANDS), axis=1)[:, -min(5, N_BANDS):]
    top5 = float(np.mean(np.any(Y[np.arange(len(Y))[:, None], top5_idx] > 0, axis=1)))

    metrics = {
        "train_configs": sorted(map(int, train_cfg)),
        "test_configs": sorted(map(int, test_cfg)),
        "train_rows": int(df[df.config_id.isin(train_cfg)].shape[0]),
        "test_rows": int(df[df.config_id.isin(test_cfg)].shape[0]),
        "test_scan_steps": int(len(Y)),
        "top1_active_band": top1,
        "top5_active_band": top5,
        "mean_band_test_positive_rate": float(pred.groupby("candidate_band").truth.mean().mean()),
        "per_band": per_band,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metrics, indent=2))
    pred.to_csv(model_dir / "test_predictions_long.csv", index=False)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_band"}, indent=2))
    print(f"Saved models: {model_dir}")


if __name__ == "__main__":
    main()
