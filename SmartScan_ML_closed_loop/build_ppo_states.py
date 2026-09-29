"""
Build PPO training states.

Offline pipeline:

receiver-derived training examples
        ↓
LightGBM probabilities
        ↓
DBN transition
        ↓
PPO state

Ground truth is retained ONLY as an offline reward target.

It is NOT part of the deployed PPO observation.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from receiver_ml_config import N_BANDS, normalize


FEATURES = [
    "candidate_center_mhz",
    "candidate_lower_mhz",
    "candidate_upper_mhz",
    "time_s",
    "band_scan_count",
    "band_hit_count",
    "band_miss_count",
    "band_hit_rate",
    "time_since_scan_s",
    "last_snr_db",
    "last_max_snr_db",
    "last_pdws",
    "last_frequency_mhz",
    "global_recent_hit_rate",
]


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--training",
        required=True,
    )

    ap.add_argument(
        "--model-dir",
        required=True,
    )

    ap.add_argument(
        "--dbn",
        required=True,
    )

    ap.add_argument(
        "--output",
        default="ppo_states.csv",
    )

    args = ap.parse_args()

    df = pd.read_csv(args.training)

    model_dir = Path(args.model_dir)

    models = [
        lgb.Booster(
            model_file=str(
                model_dir / f"band_{i:03d}.txt"
            )
        )
        for i in range(N_BANDS)
    ]

    transition = pd.read_csv(
        Path(args.dbn) / "transition_matrix.csv",
        index_col=0,
    ).to_numpy(float)

    parts = []

    for (config_id, step), group in df.groupby(
        ["config_id", "step"],
        sort=True,
    ):

        group = group.sort_values(
            "candidate_band"
        ).reset_index(drop=True)

        if len(group) != N_BANDS:
            raise ValueError(
                f"Expected {N_BANDS} candidate bands "
                f"for config={config_id}, step={step}; "
                f"found {len(group)}."
            )

        X = group[FEATURES]

        probabilities = np.zeros(
            N_BANDS,
            dtype=float,
        )

        for band in range(N_BANDS):

            prediction = models[band].predict(
                X.iloc[[band]]
            )

            probabilities[band] = max(
                0.0,
                float(prediction[0]),
            )

        probabilities = normalize(
            probabilities
        )

        # Predict the next hidden state.
        predicted_belief = normalize(
            transition.T @ probabilities
        )

        row = {
            "config_id": config_id,
            "step": step,
        }

        for band in range(N_BANDS):
            row[
                f"belief_{band}"
            ] = predicted_belief[band]

        # Offline reward target only.
        target_columns = [
            f"active_{i}"
            for i in range(N_BANDS)
        ]

        target = group[
            target_columns
        ].iloc[0].to_numpy(float)

        for band in range(N_BANDS):
            row[
                f"targets_{band}"
            ] = float(target[band])

        parts.append(row)

    out = pd.DataFrame(parts)

    out.to_csv(
        args.output,
        index=False,
    )

    print(
        f"PPO states: {len(out):,}"
    )

    print(
        f"Configurations: {out.config_id.nunique():,}"
    )

    print(
        f"State dimension: {N_BANDS}"
    )

    print(
        f"Action dimension: {N_BANDS}"
    )

    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()