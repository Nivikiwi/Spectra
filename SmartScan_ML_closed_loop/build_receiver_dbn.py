"""
Build the temporal DBN belief artifacts.

This implementation uses:
    - a 238-state transition matrix
    - receiver-observation likelihoods

Ground truth is used OFFLINE only to estimate the artifacts.
It is never required by the online controller.

Runtime:
    LightGBM probabilities
        -> DBN temporal prediction
        -> observation update
        -> posterior belief
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from receiver_ml_config import N_BANDS, normalize


def safe_float(value, default=0.0):
    if pd.isna(value):
        return default
    return float(value)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
        help="receiver_lgbm_training.csv",
    )

    ap.add_argument(
        "--output-dir",
        default="receiver_dbn",
    )

    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = [
        "config_id",
        "step",
        "candidate_band",
        "band_hit_rate",
        "last_snr_db",
        "last_max_snr_db",
        "last_pdws",
    ]

    target_cols = [f"active_{i}" for i in range(N_BANDS)]

    missing = [
        c for c in required + target_cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required DBN columns: " + ", ".join(missing)
        )

    # ------------------------------------------------------------
    # Hidden state
    #
    # Each observation corresponds to a belief over which RF band
    # is currently active.
    #
    # Ground truth is used only to learn the offline transition
    # statistics.
    # ------------------------------------------------------------

    state_cols = ["config_id", "step"] + target_cols

    states = (
        df[state_cols]
        .drop_duplicates(["config_id", "step"])
        .sort_values(["config_id", "step"])
        .copy()
    )

    target_matrix = states[target_cols].to_numpy(float)

    states["state"] = np.argmax(
        target_matrix,
        axis=1,
    )

    # ------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------

    transition = np.zeros(
        (N_BANDS, N_BANDS),
        dtype=float,
    )

    for _, group in states.groupby("config_id", sort=True):

        sequence = group["state"].to_numpy(dtype=int)

        if len(sequence) < 2:
            continue

        for previous_state, next_state in zip(
            sequence[:-1],
            sequence[1:],
        ):
            transition[
                previous_state,
                next_state
            ] += 1.0

    for i in range(N_BANDS):

        if transition[i].sum() > 0:
            transition[i] = normalize(
                transition[i]
            )
        else:
            # If a state never transitions during training,
            # keep it persistent.
            transition[i, i] = 1.0

    transition_df = pd.DataFrame(
        transition,
        index=np.arange(N_BANDS),
        columns=np.arange(N_BANDS),
    )

    transition_df.to_csv(
        out / "transition_matrix.csv",
        index_label="from_band_index",
    )

    # ------------------------------------------------------------
    # Observation model
    #
    # We estimate:
    #
    #   P(observation band | hidden active band)
    #
    # using the band that was actually scanned.
    #
    # This is deliberately separate from the transition model.
    # ------------------------------------------------------------

    state_lookup = states.set_index(
        ["config_id", "step"]
    )["state"]

    scan = df[
        [
            "config_id",
            "step",
            "candidate_band",
            "band_hit_rate",
            "last_snr_db",
            "last_max_snr_db",
            "last_pdws",
        ]
    ].copy()

    scan["state"] = [
        state_lookup.get(
            (row.config_id, row.step),
            0,
        )
        for row in scan.itertuples()
    ]

    observation_counts = np.zeros(
        (N_BANDS, N_BANDS),
        dtype=float,
    )

    for row in scan.itertuples():

        state = int(row.state)
        band = int(row.candidate_band)

        if 0 <= state < N_BANDS and 0 <= band < N_BANDS:
            observation_counts[state, band] += 1.0

    observation = np.zeros_like(
        observation_counts
    )

    global_counts = observation_counts.sum(axis=0)

    if global_counts.sum() > 0:
        global_distribution = normalize(
            global_counts
        )
    else:
        global_distribution = np.ones(
            N_BANDS
        ) / N_BANDS

    for state in range(N_BANDS):

        row = observation_counts[state]

        if row.sum() > 0:
            empirical = normalize(row)

            # Smooth sparse states toward the global distribution.
            strength = row.sum()

            weight = strength / (
                strength + 20.0
            )

            observation[state] = normalize(
                weight * empirical
                + (1.0 - weight) * global_distribution
            )

        else:
            observation[state] = global_distribution

    observation_df = pd.DataFrame(
        observation,
        index=np.arange(N_BANDS),
        columns=np.arange(N_BANDS),
    )

    observation_df.to_csv(
        out / "observation_model.csv",
        index_label="state_index",
    )

    # ------------------------------------------------------------
    # Also save metadata.
    # ------------------------------------------------------------

    metadata = {
        "n_bands": N_BANDS,
        "model_type": "temporal_probabilistic_belief",
        "transition_file": "transition_matrix.csv",
        "observation_file": "observation_model.csv",
        "ground_truth_used_offline": True,
        "ground_truth_used_online": False,
    }

    pd.Series(metadata).to_json(
        out / "metadata.json",
        indent=2,
    )

    print(
        f"DBN transition matrix: {transition.shape}"
    )

    print(
        f"DBN observation model: {observation.shape}"
    )

    print(
        f"Configurations: {df.config_id.nunique()}"
    )

    print(
        f"States: {len(states):,}"
    )

    print(
        "Saved DBN artifacts to:",
        out,
    )


if __name__ == "__main__":
    main()