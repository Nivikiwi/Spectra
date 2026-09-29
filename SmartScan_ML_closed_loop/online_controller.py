"""
Smart Scan online controller.

CUSTOM:
    receiver observations
        -> LightGBM
        -> DBN
        -> PPO
        -> ml_scan_command.txt

BASELINES:
    receiver observations -> Whittle
    receiver observations -> Round Robin

Ground truth is NEVER read by this controller.
"""

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from receiver_ml_config import (
    N_BANDS,
    WINDOW_CENTRES_MHZ,
    DEFAULT_DWELL_S,
    normalize,
)

from whittle_scheduler import (
    WhittleScheduler,
)

from round_robin_scheduler import (
    RoundRobinScheduler,
)


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


class Runtime:

    def __init__(
        self,
        model_dir,
        dbn_dir,
        ppo_model=None,
    ):

        model_dir = Path(
            model_dir
        )

        dbn_dir = Path(
            dbn_dir
        )

        # --------------------------------------------------------
        # LightGBM
        # --------------------------------------------------------

        self.models = [
            lgb.Booster(
                model_file=str(
                    model_dir
                    / f"band_{i:03d}.txt"
                )
            )
            for i in range(N_BANDS)
        ]

        # --------------------------------------------------------
        # DBN artifacts
        # --------------------------------------------------------

        self.transition = pd.read_csv(
            dbn_dir
            / "transition_matrix.csv",
            index_col=0,
        ).to_numpy(float)

        self.observation = pd.read_csv(
            dbn_dir
            / "observation_model.csv",
            index_col=0,
        ).to_numpy(float)

        if self.transition.shape != (
            N_BANDS,
            N_BANDS,
        ):
            raise ValueError(
                "Invalid DBN transition matrix."
            )

        if self.observation.shape != (
            N_BANDS,
            N_BANDS,
        ):
            raise ValueError(
                "Invalid DBN observation model."
            )

        # --------------------------------------------------------
        # PPO
        # --------------------------------------------------------

        self.ppo = None

        if ppo_model:

            self.ppo = PPO.load(
                ppo_model
            )

        # --------------------------------------------------------
        # Temporal belief
        # --------------------------------------------------------

        self.belief = np.ones(
            N_BANDS,
            dtype=float,
        ) / N_BANDS

        self.history = []

        # --------------------------------------------------------
        # Baseline state
        #
        # IMPORTANT:
        # These objects are persistent.
        # They are NOT recreated for every decision.
        # --------------------------------------------------------

        self.whittle = (
            WhittleScheduler()
        )

        self.round_robin = (
            RoundRobinScheduler()
        )

    # ============================================================
    # Candidate features
    # ============================================================

    def feature_frame(self, current_time):

        history = pd.DataFrame(
            self.history
        )

        rows = []

        for band, center in enumerate(
            WINDOW_CENTRES_MHZ
        ):

            if history.empty:

                band_history = history

            else:

                band_history = history[
                    np.isclose(
                        history[
                            "scan_center_mhz"
                        ].astype(float),
                        center,
                        atol=1e-6,
                    )
                ]

            scans = len(
                band_history
            )

            if scans:

                hits = int(
                    band_history[
                        "hit"
                    ]
                    .fillna(0)
                    .sum()
                )

                last = (
                    band_history
                    .iloc[-1]
                )

                last_time = float(
                    last.scan_time_s
                )

                age = max(
                    0.0,
                    current_time
                    - last_time,
                )

                last_snr = (
                    0.0
                    if pd.isna(
                        last.mean_snr_db
                    )
                    else float(
                        last.mean_snr_db
                    )
                )

                last_max_snr = (
                    0.0
                    if pd.isna(
                        last.max_snr_db
                    )
                    else float(
                        last.max_snr_db
                    )
                )

                last_pdws = (
                    0.0
                    if pd.isna(
                        last.total_pdws
                    )
                    else float(
                        last.total_pdws
                    )
                )

                last_frequency = (
                    center
                    if pd.isna(
                        last.mean_frequency_mhz
                    )
                    else float(
                        last.mean_frequency_mhz
                    )
                )

            else:

                hits = 0
                age = 1e6
                last_snr = 0.0
                last_max_snr = 0.0
                last_pdws = 0.0
                last_frequency = center

            if history.empty:

                global_hit_rate = 0.0

            else:

                recent_hits = (
                    history[
                        "hit"
                    ]
                    .fillna(0)
                    .tail(20)
                )

                global_hit_rate = float(
                    recent_hits.mean()
                )

            rows.append(
                {
                    "candidate_center_mhz":
                        float(center),

                    "candidate_lower_mhz":
                        float(center - 25.0),

                    "candidate_upper_mhz":
                        float(center + 25.0),

                    "time_s":
                        float(current_time),

                    "band_scan_count":
                        scans,

                    "band_hit_count":
                        hits,

                    "band_miss_count":
                        scans - hits,

                    "band_hit_rate":
                        hits / scans
                        if scans
                        else 0.0,

                    "time_since_scan_s":
                        age,

                    "last_snr_db":
                        last_snr,

                    "last_max_snr_db":
                        last_max_snr,

                    "last_pdws":
                        last_pdws,

                    "last_frequency_mhz":
                        last_frequency,

                    "global_recent_hit_rate":
                        global_hit_rate,
                }
            )

        return pd.DataFrame(
            rows
        )[FEATURES]

    # ============================================================
    # LightGBM
    # ============================================================

    def lightgbm_probability(
        self,
        current_time,
    ):

        X = self.feature_frame(
            current_time
        )

        probability = np.zeros(
            N_BANDS,
            dtype=float,
        )

        for band in range(N_BANDS):

            prediction = self.models[
                band
            ].predict(
                X.iloc[[band]]
            )

            probability[band] = max(
                0.0,
                float(prediction[0]),
            )

        return normalize(
            probability
        )

    # ============================================================
    # DBN
    # ============================================================

    def dbn_predict(
        self,
        lightgbm_probability,
    ):

        predicted = normalize(
            self.transition.T
            @ lightgbm_probability
        )

        return predicted

    def dbn_observation_update(
        self,
        predicted,
        scanned_band,
        hit,
    ):

        # The learned observation model gives:
        #
        # P(observed scan band | hidden state)
        #
        # Use the scanned band as likelihood evidence.
        likelihood = self.observation[
            :,
            int(scanned_band),
        ].copy()

        likelihood = np.clip(
            likelihood,
            1e-12,
            None,
        )

        # Receiver hit/miss evidence.
        #
        # A hit strengthens the selected band.
        # A miss weakens it.
        evidence = np.ones(
            N_BANDS,
            dtype=float,
        )

        if int(hit) == 1:

            evidence[
                int(scanned_band)
            ] = 4.0

        else:

            evidence[
                int(scanned_band)
            ] = 0.25

        posterior = (
            predicted
            * likelihood
            * evidence
        )

        return normalize(
            posterior
        )

    # ============================================================
    # PPO state
    # ============================================================

    def ppo_observation(self):

        history = pd.DataFrame(
            self.history
        )

        if history.empty:

            hit_rate = np.zeros(
                N_BANDS
            )

            recency = np.ones(
                N_BANDS
            )

        else:

            hit_rate = np.zeros(
                N_BANDS
            )

            scans = np.zeros(
                N_BANDS
            )

            for band, center in enumerate(
                WINDOW_CENTRES_MHZ
            ):

                g = history[
                    np.isclose(
                        history[
                            "scan_center_mhz"
                        ].astype(float),
                        center,
                        atol=1e-6,
                    )
                ]

                scans[band] = len(g)

                if len(g):

                    hit_rate[band] = float(
                        g[
                            "hit"
                        ]
                        .fillna(0)
                        .mean()
                    )

            recency = 1.0 / (
                1.0 + scans
            )

        return np.concatenate(
            [
                self.belief,
                hit_rate,
                recency,
            ]
        ).astype(
            np.float32
        )

    # ============================================================
    # Decision
    # ============================================================

    def decide(
        self,
        current_time,
        policy,
    ):

        probability = (
            self.lightgbm_probability(
                current_time
            )
        )

        predicted = (
            self.dbn_predict(
                probability
            )
        )

        self.belief = predicted

        if policy == "custom":

            if self.ppo is None:

                raise RuntimeError(
                    "Custom policy requires "
                    "--ppo-model."
                )

            observation = (
                self.ppo_observation()
            )

            action, _ = (
                self.ppo.predict(
                    observation,
                    deterministic=True,
                )
            )

            band = int(
                np.asarray(action).item()
            )

            return band, self.belief

        # --------------------------------------------------------
        # Baseline: Whittle
        # --------------------------------------------------------

        if policy == "whittle":

            band = (
                self.whittle.select(
                    self.belief
                )
            )

            return band, self.belief

        # --------------------------------------------------------
        # Baseline: Round Robin
        # --------------------------------------------------------

        if policy == "round_robin":

            band = (
                self.round_robin.select(
                    self.belief
                )
            )

            return band, self.belief

        raise ValueError(
            f"Unknown policy: {policy}"
        )

    # ============================================================
    # Update after physical scan
    # ============================================================

    def update_after_scan(
        self,
        observation,
        selected_band,
    ):

        hit = (
            0
            if pd.isna(
                observation.hit
            )
            else int(
                observation.hit
            )
        )

        # Update DBN posterior using the actual
        # receiver observation.
        self.belief = (
            self.dbn_observation_update(
                self.belief,
                selected_band,
                hit,
            )
        )

        # Baseline statistics.
        self.whittle.update(
            selected_band,
            hit,
        )

        self.round_robin.update(
            selected_band,
            hit,
        )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--csv",
        required=True,
    )

    ap.add_argument(
        "--command",
        required=True,
    )

    ap.add_argument(
        "--model-dir",
        required=True,
    )

    ap.add_argument(
        "--dbn-dir",
        required=True,
    )

    ap.add_argument(
        "--ppo-model",
        default=None,
    )

    ap.add_argument(
        "--policy",
        choices=[
            "custom",
            "whittle",
            "round_robin",
        ],
        required=True,
    )

    ap.add_argument(
        "--audit",
        default="scan_history.csv",
    )

    args = ap.parse_args()

    runtime = Runtime(
        model_dir=args.model_dir,
        dbn_dir=args.dbn_dir,
        ppo_model=args.ppo_model,
    )

    csv_path = Path(
        args.csv
    )

    command_path = Path(
        args.command
    )

    audit_path = Path(
        args.audit
    )

    seen = 0

    while True:

        try:

            df = pd.read_csv(
                csv_path
            )

        except Exception:

            time.sleep(0.2)
            continue

        if len(df) <= seen:

            time.sleep(0.2)
            continue

        new_rows = df.iloc[
            seen:
        ]

        for _, row in new_rows.iterrows():

            runtime.history.append(
                row.to_dict()
            )

            seen += 1

            decision_time = (
                float(
                    row.scan_time_s
                )
                + float(
                    row.dwell_s
                )
            )

            # ----------------------------------------------------
            # Select next band
            # ----------------------------------------------------

            band, belief = (
                runtime.decide(
                    decision_time,
                    args.policy,
                )
            )

            center = float(
                WINDOW_CENTRES_MHZ[
                    band
                ]
            )

            lower = (
                center - 25.0
            )

            upper = (
                center + 25.0
            )

            # ----------------------------------------------------
            # Write command for Scilab
            # ----------------------------------------------------

            command_path.write_text(
                f"{center:.6f} "
                f"{DEFAULT_DWELL_S:.6f}\n",
                encoding="utf-8",
            )

            # ----------------------------------------------------
            # Update temporal belief after observing
            # the scan that just completed.
            # ----------------------------------------------------

            runtime.update_after_scan(
                row,
                band,
            )

            # ----------------------------------------------------
            # Audit
            # ----------------------------------------------------

            hit = (
                0
                if pd.isna(row.hit)
                else int(row.hit)
            )

            audit_row = {
                "decision_time_s":
                    decision_time,

                "policy":
                    args.policy,

                "selected_band_index":
                    band,

                "selected_center_mhz":
                    center,

                "selected_lower_mhz":
                    lower,

                "selected_upper_mhz":
                    upper,

                "belief_probability":
                    float(
                        belief[band]
                    ),

                "observation_hit":
                    hit,

                "mean_snr_db":
                    row.mean_snr_db,

                "total_pdws":
                    row.total_pdws,
            }

            pd.DataFrame(
                [audit_row]
            ).to_csv(
                audit_path,
                index=False,
                mode="a",
                header=not audit_path.exists(),
            )

        time.sleep(0.2)


if __name__ == "__main__":
    main()