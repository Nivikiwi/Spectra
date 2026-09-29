"""
Whittle-style baseline scheduler.

This is an independent baseline for comparison with the
LightGBM -> DBN -> PPO custom ML pipeline.

It maintains persistent per-band statistics.
"""

import numpy as np

from receiver_ml_config import N_BANDS


class WhittleScheduler:

    def __init__(
        self,
        exploration=0.25,
        age_bonus=0.02,
    ):

        self.exploration = float(
            exploration
        )

        self.age_bonus = float(
            age_bonus
        )

        self.scans = np.zeros(
            N_BANDS,
            dtype=float,
        )

        self.hits = np.zeros(
            N_BANDS,
            dtype=float,
        )

        self.last_selected = np.full(
            N_BANDS,
            -1,
            dtype=int,
        )

        self.step_count = 0

    def select(self, belief):

        belief = np.asarray(
            belief,
            dtype=float,
        )

        if len(belief) != N_BANDS:
            raise ValueError(
                f"Expected {N_BANDS} belief values."
            )

        belief = np.clip(
            belief,
            0.0,
            None,
        )

        if belief.sum() > 0:
            belief = (
                belief
                / belief.sum()
            )

        hit_rate = np.divide(
            self.hits,
            self.scans,
            out=np.zeros(N_BANDS),
            where=self.scans > 0,
        )

        uncertainty = 1.0 / np.sqrt(
            1.0 + self.scans
        )

        age = np.where(
            self.last_selected < 0,
            1.0,
            np.clip(
                (
                    self.step_count
                    - self.last_selected
                ) / 10.0,
                0.0,
                1.0,
            ),
        )

        # Practical index approximation.
        index = (
            0.60 * belief
            + 0.15 * hit_rate
            + self.exploration
            * 0.20
            * uncertainty
            + self.age_bonus
            * 0.05
            * age
        )

        return int(
            np.argmax(index)
        )

    def update(
        self,
        band,
        hit,
    ):

        band = int(band)

        if not 0 <= band < N_BANDS:
            raise ValueError(
                f"Band must be 0..{N_BANDS - 1}"
            )

        self.scans[band] += 1.0
        self.hits[band] += float(
            hit
        )

        self.last_selected[band] = (
            self.step_count
        )

        self.step_count += 1