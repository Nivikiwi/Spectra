"""
Train PPO for Smart Scan.

Custom ML path:

LightGBM -> DBN -> PPO

PPO:
    - action = one of 238 RF bands
    - observation = belief + scan history
    - ground truth = offline reward only
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium import spaces
from stable_baselines3 import PPO

from receiver_ml_config import (
    N_BANDS,
    normalize,
)


class SmartScanEnv(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        states,
        episode_len=200,
    ):

        super().__init__()

        self.states = states
        self.episode_len = episode_len

        # --------------------------------------------------------
        # PPO observation:
        #
        # 1. DBN belief
        # 2. historical hit rate
        # 3. scan recency
        #
        # Total = 3 * 238
        # --------------------------------------------------------

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(N_BANDS * 3,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(
            N_BANDS
        )

        self.rng = np.random.default_rng(
            42
        )

        self.configurations = sorted(
            states.config_id.unique()
        )

        if not self.configurations:
            raise ValueError(
                "No configurations available."
            )

    def _select_episode(self):

        for _ in range(100):

            config_id = int(
                self.rng.choice(
                    self.configurations
                )
            )

            group = (
                self.states[
                    self.states.config_id
                    == config_id
                ]
                .sort_values("step")
                .reset_index(drop=True)
            )

            if len(group) >= 2:

                max_start = max(
                    1,
                    len(group) - 1,
                )

                start = int(
                    self.rng.integers(
                        0,
                        max_start,
                    )
                )

                end = min(
                    len(group),
                    start + self.episode_len,
                )

                return group.iloc[
                    start:end
                ].copy()

        raise RuntimeError(
            "Could not create a PPO episode."
        )

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)

        self.episode = (
            self._select_episode()
        )

        self.index = 0

        self.belief = normalize(
            self.episode.iloc[
                0
            ].belief.copy()
        )

        self.scans = np.zeros(
            N_BANDS,
            dtype=float,
        )

        self.hits = np.zeros(
            N_BANDS,
            dtype=float,
        )

        self.last_scan_step = np.full(
            N_BANDS,
            -1,
            dtype=int,
        )

        return self._observation(), {}

    def _observation(self):

        hit_rate = np.divide(
            self.hits,
            self.scans,
            out=np.zeros(N_BANDS),
            where=self.scans > 0,
        )

        # Recency representation:
        # 1 = never scanned
        # approaches 0 as scan count increases.
        recency = 1.0 / (
            1.0 + self.scans
        )

        observation = np.concatenate(
            [
                self.belief,
                hit_rate,
                recency,
            ]
        )

        return observation.astype(
            np.float32
        )

    def step(self, action):

        action = int(action)

        row = self.episode.iloc[
            self.index
        ]

        targets = row.targets

        hit = float(
            targets[action] > 0
        )

        old_entropy = -np.sum(
            self.belief
            * np.log(
                self.belief + 1e-12
            )
        )

        # --------------------------------------------------------
        # Update action history
        # --------------------------------------------------------

        self.scans[action] += 1.0
        self.hits[action] += hit
        self.last_scan_step[action] = (
            self.index
        )

        # --------------------------------------------------------
        # Receiver observation update
        #
        # This is a simplified training-time observation update.
        # The actual runtime controller uses the learned DBN
        # transition/observation artifacts.
        # --------------------------------------------------------

        posterior = self.belief.copy()

        if hit > 0:

            posterior[action] = max(
                posterior[action],
                0.90,
            )

        else:

            posterior[action] *= 0.20

        posterior = normalize(
            posterior
        )

        new_entropy = -np.sum(
            posterior
            * np.log(
                posterior + 1e-12
            )
        )

        information_gain = (
            old_entropy
            - new_entropy
        )

        self.belief = posterior

        # --------------------------------------------------------
        # Reward
        # --------------------------------------------------------

        detection_reward = 1.0 * hit

        information_reward = (
            0.10 * information_gain
        )

        scan_cost = 0.02

        reward = (
            detection_reward
            + information_reward
            - scan_cost
        )

        self.index += 1

        terminated = (
            self.index
            >= len(self.episode)
        )

        truncated = False

        info = {
            "hit": hit,
            "band": action,
        }

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--states",
        required=True,
    )

    ap.add_argument(
        "--output",
        default="ppo_smart_scan",
    )

    ap.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
    )

    args = ap.parse_args()

    if args.states.endswith(".pkl"):
        df = pd.read_pickle(
            args.states
        )
    else:
        df = pd.read_csv(
            args.states
        )

    belief_columns = [
        f"belief_{i}"
        for i in range(N_BANDS)
    ]

    target_columns = [
        f"targets_{i}"
        for i in range(N_BANDS)
    ]

    missing = [
        c
        for c in belief_columns + target_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing PPO state columns: "
            + ", ".join(missing[:20])
        )

    df["belief"] = df[
        belief_columns
    ].apply(
        lambda row:
        row.to_numpy(dtype=float),
        axis=1,
    )

    df["targets"] = df[
        target_columns
    ].apply(
        lambda row:
        row.to_numpy(dtype=float),
        axis=1,
    )

    env = SmartScanEnv(
        df
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )

    model.learn(
        total_timesteps=args.timesteps
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        str(output)
    )

    print(
        "Saved PPO model:",
        output,
    )


if __name__ == "__main__":
    main()