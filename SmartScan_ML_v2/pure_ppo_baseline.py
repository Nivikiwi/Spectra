"""
PURE PPO BASELINE - reinforcement learning on its own, with no LightGBM, no DBN and no
Whittle index. PPO chooses directly among all 322 bands.

This is the "why not just use RL?" comparison: several competing approaches (DQN, LinUCB,
plain bandits) work this way. It sees exactly the same information Spectra's scheduler is
allowed to see - its own scan history, no emitter truth - so the comparison is fair.

    python pure_ppo_baseline.py --timesteps 180000        (trains, then scores configs 7-11)

State (3 numbers per band, 966 in total): hit rate, time since last scan, last result.
Action: one of 322 bands.   Reward: 1 per detection + 1 per newly found transmitter
(the same reward Spectra's PPO used).
"""
import argparse
import json
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smartscan_config import PREPARED_DIR, MODELS_DIR, REPORTS_DIR, N_BANDS, parse_config_list
from smartscan_core import load_grid
from scan_metrics import ConfigTruth, score_config, combine
from step5_evaluate_audit import emitter_cells

AGE_CAP = 2000.0


class BandHistory:
    """The only thing this baseline knows: its own scan history."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.scans = np.zeros(N_BANDS, dtype=np.float32)
        self.hits = np.zeros(N_BANDS, dtype=np.float32)
        self.last_scan = np.full(N_BANDS, -1.0, dtype=np.float32)
        self.last_hit = np.zeros(N_BANDS, dtype=np.float32)

    def observation(self, t):
        never = self.last_scan < 0
        age = np.where(never, AGE_CAP, np.minimum(t - self.last_scan, AGE_CAP))
        rate = (self.hits + 0.02) / (self.scans + 1.0)
        return np.concatenate([rate, np.log1p(age) / np.log1p(AGE_CAP), self.last_hit]).astype(np.float32)

    def update(self, t, band, hit):
        self.scans[band] += 1
        self.hits[band] += hit
        self.last_scan[band] = t
        self.last_hit[band] = hit


class PureBandEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, configs, episode_len=256, seed=0):
        super().__init__()
        self.grids = [load_grid(PREPARED_DIR / f"config_{c}.npz") for c in configs]
        self.cells = [emitter_cells(g)[0] for g in self.grids]
        self.episode_len = episode_len
        self.hist = BandHistory()
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(3 * N_BANDS,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_BANDS)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        k = int(self.rng.integers(len(self.grids)))
        self.g, self.g_cells = self.grids[k], self.cells[k]
        D = self.g["hit"].shape[0]
        self.t = int(self.rng.integers(0, max(1, D - self.episode_len)))
        self.end = min(D, self.t + self.episode_len)
        self.found = set()
        self.hist.reset()
        return self.hist.observation(self.t), {}

    def step(self, action):
        band = int(action)
        t = self.t
        hit = int(self.g["hit"][t, band])
        new = self.g_cells.get((t, band), set()) - self.found if hit else set()
        self.found |= new
        self.hist.update(t, band, hit)
        self.t += 1
        done = self.t >= self.end
        obs = np.zeros(3 * N_BANDS, dtype=np.float32) if done else self.hist.observation(self.t)
        return obs, float(hit) + float(len(new)), done, False, {}


def evaluate(model, cfgs):
    hist = BandHistory()
    parts = []
    for cfg in cfgs:
        g = load_grid(PREPARED_DIR / f"config_{cfg}.npz")
        truth = ConfigTruth(g)
        hist.reset()
        scans = []
        for t in range(g["hit"].shape[0]):
            a, _ = model.predict(hist.observation(t), deterministic=True)
            b = int(np.asarray(a).item())
            h = int(g["hit"][t, b])
            hist.update(t, b, h)
            scans.append((float(g["dwell_time_s"][t]), 0.010, b, h, float(g["pdws"][t, b])))
        parts.append(score_config(truth, scans))
    return combine(parts)


def main():
    from stable_baselines3 import PPO
    ap = argparse.ArgumentParser(description="Pure PPO baseline (no LightGBM / DBN / Whittle)")
    rep = json.loads((MODELS_DIR / "lightgbm_report.json").read_text())
    ap.add_argument("--train-configs", default=",".join(map(str, rep["train_configs"])))
    ap.add_argument("--test-configs", default=",".join(map(str, rep.get("test_configs", [7, 8, 9, 10, 11]))))
    ap.add_argument("--timesteps", type=int, default=180000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train_cfg = parse_config_list(args.train_configs)
    test_cfg = [c for c in parse_config_list(args.test_configs) if (PREPARED_DIR / f"config_{c}.npz").exists()]

    t0 = time.time()
    print(f"Pure PPO: {N_BANDS} actions, {3 * N_BANDS} inputs, training on configs {train_cfg} "
          f"for {args.timesteps:,} steps ...")
    model = PPO("MlpPolicy", PureBandEnv(train_cfg, seed=args.seed), n_steps=512, batch_size=128,
                n_epochs=10, learning_rate=3e-4, gamma=0.95, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128]), seed=args.seed, verbose=0, device="cpu")
    done = 0
    while done < args.timesteps:
        n = min(20480, args.timesteps - done)
        model.learn(total_timesteps=n, reset_num_timesteps=False)
        done += n
        print(f"  {done:,}/{args.timesteps:,}  ({time.time() - t0:.0f} s)")
    m = evaluate(model, test_cfg)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"algorithm": "Pure PPO (no LightGBM/DBN/Whittle)", "train_configs": train_cfg,
           "test_configs": test_cfg, "timesteps": args.timesteps, "seed": args.seed,
           "transmitters_found": m["transmitters_found"], "detectable": m["transmitters_detectable"],
           "time_to_first_detection_s": round(m["time_to_first_detection_s"], 2),
           "detections_per_s": round(m["detections_per_s"], 2), "pd_pct": round(m["pd_pct"], 2),
           "pfa_pct": round(m["pfa_pct"], 3), "pulses": m["pdws_intercepted"]}
    (REPORTS_DIR / "pure_ppo_baseline.json").write_text(json.dumps(out, indent=2))
    print(f"\nPURE PPO on unseen configs {test_cfg}: {m['transmitters_found']}/{m['transmitters_detectable']} "
          f"transmitters found, first detection {m['time_to_first_detection_s']:.2f} s, "
          f"{m['detections_per_s']:.2f} detections/s")
    print(f"Saved {REPORTS_DIR / 'pure_ppo_baseline.json'}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
