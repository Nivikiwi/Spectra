"""
ppo/evaluate.py
---------------
Loads a trained PPO checkpoint and evaluates it over many episodes,
alongside a random-policy baseline for comparison.

WHAT'S NEW vs the Day-2 version:
  1. Runs 50-100+ episodes instead of a single one, tracking reward
     and episode length per episode.
  2. Also runs a completely random policy over the same number of
     episodes, so we have a baseline: Random vs PPO.
  3. Prints mean/std reward, mean episode length, and any info dict
     contents for both policies.

USAGE:
    python ppo/evaluate.py
    python ppo/evaluate.py --checkpoint ppo/checkpoints/ppo_scan_v0 --episodes 100
"""

import argparse
import os
import sys
import numpy as np

# Make sure Python can see the folder ABOVE this script (where rl_env/ lives),
# no matter which folder you run this command from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from rl_env.envs.scan_env import ScanEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO on ScanEnv vs a random baseline")
    parser.add_argument("--bands", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default="ppo/checkpoints/ppo_scan_v0",
                         help="Path (no .zip) to the checkpoint to load")
    parser.add_argument("--episodes", type=int, default=100,
                         help="Number of evaluation episodes (default: 100)")
    return parser.parse_args()


def run_episodes(env, n_episodes, policy_fn, deterministic_label):
    """
    Runs `n_episodes` full episodes using `policy_fn(obs) -> action`.
    Returns a dict with reward/length stats and any collected info.
    """
    episode_rewards = []
    episode_lengths = []
    collected_infos = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if info:
            collected_infos.append(info)

    results = {
        "label": deterministic_label,
        "episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "infos": collected_infos,
    }
    return results


def print_results(results):
    print(f"\n----- {results['label']} -----")
    print(f"Episodes: {results['episodes']}")
    print(f"Mean reward: {results['mean_reward']:.3f}")
    print(f"Std reward: {results['std_reward']:.3f}")
    print(f"Mean episode length: {results['mean_length']:.1f}")
    if results["infos"]:
        # Print whatever the environment's info dict contains, just once as an example
        print(f"Example info dict (from one episode): {results['infos'][0]}")
    else:
        print("No extra info returned by the environment.")


def main():
    args = parse_args()
    env = ScanEnv(num_bands=args.bands)

    # ---------- 1. Evaluate PPO ----------
    print(f"Loading checkpoint from {args.checkpoint}.zip ...")
    try:
        model = PPO.load(args.checkpoint, env=env)
        checkpoint_loaded = True
        print("Checkpoint loaded successfully.")
    except Exception as e:
        checkpoint_loaded = False
        print(f"PROBLEM loading checkpoint: {e}")
        print("\nCheckpoint reload: NO")
        print("Inference: NO")
        return

    def ppo_policy(obs):
        action, _states = model.predict(obs, deterministic=True)
        return action

    inference_ok = True
    try:
        ppo_results = run_episodes(env, args.episodes, ppo_policy, "PPO POLICY (deterministic)")
    except Exception as e:
        inference_ok = False
        print(f"PROBLEM during PPO inference: {e}")
        ppo_results = None

    # ---------- 2. Evaluate random-policy baseline ----------
    def random_policy(obs):
        return env.action_space.sample()

    random_results = run_episodes(env, args.episodes, random_policy, "RANDOM POLICY (baseline)")

    # ---------- 3. Print everything ----------
    if ppo_results:
        print_results(ppo_results)
    print_results(random_results)

    print("\n----- COMPARISON -----")
    if ppo_results:
        print(f"PPO mean reward:    {ppo_results['mean_reward']:.3f}")
    print(f"Random mean reward: {random_results['mean_reward']:.3f}")
    print("(Not expected to differ much yet - the reward is still a stub.)")

    print("\n----- EVALUATE SUMMARY (copy into your report) -----")
    print(f"Checkpoint reload: {'YES' if checkpoint_loaded else 'NO'}")
    print(f"Inference: {'YES' if inference_ok else 'NO'}")
    if ppo_results:
        print(f"PPO mean reward: {ppo_results['mean_reward']:.3f}")
    print(f"Random mean reward: {random_results['mean_reward']:.3f}")
    print(f"Number of evaluation episodes: {args.episodes}")


if __name__ == "__main__":
    main()
