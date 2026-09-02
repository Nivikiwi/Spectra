"""
ppo/train.py
------------
Configurable PPO training script for ScanEnv.

WHAT'S NEW vs the Day-2 version:
  1. Command-line arguments instead of hardcoded settings
     (--bands, --timesteps, --checkpoint-freq)
  2. Periodic checkpoints saved during training, not just at the end
     (ppo_scan_10000_steps.zip, ppo_scan_20000_steps.zip, ...)
  3. Proper environment validation using Stable-Baselines3's own
     check_env() function, in addition to our custom interface checks.

USAGE:
    python ppo/train.py
    python ppo/train.py --bands 20 --timesteps 50000
    python ppo/train.py --bands 30 --timesteps 100000 --checkpoint-freq 5000

Run from the folder that has rl_env/ next to it (see README.md).
"""

import argparse
import os
import sys
import numpy as np

# Make sure Python can see the folder ABOVE this script (where rl_env/ lives),
# no matter which folder you run this command from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_env.envs.scan_env import ScanEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on ScanEnv")
    parser.add_argument("--bands", type=int, default=20,
                         help="Number of spectrum bands (default: 20)")
    parser.add_argument("--timesteps", type=int, default=50_000,
                         help="Total training timesteps (default: 50000)")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000,
                         help="Save a checkpoint every N timesteps (default: 10000)")
    parser.add_argument("--checkpoint-dir", type=str, default="ppo/checkpoints",
                         help="Folder to save checkpoints in")
    parser.add_argument("--final-name", type=str, default="ppo_scan_v0",
                         help="Filename (no .zip) for the final saved model")
    return parser.parse_args()


def custom_interface_check(env):
    """
    Our own quick checks, matching the specific things the team wants
    verified: reset()/step() formats, observation shape/dtype, action
    validity, no NaN/Inf, episodes terminate correctly.
    Returns True/False.
    """
    ok = True
    print("\n----- CUSTOM INTERFACE CHECK -----")

    reset_result = env.reset()
    if not (isinstance(reset_result, tuple) and len(reset_result) == 2):
        print("PROBLEM: reset() should return (observation, info).")
        ok = False
    else:
        obs, info = reset_result
        obs_arr = np.array(obs)
        print(f"Observation shape: {obs_arr.shape}, dtype: {obs_arr.dtype}")
        if not env.observation_space.contains(obs):
            print("PROBLEM: observation from reset() is not inside observation_space.")
            ok = False
        if np.isnan(obs_arr.astype(np.float64)).any() or np.isinf(obs_arr.astype(np.float64)).any():
            print("PROBLEM: observation contains NaN/Inf.")
            ok = False

    action = env.action_space.sample()
    step_result = env.step(action)
    if not (isinstance(step_result, tuple) and len(step_result) == 5):
        print("PROBLEM: step() should return exactly 5 values.")
        ok = False
    else:
        obs, reward, terminated, truncated, info = step_result
        if np.isnan(np.float64(reward)) or np.isinf(np.float64(reward)):
            print("PROBLEM: reward is NaN/Inf.")
            ok = False

    # Check that an episode actually terminates within a reasonable number of steps
    env.reset()
    ended = False
    for _ in range(2000):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            ended = True
            break
    if not ended:
        print("PROBLEM: episode did not terminate/truncate within 2000 steps.")
        ok = False

    print(f"----- CUSTOM CHECK {'PASSED' if ok else 'FAILED'} -----")
    return ok


def validate_environment(env):
    """
    Runs BOTH our custom checks and Stable-Baselines3's official
    check_env() utility. Returns True only if both pass.
    """
    custom_ok = custom_interface_check(env)

    print("\n----- STABLE-BASELINES3 check_env() -----")
    sb3_ok = True
    try:
        check_env(env, warn=True)
        print("check_env(): no errors reported.")
    except Exception as e:
        sb3_ok = False
        print(f"check_env() FAILED: {e}")

    overall_ok = custom_ok and sb3_ok
    print(f"\n----- ENVIRONMENT VALIDATION: {'PASS' if overall_ok else 'FAIL'} -----\n")
    return overall_ok


def main():
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Create environment and validate it BEFORE spending time training
    env = ScanEnv(num_bands=args.bands)
    validation_passed = validate_environment(env)
    if not validation_passed:
        print("Environment validation FAILED. Fix the PROBLEM lines above before training. Stopping.")
        return

    env.reset()

    # 2. Set up periodic checkpointing
    #    CheckpointCallback saves a file every `save_freq` steps, named
    #    like ppo_scan_10000_steps.zip, ppo_scan_20000_steps.zip, etc.
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.checkpoint_dir,
        name_prefix="ppo_scan",
    )

    # 3. Create and train the PPO agent
    model = PPO(policy="MlpPolicy", env=env, verbose=1)

    print(f"Training for {args.timesteps} timesteps "
          f"(checkpoint every {args.checkpoint_freq} steps)...")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    print("Training complete.")

    # 4. Save the final model under a fixed name, separate from the periodic checkpoints
    final_path = os.path.join(args.checkpoint_dir, args.final_name)
    model.save(final_path)
    print(f"Final model saved to: {final_path}.zip")

    print("\n----- TRAIN SUMMARY -----")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Timesteps trained: {args.timesteps}")
    print("PPO training: YES")
    print("Checkpointing: YES")
    print(f"Environment validation: {'PASS' if validation_passed else 'FAIL'}")


if __name__ == "__main__":
    main()
