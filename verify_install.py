"""
verify_install.py
------------------
Run this FIRST, before train.py.

Purpose: confirm that gymnasium, stable-baselines3, and torch are all
installed correctly, by creating CartPole-v1, taking a few random
actions, and printing the results. This does NOT train anything —
it just proves your environment is set up correctly.

Run with:
    python verify_install.py

If this prints "INSTALL OK" at the end with no errors above it,
you are ready to move on to train.py.
"""

import gymnasium as gym
import torch
from stable_baselines3 import PPO


def main():
    print(f"Torch version: {torch.__version__}")
    print(f"Torch sees a GPU: {torch.cuda.is_available()}")  # False is fine, CPU works too

    # Create the CartPole environment
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    print(f"Initial observation (cart position, cart velocity, pole angle, pole velocity): {obs}")

    # Take 5 random actions just to prove step() works
    for i in range(5):
        action = env.action_space.sample()  # a random action: 0 = push left, 1 = push right
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward}, done={terminated or truncated}")
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    # Prove Stable-Baselines3 can at least construct a PPO agent (not train it yet)
    test_env = gym.make("CartPole-v1")
    model = PPO(policy="MlpPolicy", env=test_env, verbose=0)
    print("PPO model object created successfully.")
    test_env.close()

    print("\nINSTALL OK - gymnasium, torch, and stable-baselines3 are all working.")


if __name__ == "__main__":
    main()
