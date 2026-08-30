"""
train.py
--------
Trains a PPO agent on our environment (CartPole-v1 for now) and saves
the trained model to disk as a checkpoint (.zip file).

Run with:
    python train.py

Expect this to take a few minutes on a normal laptop CPU.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env_utils import make_env

# ---------------------- SETTINGS you can tweak ----------------------
ENV_NAME = "CartPole-v1"
TOTAL_TIMESTEPS = 50_000            # how many steps the agent practices
MODEL_SAVE_PATH = "models/ppo_cartpole"   # SB3 will add ".zip" automatically
LOG_DIR = "logs"
# ----------------------------------------------------------------------


def main():
    # 1. Make sure the folders we need actually exist (create if missing)
    os.makedirs("models", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 2. Create the training environment.
    #    make_vec_env is SB3's helper that wraps our env so PPO can use it.
    #    n_envs=1 means "run 1 copy of the environment" (fine for CartPole).
    env = make_vec_env(lambda: make_env(ENV_NAME), n_envs=1)

    # 3. Create the PPO agent.
    #    "MlpPolicy" = a small neural network (Multi-Layer Perceptron) as
    #    the agent's brain. verbose=1 prints training progress to the screen.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # 4. Train the agent. This is where the actual learning happens.
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 5. Save the trained model to disk so we can reload it later.
    model.save(MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to: {MODEL_SAVE_PATH}.zip")

    env.close()


if __name__ == "__main__":
    main()
