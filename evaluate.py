"""
evaluate.py
-----------
Loads a PREVIOUSLY TRAINED PPO model from disk (the checkpoint saved by
train.py) and runs it in the environment to measure how well it performs.

This proves the "load checkpoint -> evaluate again" half of the pipeline.

Run with:
    python evaluate.py

(Run train.py first, or this will fail to find the saved model file.)
"""

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env_utils import make_env

MODEL_PATH = "models/ppo_cartpole"   # SB3 automatically looks for ppo_cartpole.zip
ENV_NAME = "CartPole-v1"
N_EVAL_EPISODES = 10


def main():
    # 1. Create a fresh environment to evaluate in
    env = make_env(ENV_NAME)

    # 2. Load the trained model back from disk
    print(f"Loading model from {MODEL_PATH}.zip ...")
    model = PPO.load(MODEL_PATH, env=env)
    print("Model loaded successfully.")

    # 3. Run the agent for N_EVAL_EPISODES episodes and measure average reward.
    #    For CartPole-v1, a well-trained agent should score close to 500
    #    (the maximum possible reward per episode).
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=N_EVAL_EPISODES
    )
    print(f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 4. Run a single demo episode manually, so you can see step-by-step behavior
    obs, info = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Single demo episode total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
