"""
env_utils.py
-------------
IMPORTANT FILE - this is the ONLY place in the whole project where we
create the environment.

Right now, make_env() creates CartPole-v1 (a standard test environment,
just to prove the pipeline works).

LATER, when Chahek hands you her spectrum-search environment (state
space, action space, reward function), you will only need to edit the
inside of make_env() to build and return HER environment instead.
train.py and evaluate.py will not need to change at all, because they
only ever call make_env() and don't care what's inside it.
This is what "modular" means in practice.
"""

import gymnasium as gym


def make_env(env_name: str = "CartPole-v1", render_mode: str = None):
    """
    Creates and returns a Gymnasium-compatible environment.

    Parameters
    ----------
    env_name : str
        Name of the environment to create. Default is "CartPole-v1".
    render_mode : str or None
        Set to "human" if you want a visual window showing the agent
        acting in real time. Leave as None for fast, invisible training.

    Returns
    -------
    env : gymnasium.Env
        A ready-to-use environment object with .reset() and .step()
        methods, which is all Stable-Baselines3 needs to train an agent.
    """
    env = gym.make(env_name, render_mode=render_mode)
    return env
