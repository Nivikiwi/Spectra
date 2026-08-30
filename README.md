# Day 1 – AI/ML Setup (Smart Scan, SIH 26055)

## What's in this folder

| File | What it does | Will it change later? |
|---|---|---|
| `requirements.txt` | List of libraries to install | No |
| `verify_install.py` | Quick check that everything installed correctly | No |
| `env_utils.py` | Creates the environment (currently CartPole-v1) | **Yes** — this is where Chahek's spectrum environment will plug in |
| `train.py` | Trains a PPO agent and saves it to `models/` | No |
| `evaluate.py` | Loads the saved model and tests it | No |

## Why it's split into files like this

Your task said to keep environment, training, and evaluation separate so that later,
swapping CartPole for Chahek's spectrum-search environment only means editing
`env_utils.py`. `train.py` and `evaluate.py` never touch the environment directly —
they just call `make_env()` — so they don't need to change at all.

## Exact steps to run today

1. Put all 5 files in the same folder on your computer (or upload them to a
   Colab notebook / Colab files panel).
2. Install the libraries:
   ```
   pip install stable-baselines3[extra] gymnasium torch
   ```
   (In Colab, put a `!` in front: `!pip install stable-baselines3[extra] gymnasium torch`)
3. Run the smoke test first:
   ```
   python verify_install.py
   ```
   You should see version numbers print, some CartPole steps, and finally
   `INSTALL OK`. If you see red error text instead, stop here and share the
   error message — don't move to the next step yet.
4. Train the agent:
   ```
   python train.py
   ```
   This will print a lot of numbers scrolling by (that's normal — it's SB3's
   training log, showing things like average reward improving over time).
   It ends with a line like:
   `Training complete. Model saved to: models/ppo_cartpole.zip`
5. Evaluate the agent:
   ```
   python evaluate.py
   ```
   You should see something like:
   `Mean reward over 10 episodes: 480.30 +/- 25.10`
   (Higher is better; CartPole's max per episode is 500.)

## What "checkpoint save/load" actually means here

- `model.save("models/ppo_cartpole")` in `train.py` writes the agent's learned
  neural network weights to a file called `ppo_cartpole.zip` inside the `models/`
  folder.
- `PPO.load("models/ppo_cartpole", env=env)` in `evaluate.py` reads that same
  file back into a fresh PPO object — this is exactly like saving your progress
  in a game and reloading it later. You've now proven the whole loop:
  **environment → train → evaluate → save → reload → evaluate again.**

## If something goes wrong

- **"ModuleNotFoundError"** → the pip install step didn't finish or you're in
  the wrong virtual environment. Re-run the pip install command.
- **Training runs but reward stays low (~20-50)** → try increasing
  `TOTAL_TIMESTEPS` in `train.py` (e.g., to 100_000).
- **"FileNotFoundError" in evaluate.py** → you haven't run `train.py` yet, or
  you're running the two scripts from different folders.

## What happens next (Day 2 preview)

Once Chahek hands over her math formulation (state space, action space,
reward function) for the spectrum-search problem, you'll create a new
Gymnasium-style environment class implementing `reset()` and `step()` with
her formulas, and swap it in inside `env_utils.py`. Everything else in this
project stays the same.
