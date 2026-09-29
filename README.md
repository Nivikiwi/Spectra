# Spectra — Smart Scan Strategy for Electronic Warfare

Smart India Hackathon 2026, problem statement **SIH26055**.

An Electronic Support receiver scheduler that decides **which frequency band to scan, when, and
for how long**, without prior intelligence about the emitters. It learns only from the receiver's
own hits and misses.

## The problem
A receiver's instantaneous bandwidth is far narrower than the spectrum it must watch, so it can
only listen to one 50 MHz band at a time. Fixed sweeps waste time on empty spectrum and miss
short or agile emitters.

## How it works
Every 10 ms dwell:

1. **LightGBM** scores all 322 bands from 15 features of the scan history → P(hit) per band
2. **Dynamic Bayesian Network** — one two-state hidden Markov chain per band — tracks whether a
   transmitter is active, and updates by Bayes' rule after every scan
3. **Fusion** combines the two into **P(target)**
4. **Whittle index** (classical, exact value iteration) gives the value of *information* per band
5. **Rotation model** estimates each band's detection period and boosts bands whose beam is due
6. Band score: `Whittle index + 0.5 × P(target) + 0.05 × revisit boost`
7. **PPO** picks one of the top 8 candidates
8. **Adaptive dwell** (5–15 ms) set from the model's confidence
9. The **Scilab IRSA receiver** scans that band; the result updates every component

A tiered failover (PPO → Whittle) keeps the scheduler running if the model stalls.

## Results — unseen configurations (never used for training or tuning)

| Algorithm | Transmitters found (of 82) |
|---|---|
| Pure PPO (RL alone, 322 actions) | 2 |
| ε-greedy | 6 |
| UCB1 | 8 |
| Thompson sampling | 14 |
| Random | 15 |
| LightGBM only | 25 |
| **Whittle index** | **29** |
| **Spectra** | **28–31** |

On the 7 training configurations (203 transmitters, a larger sample), Spectra finds **71 vs the
Whittle index's 63 (+12.7%)**, winning 6 of 7 configurations.

**Honest summary:** Spectra clearly beats every standard learning algorithm and each of its own
components alone. Against the Whittle index it is ahead on the larger sample and statistically
indistinguishable on the smaller held-out set. See `CHANGELOG.md` for the full ablation,
including five changes that were tested and rejected.

## No ground-truth leakage
The scheduler, LightGBM, DBN, PPO and the classifiers use **only** receiver observations — hits,
misses, SNR, pulse counts and timestamps. Simulator truth is read by the metrics and audit layer
only. The Scilab receiver runs as a separate process, so the scheduler cannot reach its state.

## Also included
- **LPI / LPD classification** of intercepted emitters
- **Frequency-mode classifier** (FIXED / FIXED_MULTI / HOPPING / SWEEP / AGILE) — 97.4% accurate
- **Three-station geolocation** from the Scilab receivers
- **Live dashboard** with band heatmap, decision log, model health and strategy comparison
- **Excel audit export** of every scan decision

## Footprint
About 1 MB of models and under a million simple operations per decision: roughly 4 ms in Python
on one CPU core, no GPU. Suitable for an embedded receiver controller after a C port.

## Running it
Large data files are excluded from this repository (see `.gitignore`). To reproduce:

1. Put the pulse dataset at `SmartScan_fast/data/whole_dataset_1.24m.csv`
2. In Scilab, run `sweep_fast.sce` to record the baseline receiver sweep
3. `python run_all_training.py --scilab-dir "...\SmartScan_fast"`
4. `python dashboard_server.py` then open http://localhost:8000

Full instructions are in `SmartScan_ML_v2/README_STEPS.txt`.

## Known limitations
- The RF environment and receiver model were built by us in Scilab; results are simulation-only.
- The pulse dataset was truncated by a spreadsheet at 1,048,575 rows, so configuration 11 is
  shorter than the rest.
- Not tested against reactive or adversarial emitters.
- PPO's contribution is within measurement noise; the deterministic scorer does the heavy lifting.
