# Spectra changelog

## Workstream C - decision labels (kept)
Raw code stays in the record and tooltip; the dashboard shows the label.

| raw code | label |
|---|---|
| PPO_TOP_INDEX | Track: highest-belief band |
| PPO_ALT_n | Discover: alternative band |
| WHITTLE_TIMEOUT | Failover: PPO too slow |
| WHITTLE_ERROR | Failover: PPO error |
| WHITTLE_FORCED | Failover: forced by operator |
| WHITTLE_UNAVAILABLE | Failover: PPO unavailable |
| POLICY_WHITTLE | Baseline: Whittle staleness |
| POLICY_ROUND_ROBIN | Baseline: sequential sweep |
| T1 / T2 | Tier 1 (PPO) / Tier 2 (Whittle) |

Bug fixed: tier-2 reason codes were truncated to "WHITTLE", hiding which trigger fired.
Each trigger now has a distinct reason code in the failover log. A "Why" row was added
to the Live Decision panel.

## Workstream B - force PPO failure (kept)
Header button beside COMPARE ALL; red when active ("PPO FORCED OFF - CLICK TO RESTORE").
Makes Tier 1 unavailable from the next scan, so Tier 2 (Whittle) decides. Works before,
during and after a run. Affects ONLY Spectra - the standalone Whittle and Round Robin
strategies are untouched (R2). API: POST /api/v1/simulation/control
{"command":"FORCE_PPO_FAILURE","enabled":true,"source":"..."}; state is in
GET /api/v1/simulation/state under "ppoForced". Each forced scan logs reason FORCED_OPERATOR.

## Workstream D - frequency mode classifier (kept)
New freq_mode.py. Uses intercepted carriers and arrival times ONLY - no emitter truth (R3).
Modes: FIXED, FIXED_MULTI, FREQUENCY_HOPPING, SWEEP_CHIRP, AGILE_RANDOM, UNKNOWN (< 6 pulses).
Confidence grows with the number of pulses seen.

Evaluated in the metrics layer only (D4), 266 transmitters, accuracy 97.4 %:

| truth \ predicted | AGILE_RANDOM | FIXED | FIXED_MULTI | FREQUENCY_HOPPING |
|---|---|---|---|---|
| AGILE_RANDOM | 11 | 0 | 0 | 2 |
| FIXED | 4 | 140 | 0 | 0 |
| FIXED_MULTI | 0 | 1 | 32 | 0 |
| FREQUENCY_HOPPING | 0 | 0 | 0 | 76 |

Shown in the Live Decision panel, the Decision Log and the transmitter table.

## Workstream A - TESTED AND REJECTED (reverted)
Both changes were reverted because they did not generalise. Numbers are transmitters-found
coverage on the UNSEEN configurations 7-11, measured once each.

| variant | validation config 6 | unseen 7-11 coverage | verdict |
|---|---|---|---|
| baseline (kept) | 15 found | **36.59 %** | kept |
| A1 discovery reward 5.0 / 0.05 | 16 found (better) | 35.37 % | rejected |
| A1 reverted weights + A3 info gain | 15 found | 34.15 % | rejected |

A1: weighting a new transmitter at 5.0 and a repeat hit at 0.05 scored better on the
validation configuration but worse on unseen data - overfitting to one configuration.
A3: DBN belief entropy as an extra PPO input (state 73 -> 81) also lost coverage.
Reward weights are now configurable (DISCOVERY_REWARD, REPEAT_HIT_REWARD) and logged
in ppo_report.json, so the experiment can be repeated.

Correction to the spec: there was no random or alt-k exploration to delete (A9 ii).
PPO_ALT_k is PPO's own choice among candidates, not random exploration.

## Not done
A2 (DISCOVER/TRACK modes), A4-A8, A9(i) ablation table, E (PostgreSQL), F1 benchmark
(dropped at your request). E was skipped deliberately: the xlsx audit and per-scan logs
already cover the reporting need, and adding a database server is deployment burden
without demo value.

## Average intercept time error (problem statement 26055 figure of merit)
RevisitPredictor now records the return time it PREDICTS for each band and compares it with the
actual return, reporting the mean absolute error. Measured: config 3 -> 803 ms over 16 verified
predictions, config 7 -> 622 ms over 18. Available as
    core.revisit.intercept_time_error_s()   and   core.revisit.predictions_made()
STILL TO DO: surface this and the other six figures of merit (probability of detection, false
alarm, sensitivity, average intercept rate, average reward/cost, percentage of correct
predictions) on the dashboard and in the audit export. The current dashboard does not display
them; an attempt to add them broke the live Scilab path and was reverted.

## Simulator provenance
The Scilab RF environment and the 3-station IRSA receiver model were built by the team, not
supplied. The ML is trained only on the observation table that receiver produces (hits, misses,
SNR, PDW counts); simulator truth is read by the metrics/audit layer only, and the receiver runs
as a separate process so the scheduler cannot reach the simulator's internal state.
