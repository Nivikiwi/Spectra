
# Smart Scan ML closed-loop stage

## Final runtime flow

Scilab RF environment + IRSA receiver
  -> `ml_observation_live.csv`
  -> branch:
       1. Custom: LightGBM -> DBN -> PPO
       2. Whittle baseline
       3. Round Robin baseline
  -> `ml_scan_command.txt`
  -> Scilab receiver
  -> next observation
  -> audit history / Excel

## Important dataset finding

The supplied original ML workflow reads ALL `config_*.h5` files from
`tsrd_data/scan/val_scan` in the data-generation scripts. It does not restrict
itself to config 0.

However, the current Scilab `main.sce` in SmartScan_final_fixed explicitly has
`CONFIG_ID_TO_RUN = 0`, so the live receiver currently runs only config 0.
That must be handled separately when generating receiver-derived training data.

Also, the supplied ZIP does not contain a LightGBM training script/model,
PPO, Whittle, Round Robin, or an online controller. It contains target/DBN
preparation and evaluation scripts. This package adds the missing online/RL
stage.

## Receiver action space

The original ML workflow used 500-MHz states from 250 to 17750 MHz.
That does not match the current IRSA receiver.

This package uses the actual receiver action:
- RF: 100-12000 MHz
- instantaneous BW: 50 MHz
- 238 candidate windows
- centers: 125, 175, ..., 11975 MHz

This keeps the ML action directly executable by Scilab.

## Training-data rule

`ml_observation_live.csv` is the online observation interface.
For supervised training, each observation also needs a separate ground-truth
activity vector for all candidate windows. Ground truth is training/evaluation
only and is never supplied to the online scheduler.

Recommended files:
- `receiver_training_observations.csv`: receiver observations, plus `config_id` and `step`
- `receiver_training_ground_truth.csv`: `config_id`, `step`, and `active_0..active_237`

Run:
1. `prepare_lightgbm_training.py`
2. `train_receiver_lightgbm.py`
3. `build_receiver_dbn.py`
4. `build_ppo_states.py`
5. `train_ppo_scheduler.py`

Whittle and Round Robin require no training.

## Configuration split

LightGBM uses a configuration-level train/test split. It never randomly
splits individual rows from the same configuration across train and test.

DBN artifacts can be built from all training configurations. For a final
deployment model, rebuild from all training configurations after validation.
Do not use held-out test configurations to train the final reported model
before the evaluation is complete.

## PPO

PPO is trained against the same 238-band action space. The state contains
the LightGBM/DBN belief and scan-history information. The physical receiver
is not bypassed; at deployment PPO writes the selected center and dwell to
`ml_scan_command.txt`.

## Audit

`audit_report.py` produces `smart_scan_audit.xlsx`.


## Practical training-data size
The preparation stage deliberately does NOT create 238 candidate rows for every observation.
That would explode to hundreds of millions of rows for the full 12-configuration replay.
Instead, each baseline scan supplies one training row for the band actually scanned; the 238
per-band models then learn from their own historical scan examples. At runtime the controller
constructs the same candidate features for all 238 bands and scores them without using truth.
