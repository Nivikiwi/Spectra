"""
Runs steps 1-7 in order. Example (PowerShell, from this folder):

python run_all_training.py --scilab-dir "C:\\Users\\bansa\\Documents\\7finalSSS\\SmartScan_fast"

Optional: --train-configs 0-6  --test-configs 7-11  --ppo-timesteps 60000
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(cmd):
    print("\n" + "=" * 70 + "\n> " + " ".join(cmd) + "\n" + "=" * 70, flush=True)
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(cmd)} (exit code {r.returncode}). Fix the error above, then re-run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scilab-dir", required=True, help="folder that contains ml_observation_live.csv")
    ap.add_argument("--train-configs", default="0-6")
    ap.add_argument("--test-configs", default="7-11",
                    help="unseen configurations for testing (9-11 need ml_observation_sweep.csv)")
    ap.add_argument("--ppo-timesteps", default="60000")
    ap.add_argument("--skip-prepare", action="store_true", help="skip step 1 if prepared/ already exists")
    ap.add_argument("--scilab-exe", default=None,
                    help="optional: WScilex-cli.exe for step 7's live adaptive-dwell run (auto-detected)")
    a = ap.parse_args()
    sdir = Path(a.scilab_dir)
    obs = sdir / "ml_observation_live.csv"
    sweep = sdir / "ml_observation_sweep.csv"
    ds = sdir / "data" / "whole_dataset_1.24m.csv"
    for f in (obs, ds):
        if not f.exists():
            raise SystemExit(f"Not found: {f}")
    t0 = time.time()
    if not a.skip_prepare:
        files = [str(obs)] + ([str(sweep)] if sweep.exists() else [])
        if not sweep.exists():
            print("NOTE: ml_observation_sweep.csv not found - configurations 9-11 are skipped "
                  "(run sweep_fast.sce in Scilab to add them).")
        run(["step1_prepare_data.py", "--observations", *files, "--dataset", str(ds)])
    import json
    ready = {int(x["config_id"]) for x in json.loads((Path(__file__).resolve().parent / "prepared" / "summary.json").read_text())}
    from smartscan_config import parse_config_list
    test = [c for c in parse_config_list(a.test_configs) if c in ready]
    missing = [c for c in parse_config_list(a.test_configs) if c not in ready]
    if missing:
        print(f"NOTE: test configurations {missing} are not prepared and are skipped.")
    test_s = ",".join(map(str, test))
    run(["step2_train_lightgbm.py", "--train-configs", a.train_configs, "--test-configs", test_s])
    run(["step3_build_dbn.py", "--train-configs", a.train_configs])
    run(["step4_train_ppo.py", "--timesteps", a.ppo_timesteps])
    run(["step5_evaluate_audit.py", "--test-configs", test_s])
    run(["step6_train_lpi_lpd.py", "--dataset", str(ds), "--demo-configs", test_s])
    step7 = ["step7_compare_algorithms.py", "--test-configs", test_s, "--scilab-dir", str(sdir)]
    if a.scilab_exe:
        step7 += ["--scilab-exe", a.scilab_exe]
    run(step7)
    print(f"\nALL STEPS DONE in {time.time() - t0:.0f} s. Open reports\\algorithm_comparison.xlsx")


if __name__ == "__main__":
    main()
