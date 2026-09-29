"""
STEP 7 - Compare Spectra with every standard scanning / learning algorithm.

Same receiver, same unseen configurations (default 7-11), same scoring (scan_metrics.py,
simulation-time based, so different dwell lengths are compared fairly).

  Round Robin, Random                      classic scanning
  Epsilon-greedy, UCB1, Thompson sampling  online learning (multi-armed bandits)
  Whittle index                            restless-bandit index policy
  LightGBM only, DBN only                  parts of the custom model on their own
  Spectra, fixed 10 ms                     LightGBM -> DBN -> Whittle-informed index (+ revisit) -> PPO
  SPECTRA, ADAPTIVE DWELL (live Scilab)    the full Spectra model: the real Scilab receiver
                                           applies its 5-15 ms dwell choices

All 10-ms strategies use the recorded receiver sweep (verified identical to live Scilab).
The adaptive-dwell custom model needs Scilab (auto-detected, or --scilab-exe); without it,
that row is skipped.

Output: reports/algorithm_comparison.xlsx
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from smartscan_config import PREPARED_DIR, MODELS_DIR, REPORTS_DIR, N_BANDS, FIXED_DWELL_S, parse_config_list, band_of_centre
from smartscan_core import Controller, load_grid
from scan_metrics import ConfigTruth, score_config, combine

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- baseline learners (10 ms)
class RandomScan:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def decide(self, t):
        return int(self.rng.integers(N_BANDS))

    def observe(self, t, band, hit):
        pass


class EpsilonGreedy:
    def __init__(self, eps=0.1, seed=0):
        self.eps, self.rng = eps, np.random.default_rng(seed)
        self.n, self.s = np.zeros(N_BANDS), np.zeros(N_BANDS)

    def decide(self, t):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(N_BANDS))
        q = np.divide(self.s, self.n, out=np.zeros(N_BANDS), where=self.n > 0)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def observe(self, t, band, hit):
        self.n[band] += 1
        self.s[band] += hit


class UCB1:
    def __init__(self):
        self.n, self.s = np.zeros(N_BANDS), np.zeros(N_BANDS)

    def decide(self, t):
        untried = np.flatnonzero(self.n == 0)
        if len(untried):
            return int(untried[0])
        total = self.n.sum()
        return int(np.argmax(self.s / self.n + np.sqrt(2.0 * np.log(total) / self.n)))

    def observe(self, t, band, hit):
        self.n[band] += 1
        self.s[band] += hit


class Thompson:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.a, self.b = np.ones(N_BANDS), np.ones(N_BANDS)

    def decide(self, t):
        return int(np.argmax(self.rng.beta(self.a, self.b)))

    def observe(self, t, band, hit):
        self.a[band] += hit
        self.b[band] += 1 - hit


def _call(fn, *args, now=None):
    """Call decide/observe with the simulation time if the strategy accepts it."""
    try:
        return fn(*args, now_s=now)
    except TypeError:
        return fn(*args)


class ModelPart:
    """LightGBM-only or DBN-only greedy, using the trained custom-model components."""

    def __init__(self, which):
        self.ctrl = Controller("greedy", MODELS_DIR)
        self.which = which

    def decide(self, t, now_s=None):
        core = self.ctrl.core
        core.score(t, now_s)
        p = core.last["p_lgbm"] if self.which == "lgbm" else core.last["p_dbn"]
        return int(np.argmax(p))

    def observe(self, t, band, hit, snr=0.0, pdws=0.0, now_s=None):
        self.ctrl.observe(t, band, hit, snr, pdws, now_s)


class Wrapped:
    """Controller (custom / whittle / round_robin) with the same interface."""

    def __init__(self, policy):
        self.ctrl = Controller(policy, MODELS_DIR)

    def decide(self, t, now_s=None):
        return self.ctrl.decide(t, now_s)[0]

    def observe(self, t, band, hit, snr=0.0, pdws=0.0, now_s=None):
        self.ctrl.observe(t, band, hit, snr, pdws, now_s)


def run_recorded(make, grid):
    """10-ms strategy on the recorded receiver sweep. Returns scans (start, dwell, band, hit, pdws)."""
    agent = make()
    hit, snr, pdws, times = grid["hit"], grid["snr"], grid["pdws"], grid["dwell_time_s"]
    scans = []
    for t in range(hit.shape[0]):
        now = float(times[t])
        b = _call(agent.decide, t, now=now)
        h = int(hit[t, b])
        try:
            agent.observe(t, b, h, float(snr[t, b]), float(pdws[t, b]), now_s=now)
        except TypeError:
            agent.observe(t, b, h)
        scans.append((float(times[t]), FIXED_DWELL_S, b, h, float(pdws[t, b])))
    return scans


# ---------------------------------------------------------------- custom model, adaptive dwell, live Scilab
def find_scilab(explicit):
    if explicit:
        return explicit
    import glob
    for name in ("WScilex-cli.exe", "Scilex.exe", "WScilex.exe"):
        for root in (r"C:\Program Files\scilab-*\bin", r"C:\Program Files (x86)\scilab-*\bin"):
            f = sorted(glob.glob(os.path.join(root, name)))
            if f:
                return f[-1]
    for name in ("scilab-cli", "scilab"):
        from shutil import which
        if which(name):
            return which(name)
    return None


def run_live_custom(cfg, scilab_dir, scilab_exe, tag="eval"):
    from online_controller import CsvFollower, read_status, write_command, to_float
    from decision_tiers import TieredDecider
    sdir = Path(scilab_dir)
    sfx = "_" + tag
    status, cmd = sdir / f"ml_closed_loop_status{sfx}.txt", sdir / f"ml_scan_command{sfx}.txt"
    for p in (status, cmd):
        if p.exists():
            p.unlink()
    (sdir / f"ml_closed_loop_request{sfx}.txt").write_text(f"{cfg} 0\n")
    script = str(sdir / "main_closed_loop.sce").replace("\\", "/")
    kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)} if os.name == "nt" else {}
    log = open(sdir / f"scilab_eval{sfx}.log", "w")
    proc = subprocess.Popen([scilab_exe, "-nb", "-e", f"exec('{script}',-1); exit();"], cwd=str(sdir),
                            env=dict(os.environ, SPECTRA_TAG=tag), stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, **kw)
    ctrl = TieredDecider(MODELS_DIR, force_tier=1, adaptive_dwell=True)
    t0 = time.time()
    while True:
        st = read_status(status)
        if st and st[0] == "READY":
            break
        if proc.poll() is not None or time.time() - t0 > 300:
            raise RuntimeError(f"Scilab did not start for configuration {cfg} (see {log.name})")
        time.sleep(0.2)
    fol = CsvFollower(sdir / f"ml_observation_closed_loop{sfx}.csv")
    band, _ = ctrl.decide(0, None)
    pending = {1: band}
    write_command(cmd, 1, band, ctrl.last_info["dwell_s"])
    last_end = None
    scans, done = [], False
    while True:
        rows = fol.new_rows()
        if not rows:
            if done:
                break
            st = read_status(status)
            if st and st[0] == "DONE":
                done = True
                continue
            if proc.poll() is not None and not (st and st[0] == "DONE"):
                raise RuntimeError(f"Scilab stopped during configuration {cfg} (see {log.name})")
            time.sleep(0.002)
            continue
        for r in rows:
            step = int(to_float(r["step"]))
            if step not in pending:
                continue
            pending.pop(step)
            b = int(band_of_centre(to_float(r["scan_center_mhz"])))
            p = to_float(r["total_pdws"])
            hit = int(p > 0 or to_float(r["hit"]) > 0)
            ts, ds = to_float(r["scan_time_s"]), to_float(r["dwell_s"])
            ctrl.observe(step - 1, b, hit, to_float(r["mean_snr_db"]), p, ts)
            scans.append((ts, ds, b, hit, p))
            last_end = ts + ds
            band, _ = ctrl.decide(step, last_end)
            pending[step + 1] = band
            write_command(cmd, step + 1, band, ctrl.last_info["dwell_s"])
    proc.wait(timeout=60)
    log.close()
    return scans, ctrl.tier_counts


# ---------------------------------------------------------------- main
ALGORITHMS = [
    ("Round Robin", "classic", lambda: Wrapped("round_robin")),
    ("Random", "classic", lambda: RandomScan()),
    ("Epsilon-greedy (e=0.1)", "online learning", lambda: EpsilonGreedy()),
    ("UCB1", "online learning", lambda: UCB1()),
    ("Thompson sampling", "online learning", lambda: Thompson()),
    ("Whittle index", "restless bandit", lambda: Wrapped("whittle")),
    ("LightGBM only", "ML (part of Spectra)", lambda: ModelPart("lgbm")),
    ("DBN only", "ML (part of Spectra)", lambda: ModelPart("dbn")),
    ("Spectra, fixed 10 ms", "Spectra (no adaptive dwell)", lambda: Wrapped("custom")),
]

COLUMNS = [  # (key, label, better)
    ("transmitters_found", "Transmitters found", "higher"),
    ("time_to_first_detection_s", "Time to first detection (s, lower = better)", "lower"),
    ("detections_per_s", "Detections per second", "higher"),
    ("pd_pct", "Pd on active bands (%)", "higher"),
    ("pfa_pct", "False alarms Pfa (%, lower = better)", "lower"),
    ("pdws_intercepted", "Pulses intercepted", "higher"),
    ("lpd_found", "LPD transmitters found", "higher"),
    ("lpi_found", "LPI transmitters found", "higher"),
]


def main():
    ap = argparse.ArgumentParser(description="Step 7: compare the custom model with all baseline algorithms")
    rep = json.loads((MODELS_DIR / "lightgbm_report.json").read_text())
    ap.add_argument("--test-configs", default=",".join(map(str, rep.get("test_configs", [7, 8, 9, 10, 11]))))
    ap.add_argument("--scilab-dir", default=str(HERE.parent / "SmartScan_fast"))
    ap.add_argument("--scilab-exe", default=None)
    ap.add_argument("--output", default=str(REPORTS_DIR / "algorithm_comparison.xlsx"))
    args = ap.parse_args()
    cfgs = [c for c in parse_config_list(args.test_configs) if (PREPARED_DIR / f"config_{c}.npz").exists()]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    truths = {c: ConfigTruth(load_grid(PREPARED_DIR / f"config_{c}.npz")) for c in cfgs}
    t0 = time.time()
    results, per_cfg = [], []
    for name, family, make in ALGORITHMS:
        parts = []
        for c in cfgs:
            part = score_config(truths[c], run_recorded(make, load_grid(PREPARED_DIR / f"config_{c}.npz")))
            parts.append(part)
            per_cfg.append({"algorithm": name, "config": c, "transmitters_found": part["found"], "detections": part["hits"]})
        results.append({"algorithm": name, "family": family, "dwell": "10 ms fixed", "receiver": "recorded sweep",
                        **combine(parts)})
        print(f"  {name:28s} done ({time.time() - t0:.0f} s)", flush=True)

    exe = find_scilab(args.scilab_exe)
    sdir = Path(args.scilab_dir)
    if exe and (sdir / "main_closed_loop.sce").exists():
        parts, tiers = [], {1: 0, 2: 0}
        for c in cfgs:
            scans, tc = run_live_custom(c, sdir, exe)
            part = score_config(truths[c], scans)
            parts.append(part)
            per_cfg.append({"algorithm": "SPECTRA (full, adaptive dwell)", "config": c, "transmitters_found": part["found"],
                            "detections": part["hits"]})
            for k in tiers:
                tiers[k] += tc.get(k, 0)
            print(f"  SPECTRA adaptive dwell, config {c} done ({time.time() - t0:.0f} s)", flush=True)
        results.append({"algorithm": "SPECTRA (full, adaptive dwell)", "family": "Spectra (full)",
                        "dwell": "5-15 ms adaptive", "receiver": "LIVE Scilab", **combine(parts)})
        if tiers[2]:
            print(f"  note: {tiers[2]} Spectra decisions used the Whittle failover")
    else:
        print("  Scilab not found - the adaptive-dwell Spectra row is skipped (use --scilab-exe).")

    df = pd.DataFrame(results)
    table = pd.DataFrame({"Algorithm": df["algorithm"], "Type": df["family"], "Dwell": df["dwell"],
                          "Receiver": df["receiver"]})
    for key, label, _ in COLUMNS:
        table[label] = df[key].round(2)
    table["Transmitters found"] = [f"{a} / {b}" for a, b in zip(df["transmitters_found"], df["transmitters_detectable"])]
    table["LPD transmitters found"] = [f"{a} / {b}" for a, b in zip(df["lpd_found"], df["lpd_detectable"])]
    table["LPI transmitters found"] = [f"{a} / {b}" for a, b in zip(df["lpi_found"], df["lpi_detectable"])]
    best = []
    for key, label, better in COLUMNS:
        vals = df[key].to_numpy(float)
        tgt = vals.max() if better == "higher" else vals.min()
        winners = df["algorithm"][vals == tgt].tolist()
        best.append({"Metric": label, "Best": " / ".join(winners), "Value": round(float(tgt), 2)})
    with pd.ExcelWriter(args.output, engine="openpyxl") as w:
        table.to_excel(w, sheet_name="Comparison", index=False)
        pd.DataFrame(best).to_excel(w, sheet_name="Best_per_metric", index=False)
        pd.DataFrame(per_cfg).to_excel(w, sheet_name="Per_configuration", index=False)
        df.to_excel(w, sheet_name="Raw", index=False)
    pd.set_option("display.width", 250)
    print("\n" + table.to_string(index=False))
    print("\n" + pd.DataFrame(best).to_string(index=False))
    print(f"\nSTEP 7 DONE ({time.time() - t0:.0f} s). Saved {args.output}")


if __name__ == "__main__":
    main()
