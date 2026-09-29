"""
SPECTRA - DASHBOARD SERVER

    python dashboard_server.py          then open  http://localhost:8000

START runs the selected configurations (default ALL, 0 -> 11) with ONE policy (single mode,
the default) - only that policy's decision maker and ONE live Scilab receiver exist:
    Spectra      LightGBM -> DBN -> Whittle-informed index -> PPO, adaptive dwell,
                 failover to Whittle INSIDE the same run (tiered decider)
    Whittle      Whittle index, fixed 10 ms
    Round Robin  fixed 10 ms
COMPARE mode (explicit toggle) runs all three side by side, each on its own Scilab receiver.
If Scilab is not available, the policies run on the recorded receiver sweep (identical results
at 10 ms; Spectra then uses a fixed 10 ms dwell).

API:  GET /api/v1/simulation/state   POST /api/v1/simulation/control   GET /api/v1/audit.xlsx
"""
import argparse
import collections
import glob
import io
import json
import math
import mimetypes
import os
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from smartscan_config import (PREPARED_DIR, MODELS_DIR, REPORTS_DIR, BAND_CENTRE_MHZ, N_BANDS, FIXED_DWELL_S,
                              band_of_centre)
from smartscan_core import Controller, load_grid
from online_controller import CsvFollower, read_status, write_command
from decision_tiers import TieredDecider, binary_entropy_bits, decision_label, TIER_NAMES
from scan_metrics import ConfigTruth
from lpi_lpd import pulse_features, MIN_PULSES, LPI_POWER_W, LPD_DETECT_FRACTION
from freq_mode import FreqModeTracker

KEYS = ("ai", "whittle", "base")
LABEL = {"ai": "Spectra", "whittle": "Whittle Index", "base": "Round Robin"}
STRATEGY_TO_KEY = {"LightGBM + DBN": "ai", "Custom": "ai", "Spectra": "ai", "Whittle Index": "whittle",
                   "Round Robin": "base"}
KEY_TO_STRATEGY = {"ai": "LightGBM + DBN", "whittle": "Whittle Index", "base": "Round Robin"}
POLICY = {"whittle": "whittle", "base": "round_robin"}
LOG_COLUMNS = ["config_id", "dwell", "time_s", "dwell_s", "centre_mhz", "decision", "p_target", "confidence",
               "receiver_hit", "pdws"]                      # same 10 columns as online_controller.py
HEAT_N = 150                                                # scans shown in the band heatmap
MAX_POINTS = 1500
READY_TIMEOUT_S = 300


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def thin_idx(n):
    if n <= MAX_POINTS:
        return np.arange(n)
    return np.unique(np.r_[np.linspace(0, n - 1, MAX_POINTS).astype(int), n - 1])


def find_scilab_exe(explicit):
    """Console versions first (no window): WScilex-cli.exe, Scilex.exe, then WScilex.exe."""
    if explicit:
        return explicit
    roots = [r"C:\Program Files\scilab-*\bin", r"C:\Program Files (x86)\scilab-*\bin",
             os.path.expanduser(r"~\AppData\Local\Programs\scilab-*\bin")]
    for name in ("WScilex-cli.exe", "Scilex.exe", "WScilex.exe"):
        found = sorted(f for r in roots for f in glob.glob(os.path.join(r, name)))
        if found:
            return found[-1]
    return None


class Segment:
    def __init__(self, cfg, test_configs):
        self.cfg = int(cfg)
        self.split = "unseen" if self.cfg in test_configs else "training"
        self.g = load_grid(PREPARED_DIR / f"config_{cfg}.npz")
        self.D = self.g["hit"].shape[0]
        self.truth = ConfigTruth(self.g)
        self.end_s = self.truth.end_s


def new_counter():
    return dict(scans=0, hits=0, tp=0, fp=0, tn=0, fn=0, retunes=0, pdws=0.0, listen=0.0, found={},
                progress_s=0.0, last_band=None)


class Engine:
    def __init__(self, models_dir, scilab_dir, scilab_exe):
        self.models_dir = Path(models_dir)
        self.scilab_dir = Path(scilab_dir) if scilab_dir else None
        self.scilab_exe = find_scilab_exe(scilab_exe)
        self.mode = ("scilab" if self.scilab_dir and (self.scilab_dir / "main_closed_loop.sce").exists()
                     and self.scilab_exe else "replay")
        self.lock = threading.RLock()
        self.available = sorted(int(p.stem.split("_")[1]) for p in PREPARED_DIR.glob("config_*.npz"))
        if not self.available:
            raise SystemExit("No prepared/config_*.npz - run the training first (run_all_training.py).")
        rep = json.loads((self.models_dir / "lightgbm_report.json").read_text())
        self.test_configs = [int(c) for c in rep.get("test_configs", [])]
        self.segments = {}
        self.ctrl = {}                     # created at START, ONLY for the policies that run
        self.run_mode = "single"           # "single" = one policy; "compare" = all three side by side
        self.run_keys = ("ai",)
        self.lpi_models, self.lpi_cv = None, None
        try:
            import lightgbm as lgb
            if (self.models_dir / "lpi_model.txt").exists() and (self.models_dir / "lpd_model.txt").exists():
                self.lpi_models = {k: lgb.Booster(model_file=str(self.models_dir / f"{k}_model.txt"))
                                   for k in ("lpi", "lpd")}
                r6 = json.loads((self.models_dir / "lpi_lpd_report.json").read_text())
                self.lpi_cv = {k: r6.get(f"{k}_cv_all_pulses") for k in ("lpi", "lpd")}
        except Exception as e:
            print(f"LPI/LPD models not loaded: {e}")
        self.algorithms = self._load_algorithm_comparison()
        self.selection = "all"
        self.active = "ai"
        self.running = False
        self.run_id = 0
        self.workers = []
        self.phase = "idle"
        self.message = ""
        self.last_audit_file = None
        # B: operator "force PPO failure" switch (Spectra only; baselines unaffected)
        self.ppo_forced_off = False
        self.ppo_forced_at = None
        self.ppo_forced_by = None
        self._reset_run()

    def seg(self, cfg):
        if cfg not in self.segments:
            self.segments[cfg] = Segment(cfg, self.test_configs)
        return self.segments[cfg]

    def _make_ctrl(self, k):
        if k == "ai":
            # adaptive dwell needs the live receiver; otherwise the recorded 10-ms sweep is used
            d = TieredDecider(self.models_dir, adaptive_dwell=(None if self.mode == "scilab" else False))
            d.forced_off = self.ppo_forced_off
            return d
        return Controller(POLICY[k], self.models_dir)

    def _planned_keys(self):
        return KEYS if self.run_mode == "compare" else (self.active,)

    def sequence(self):
        return list(self.available) if self.selection == "all" else [int(self.selection)]

    # ------------------------------------------------------------ run state
    def _reset_run(self):
        seq = self.sequence()
        self.cur_cfg = seq[0]
        self.time_offset = {}                       # cfg -> start of that config on the run clock (s)
        off = 0.0
        for c in seq:
            self.time_offset[c] = off
            off += self.seg(c).end_s
        self.total_s = off
        self.stats = {k: {} for k in KEYS}
        self.rows = {k: [] for k in KEYS}
        self.series = {k: {"t": [], "detections": [], "rate": [], "found": []} for k in KEYS}
        self.roll = {k: collections.deque(maxlen=50) for k in KEYS}
        self.totals = {k: {"hits": 0, "found": 0} for k in KEYS}
        self.heat = {k: collections.deque(maxlen=HEAT_N) for k in KEYS}
        self.detections = {k: collections.deque(maxlen=3000) for k in KEYS}
        self.captured = {k: {} for k in KEYS}
        self.fmode = {k: FreqModeTracker() for k in KEYS}      # D: frequency mode per transmitter
        self.lpi_cache = {}
        self.geo = {k: collections.deque(maxlen=3000) for k in KEYS}
        self.positions = {"stations": [], "emitters": []}
        self.progress = {k: {"config": None, "time_s": 0.0} for k in KEYS}
        self.tier_counts = {1: 0, 2: 0}
        self.failovers = collections.deque(maxlen=200)       # real PPO -> Whittle failover events
        self.latency_ms = collections.deque(maxlen=5000)     # Spectra decision times
        self.config_starts = []
        self.events = []
        planned = self._planned_keys()
        self.log(f"Ready: configuration{'s' if len(seq) > 1 else ''} {seq[0]}"
                 f"{' -> ' + str(seq[-1]) if len(seq) > 1 else ''} ({self.total_s:,.1f} s of RF time). "
                 f"{'COMPARE mode: all three policies' if self.run_mode == 'compare' else 'Policy: ' + LABEL[planned[0]]}"
                 f" on {'live Scilab' if self.mode == 'scilab' else 'the recorded receiver sweep (Scilab not found)'}.")

    def log(self, msg):
        self.events.insert(0, f"[{time.strftime('%H:%M:%S')}] {msg}")
        del self.events[100:]

    # ------------------------------------------------------------ recording one scan
    def _belief(self, k):
        """The strategy's own band scores after its last decision (for the heatmap)."""
        c = self.ctrl.get(k)
        if c is None:
            return None
        if k == "ai" and c.core is not None and "fused" in c.core.last:
            return np.asarray(c.core.last["fused"], dtype=np.float32)
        if k == "whittle":
            return np.asarray(c.whittle.omega, dtype=np.float32)
        return None                                            # Round Robin has no model

    def _record(self, k, cfg, n, start_s, dwell_s, band, hit, snr, pdws, score, belief, info, geo_rows=()):
        seg = self.seg(cfg)
        c = self.stats[k].setdefault(cfg, new_counter())
        active, det = seg.truth.window(start_s, dwell_s, band)
        c["scans"] += 1
        c["hits"] += hit
        c["pdws"] += pdws if math.isfinite(pdws) else 0.0
        c["listen"] += dwell_s
        c["progress_s"] = start_s + dwell_s
        if active and hit:
            c["tp"] += 1
        elif active:
            c["fn"] += 1
        elif hit:
            c["fp"] += 1
        else:
            c["tn"] += 1
        c["retunes"] += int(c["last_band"] is not None and c["last_band"] != band)
        c["last_band"] = band
        new = 0
        if hit:
            for x in det:
                if x not in c["found"]:
                    c["found"][x] = start_s
                    new += 1
            if seg.truth.freq is not None:
                by_tx = {}
                for i in seg.truth.detectable_pulses(start_s, dwell_s, band):
                    tx = int(seg.truth.tx[i])
                    self.captured[k].setdefault((cfg, tx), []).append(int(i))
                    by_tx.setdefault(tx, []).append(int(i))
                # D1: classify from the intercepted carriers only (no emitter truth is read)
                for tx, idx in by_tx.items():
                    ii = np.asarray(idx)
                    self.fmode[k].add((cfg, tx), seg.truth.freq[ii], seg.truth.time[ii])
                if by_tx:
                    big = max(by_tx, key=lambda x: len(by_tx[x]))
                    m, c, _ = self.fmode[k].get((cfg, big))
                    fmode, fconf = m, c
        if info is None:
            p = score if (score is not None and math.isfinite(score)) else float("nan")
            h = binary_entropy_bits(p)
            code = f"POLICY_{POLICY[k].upper()}"
            info = {"p_target": p, "confidence": (1 - h) if math.isfinite(h) else float("nan"),
                    "decision": code, "decision_label": decision_label(code), "reason": "", "tier": 0}
        else:
            self.tier_counts[info["tier"]] = self.tier_counts.get(info["tier"], 0) + 1
            if math.isfinite(info.get("decision_ms", float("nan"))):
                self.latency_ms.append(info["decision_ms"])
            if info["tier"] != 1 and "forced" not in info["decision"]:
                reason = ("FORCED_OPERATOR" if self.ppo_forced_off
                          else str(info["decision"]).split(":", 1)[-1].split(" (")[0].strip())
                self.failovers.append({"time": time.strftime("%H:%M:%S"), "config": cfg, "scan": n,
                                       "reason_code": reason, "label": info.get("decision_label", ""),
                                       "sim_time_s": round(start_s, 4), "decision": info["decision"],
                                       "decision_ms": info.get("decision_ms")})
                self.log(f"FAILOVER config {cfg} scan {n}: {info['decision']}")
        fmode = locals().get("fmode", "")
        fconf = locals().get("fconf", float("nan"))
        run_t = self.time_offset.get(cfg, 0.0) + start_s
        self.rows[k].append({"config_id": cfg, "dwell": n, "time_s": start_s, "dwell_s": dwell_s,
                             "centre_mhz": float(BAND_CENTRE_MHZ[band]), "decision": info["decision"],
                             "p_target": info["p_target"], "confidence": info["confidence"],
                             "receiver_hit": int(hit), "pdws": pdws, "tier": info.get("tier", 0),
                             "decision_label": info.get("decision_label", ""), "why": info.get("reason", ""),
                             "freq_mode": fmode, "freq_mode_confidence": fconf,
                             "snr_db": snr if hit else None})
        self.totals[k]["hits"] += hit
        self.totals[k]["found"] += new
        self.roll[k].append(hit)
        s = self.series[k]
        s["t"].append(round(run_t, 4))
        s["detections"].append(self.totals[k]["hits"])
        s["rate"].append(round(100.0 * sum(self.roll[k]) / len(self.roll[k]), 1))
        s["found"].append(self.totals[k]["found"])
        self.heat[k].append((round(run_t, 4), None if belief is None else np.round(belief, 2), band, int(hit)))
        if hit:
            self.detections[k].append((round(run_t, 4), float(BAND_CENTRE_MHZ[band]), snr))
        for gr in geo_rows:
            self.geo[k].append(gr)
        self.progress[k] = {"config": cfg, "time_s": start_s}

    def _decide(self, k, t, now_s=None):
        """Decision + everything that must be captured AT decision time."""
        ctrl = self.ctrl[k]
        band, score = ctrl.decide(t, now_s)
        info = dict(ctrl.last_info) if k == "ai" and ctrl.last_info else None
        dwell = info["dwell_s"] if (info and self.mode == "scilab") else FIXED_DWELL_S
        return band, score, self._belief(k), info, dwell

    # ------------------------------------------------------------ run control
    def _halted(self, my):
        return self.run_id != my

    def _run(self, my):
        try:
            for cfg in self.sequence():
                if self._halted(my):
                    return
                with self.lock:
                    self.cur_cfg = cfg
                    for k in self.run_keys:
                        self.ctrl[k].reset()
                        self.stats[k][cfg] = new_counter()
                        self.geo[k].clear()
                    self.config_starts.append({"t": self.time_offset[cfg], "config": cfg})
                    seg = self.seg(cfg)
                    self.phase = "starting" if self.mode == "scilab" else "running"
                    n_rx = len(self.run_keys)
                    self.message = (f"Configuration {cfg} ({seg.split}): starting {n_rx} Scilab receiver"
                                    f"{'s' if n_rx > 1 else ''} ..." if self.mode == "scilab"
                                    else f"Configuration {cfg} ({seg.split}): replay")
                    self.log(self.message)
                errors = []
                threads = [threading.Thread(target=self._strategy_worker, args=(k, cfg, my, errors), daemon=True)
                           for k in self.run_keys]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
                if self._halted(my):
                    return
                if errors:
                    raise RuntimeError("; ".join(errors))
                with self.lock:
                    self.log(f"Configuration {cfg} done - transmitters found: " + ", ".join(
                        f"{LABEL[k]} {self.metrics(k, [cfg])['found']}" for k in self.run_keys) + ".")
            with self.lock:
                if self._halted(my):
                    return
                self.running = False
                self.phase = "finished"
                self.message = f"Run COMPLETE: configuration(s) {', '.join(map(str, self.sequence()))}."
                self.log(self.message)
                self._autosave_audit()
        except Exception as e:
            with self.lock:
                if self._halted(my):
                    return
                self.running = False
                self.phase = "error"
                self.message = f"ERROR: {e}"
                self.log(self.message)

    def _wait_if_paused(self, my):
        while not self.running and not self._halted(my):
            time.sleep(0.1)

    def _strategy_worker(self, k, cfg, my, errors):
        try:
            if self.mode == "scilab":
                self._live(k, cfg, my)
            else:
                self._replay(k, cfg, my)
        except Exception as e:
            errors.append(f"{LABEL[k]}: {type(e).__name__}: {e}")

    def _replay(self, k, cfg, my):
        g = self.seg(cfg).g
        for t in range(g["hit"].shape[0]):
            self._wait_if_paused(my)
            if self._halted(my):
                return
            with self.lock:
                now = float(g["dwell_time_s"][t])
                band, score, belief, info, _ = self._decide(k, t, now)
                hit = int(g["hit"][t, band])
                snr = float(g["snr"][t, band])
                pdws = float(g["pdws"][t, band])
                self.ctrl[k].observe(t, band, hit, snr, pdws, now)
                self._record(k, cfg, t + 1, float(g["dwell_time_s"][t]), FIXED_DWELL_S, band, hit, snr, pdws,
                             score, belief, info)
            time.sleep(0.002)

    def _files(self, k):
        d, sfx = self.scilab_dir, "_" + k
        return {"obs": d / f"ml_observation_closed_loop{sfx}.csv", "cmd": d / f"ml_scan_command{sfx}.txt",
                "status": d / f"ml_closed_loop_status{sfx}.txt", "tracks": d / f"ml_closed_loop_tracks{sfx}.csv",
                "positions": d / f"ml_closed_loop_positions{sfx}.csv",
                "request": d / f"ml_closed_loop_request{sfx}.txt", "log": d / f"scilab_background{sfx}.log"}

    def _launch(self, k, cfg):
        f = self._files(k)
        for key in ("status", "cmd"):
            try:
                f[key].unlink()
            except FileNotFoundError:
                pass
        f["request"].write_text(f"{cfg} 0\n")
        script = str(self.scilab_dir / "main_closed_loop.sce").replace("\\", "/")
        cmd = [self.scilab_exe, "-nb", "-e", f"exec('{script}',-1); exit();"]
        kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)} if os.name == "nt" else {}
        log = open(f["log"], "a", encoding="utf-8", errors="replace")
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} configuration {cfg}, strategy {k}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(self.scilab_dir), env=dict(os.environ, SPECTRA_TAG=k),
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
        log.close()
        return proc

    def _log_tail(self, k):
        try:
            return " | ".join(self._files(k)["log"].read_text(errors="replace").strip().splitlines()[-3:])
        except OSError:
            return ""

    def _stop(self, k, proc):
        if proc is None or proc.poll() is not None:
            return
        try:
            write_command(self._files(k)["cmd"], -1, 0)
            proc.wait(timeout=5)
        except Exception:
            proc.terminate()

    def _live(self, k, cfg, my):
        f = self._files(k)
        proc = self._launch(k, cfg)
        t0 = time.time()
        while True:
            if self._halted(my):
                self._stop(k, proc)
                return
            st = read_status(f["status"])
            if st and st[0] == "READY" and len(st) > 1 and int(st[1]) == cfg:
                break
            if proc.poll() is not None:
                raise RuntimeError(f"Scilab stopped before it was ready (log: {f['log']}: {self._log_tail(k)})")
            if time.time() - t0 > READY_TIMEOUT_S:
                raise RuntimeError(f"Scilab not ready after {READY_TIMEOUT_S} s (log: {f['log']})")
            time.sleep(0.2)
        with self.lock:
            if k == "ai":
                try:
                    pos = pd.read_csv(f["positions"])
                    self.positions = {"stations": pos[pos.kind == "station"][["id", "x_km", "y_km"]].values.tolist(),
                                      "emitters": pos[pos.kind == "emitter"][["id", "x_km", "y_km"]].values.tolist()}
                except Exception:
                    pass
            self.phase = "running"
            self.message = (f"LIVE: {' + '.join(LABEL[x] for x in self.run_keys)} scanning configuration {cfg} "
                            f"({self.seg(cfg).split}) on Scilab.")
        obs_f, trk_f = CsvFollower(f["obs"]), CsvFollower(f["tracks"])
        with self.lock:
            band, score, belief, info, dwell = self._decide(k, 0)
        pending = {1: (band, score, belief, info)}
        write_command(f["cmd"], 1, band, dwell)
        tracks = collections.defaultdict(list)
        last = time.time()
        done = False
        while True:
            if self._halted(my):
                self._stop(k, proc)
                return
            for r in trk_f.new_rows():
                x, y = fnum(r["x_km"]), fnum(r["y_km"])
                if math.isfinite(x) and math.isfinite(y):
                    tracks[int(fnum(r["step"]))].append((x, y, int(fnum(r["tx_id"])), fnum(r["confidence"])))
            rows = obs_f.new_rows()
            if not rows:
                if done:
                    break
                st = read_status(f["status"])
                if st and st[0] == "DONE":
                    done = True
                    continue
                if proc.poll() is not None:
                    raise RuntimeError(f"Scilab stopped (log: {f['log']}: {self._log_tail(k)})")
                if time.time() - last > 600:
                    raise RuntimeError("No data from Scilab for 10 minutes")
                time.sleep(0.002)
                continue
            for r in rows:
                step = int(fnum(r["step"]))
                if step not in pending:
                    continue
                band_chosen, score, belief, info = pending.pop(step)
                band = int(band_of_centre(fnum(r["scan_center_mhz"])))
                pdws = fnum(r["total_pdws"])
                hit = int((pdws > 0) or fnum(r["hit"]) > 0)
                snr = fnum(r["mean_snr_db"]) if hit else 0.0
                with self.lock:
                    if self._halted(my):
                        break
                    if band != band_chosen:
                        self.log(f"WARNING {LABEL[k]} scan {step}: Scilab scanned band {band + 1}, chose {band_chosen + 1}")
                    scan_t, scan_d = fnum(r["scan_time_s"]), fnum(r["dwell_s"])
                    self.ctrl[k].observe(step - 1, band, hit, snr if hit else 0.0, pdws if math.isfinite(pdws) else 0.0,
                                         scan_t)
                    self._record(k, cfg, step, fnum(r["scan_time_s"]), fnum(r["dwell_s"]), band, hit, snr,
                                 pdws, score, belief, info, geo_rows=tracks.pop(step, []))
                last = time.time()
                self._wait_if_paused(my)
                if self._halted(my):
                    break
                with self.lock:
                    band, score, belief, info, dwell = self._decide(k, step, scan_t + scan_d)
                pending[step + 1] = (band, score, belief, info)
                write_command(f["cmd"], step + 1, band, dwell)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()

    def _autosave_audit(self):
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            path = REPORTS_DIR / f"spectra_audit_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
            path.write_bytes(self.audit_xlsx())
            self.last_audit_file = str(path)
            self.log(f"Audit saved: {path}")
        except Exception as e:
            self.log(f"Audit could not be saved: {e}")

    # ------------------------------------------------------------ metrics (dwell-length aware)
    def metrics(self, k, cfgs):
        agg = new_counter()
        found = detectable = lpi_f = lpi_d = lpd_f = lpd_d = 0
        delays = []
        sim = 0.0
        for cfg in cfgs:
            c = self.stats[k].get(cfg)
            if not c or c["scans"] == 0:
                continue
            for key in ("scans", "hits", "tp", "fp", "tn", "fn", "retunes", "pdws", "listen"):
                agg[key] += c[key]
            tr = self.seg(cfg).truth
            prog = c["progress_s"]
            sim += prog
            for x, t0 in tr.first_detectable.items():
                if t0 <= prog:
                    detectable += 1
                    ft = c["found"].get(x)
                    delays.append((ft if ft is not None else prog) - t0)
                    if tr.lpi.get(x) == 1:
                        lpi_d += 1
                        lpi_f += int(ft is not None)
                    if tr.lpd.get(x) == 1:
                        lpd_d += 1
                        lpd_f += int(ft is not None)
            found += len(c["found"])
        n = max(agg["scans"], 1)
        return {"scans": agg["scans"], "hits": agg["hits"], "rate": 100.0 * agg["hits"] / n,
                "per_s": agg["hits"] / max(sim, 1e-9), "found": found, "detectable": detectable,
                "delay_s": float(np.mean(delays)) if delays else 0.0,
                "pd": 100.0 * agg["tp"] / max(agg["tp"] + agg["fn"], 1),
                "pfa": 100.0 * agg["fp"] / max(agg["fp"] + agg["tn"], 1), "pdws": agg["pdws"],
                "retunes": agg["retunes"], "mean_dwell_ms": 1000.0 * agg["listen"] / n,
                "lpi": f"{lpi_f}/{lpi_d}", "lpd": f"{lpd_f}/{lpd_d}", "lpi_n": lpi_f, "lpd_n": lpd_f,
                "tp": agg["tp"], "fp": agg["fp"], "tn": agg["tn"], "fn": agg["fn"]}

    @staticmethod
    def comparison_rows(ms):
        def v(n, text):
            return {"n": n, "text": text}
        spec = [
            ("Transmitters found", "distinct transmitters detected / detectable", "higher",
             lambda m: v(m["found"], f"{m['found']} / {m['detectable']}")),
            ("Time to first detection", "mean over ALL detectable transmitters (missed = until now)", "lower",
             lambda m: v(round(m["delay_s"], 3), f"{m['delay_s']:.2f} s")),
            ("Detections per second", "detections / seconds of RF time", "higher",
             lambda m: v(round(m["per_s"], 2), f"{m['per_s']:.2f}")),
            ("Detection rate", "share of scans with a detection", "higher",
             lambda m: v(round(m["rate"], 1), f"{m['rate']:.1f}%")),
            ("Pd on active bands", "detections / scans of bands with a real transmitter", "higher",
             lambda m: v(round(m["pd"], 1), f"{m['pd']:.1f}%")),
            ("False alarms (Pfa)", "detections on bands with no real transmitter", "lower",
             lambda m: v(round(m["pfa"], 2), f"{m['pfa']:.2f}%")),
            ("Pulses intercepted", "receiver PDWs collected", "higher",
             lambda m: v(round(m["pdws"]), f"{m['pdws']:,.0f}")),
            ("LPD transmitters found", "low-probability-of-detection transmitters detected", "higher",
             lambda m: v(m["lpd_n"], m["lpd"])),
            ("Mean dwell", "listening time per scan", None,
             lambda m: v(round(m["mean_dwell_ms"], 2), f"{m['mean_dwell_ms']:.1f} ms")),
        ]
        return [{"metric": a, "sub": b, "better": c, "values": {k: fn(ms[k]) for k in KEYS}} for a, b, c, fn in spec]

    def _load_algorithm_comparison(self):
        path = REPORTS_DIR / "algorithm_comparison.xlsx"
        if not path.exists():
            return None
        try:
            t = pd.read_excel(path, sheet_name="Comparison")
            return {"columns": list(t.columns), "rows": t.astype(object).where(t.notna(), None).values.tolist(),
                    "file": str(path)}
        except Exception as e:
            print(f"algorithm_comparison.xlsx not readable: {e}")
            return None

    def comparison(self):
        if tuple(self.run_keys) != KEYS:
            return {"note": "Single-policy run: only " + LABEL[self.run_keys[0]] + " was executed. Switch on COMPARE ALL "
                            "and press START to run all three policies side by side.",
                    "unseen": None, "train": None, "perConfig": [], "algorithms": self.algorithms,
                    "compareMode": False}
        done = [c for c in self.available if self.stats["ai"].get(c, {}).get("scans", 0) > 0]
        unseen = [c for c in done if c in self.test_configs]
        train = [c for c in done if c not in self.test_configs]

        def block(cfgs, label):
            if not cfgs:
                return None
            return {"caption": f"{label}: configuration(s) {', '.join(map(str, cfgs))}",
                    "rows": self.comparison_rows({k: self.metrics(k, cfgs) for k in KEYS})}
        per = []
        for c in done:
            m = {k: self.metrics(k, [c]) for k in KEYS}
            best = max(m[k]["found"] for k in KEYS)
            win = [LABEL[k] for k in KEYS if m[k]["found"] == best]
            per.append({"config": c, "split": self.seg(c).split,
                        **{k: f"{m[k]['found']} / {m[k]['hits']}" for k in KEYS},
                        "winner": win[0] if len(win) == 1 else "tie"})
        return {"note": ("All three strategies scan the same RF environment at the same time, each on its own "
                         "receiver. 'Unseen' configurations were never used to train or tune the ML (fair test)."),
                "unseen": block(unseen, "Unseen"), "train": block(train, "Training"), "perConfig": per,
                "algorithms": self.algorithms, "compareMode": True}

    def lpi_rows(self, k):
        if not self.lpi_models:
            return []
        out = []
        for (cfg, tx), idx in sorted(self.captured[k].items()):
            if len(idx) < MIN_PULSES:
                continue
            key = (k, cfg, tx, len(idx))
            tr = self.seg(cfg).truth
            if key not in self.lpi_cache:
                ii = np.asarray(idx)
                x = pulse_features(tr.time[ii], tr.freq[ii], tr.pw[ii], tr.amp[ii])[None, :]
                self.lpi_cache[key] = (float(self.lpi_models["lpi"].predict(x)[0]),
                                       float(self.lpi_models["lpd"].predict(x)[0]))
            p1, p2 = self.lpi_cache[key]
            fm, fc, _ = self.fmode[k].get((cfg, tx))
            out.append({"tx": f"C{cfg} TX-{tx}", "pulses": len(idx), "freqMode": fm, "freqModeConf": fc,
                        "pLPI": p1, "predLPI": int(p1 >= 0.5),
                        "trueLPI": tr.lpi.get(tx), "pLPD": p2, "predLPD": int(p2 >= 0.5), "trueLPD": tr.lpd.get(tx)})
        return out

    def live_active(self):
        return any(w.is_alive() for w in self.workers)

    # ------------------------------------------------------------ state
    def state(self):
        with self.lock:
            a = self.active
            cfgs = [c for c in self.available if self.stats[a].get(c, {}).get("scans", 0) > 0]
            kpi = {k: (self.metrics(k, cfgs) if k in self.run_keys else None) for k in KEYS}
            last = self.rows[a][-1] if self.rows[a] else None
            log = [{**{c: r[c] for c in LOG_COLUMNS}, "tier": r["tier"],
                    "decision_label": r.get("decision_label", ""), "why": r.get("why", ""),
                    "freq_mode": r.get("freq_mode", ""), "freq_mode_confidence": r.get("freq_mode_confidence")}
                   for r in self.rows[a][-40:][::-1]]
            heat = list(self.heat[a])
            has_belief = any(h[1] is not None for h in heat)
            z = (np.stack([h[1] if h[1] is not None else np.zeros(N_BANDS, np.float32) for h in heat], axis=1)
                 .round(2).tolist() if heat else [])
            series = {}
            for k in self.run_keys:
                s = self.series[k]
                idx = thin_idx(len(s["t"]))
                series[k] = {n: [s[n][i] for i in idx] for n in ("t", "detections", "rate", "found")}
            lrows = self.lpi_rows(a) if a in self.run_keys else []
            prog = {k: {"config": self.progress[k]["config"], "time_s": self.progress[k]["time_s"],
                        "scans": len(self.rows[k])} for k in self.run_keys}
            hit_miss = [{"strategy": LABEL[k], "scans": kpi[k]["scans"], "hits": kpi[k]["hits"],
                         "misses": kpi[k]["scans"] - kpi[k]["hits"], "hitPct": kpi[k]["rate"], "tp": kpi[k]["tp"],
                         "fp": kpi[k]["fp"], "tn": kpi[k]["tn"], "fn": kpi[k]["fn"],
                         "found": f"{kpi[k]['found']}/{kpi[k]['detectable']}",
                         "meanDwellMs": kpi[k]["mean_dwell_ms"]} for k in self.run_keys]
            ai = self.ctrl.get("ai")
            lat = np.asarray(self.latency_ms, dtype=float)
            latency = ({"n": int(lat.size), "mean": float(lat.mean()), "p95": float(np.percentile(lat, 95)),
                        "max": float(lat.max())} if lat.size else None)
            st = {
                "running": self.running, "phase": self.phase, "message": self.message, "mode": self.mode,
                "active": self.live_active(), "selection": self.selection, "configId": self.cur_cfg,
                "availableConfigs": self.available, "testConfigs": self.test_configs,
                "runSeconds": self.total_s, "progress": prog, "lastAudit": self.last_audit_file,
                "strategy": KEY_TO_STRATEGY[a], "strategyKey": a, "strategyLabel": LABEL[a],
                "runMode": self.run_mode, "runKeys": list(self.run_keys),
                "ppoForced": {"enabled": self.ppo_forced_off, "at": self.ppo_forced_at,
                              "source": self.ppo_forced_by},
                "runLabels": [LABEL[k] for k in self.run_keys],
                "receiverSource": ("live Scilab receiver" if self.mode == "scilab" else "recorded receiver sweep"),
                "telemetry": {
                    "last": None if last is None else {**{c: last[c] for c in LOG_COLUMNS}, "tier": last["tier"],
                                                       "tierName": TIER_NAMES.get(last["tier"], "baseline"),
                                                       "decision_label": last.get("decision_label", ""),
                                                       "why": last.get("why", ""), "snr_db": last["snr_db"],
                                                       "freq_mode": last.get("freq_mode", ""),
                                                       "freq_mode_confidence": last.get("freq_mode_confidence")},
                    "tierCounts": {str(t): n for t, n in self.tier_counts.items()},
                    "spectraRunning": "ai" in self.run_keys,
                    "forcedTier": ai.force_tier if ai else None,
                    "ppoLoaded": (ai.primary is not None) if ai else None,
                    "ppoTimeoutMs": round(ai.ppo_timeout_s * 1000, 3) if ai else None,
                    "latencyMs": latency, "failovers": list(self.failovers)[::-1][:50]},
                "decisionLog": log,
                "heatmap": {"t": [h[0] for h in heat], "centres": BAND_CENTRE_MHZ.tolist(), "z": z,
                            "hasModel": has_belief,
                            "picks": [[h[0], float(BAND_CENTRE_MHZ[h[2]]), h[3]] for h in heat],
                            "label": {"ai": "Spectra fused P(target)",
                                      "whittle": "Whittle belief (probability the band is active)",
                                      "base": "Round Robin has no model - only its scans are shown"}[a]},
                "kpi": kpi, "series": series, "configStarts": self.config_starts,
                "comparison": self.comparison(), "hitMiss": hit_miss,
                "lpi": {"available": bool(self.lpi_models), "rows": lrows, "classified": len(lrows),
                        "lpiCorrect": sum(r["predLPI"] == r["trueLPI"] for r in lrows),
                        "lpdCorrect": sum(r["predLPD"] == r["trueLPD"] for r in lrows),
                        "definitions": {"LPI": f"peak power < {LPI_POWER_W:g} W and agile frequency or PRI",
                                        "LPD": f"< {LPD_DETECT_FRACTION:.0%} of pulses above the receiver threshold"},
                        "cv": self.lpi_cv},
                "receiver": {"detections": list(self.detections[a]), "geo": {
                    "stations": self.positions["stations"], "emitters": self.positions["emitters"],
                    "estimates": [list(g) for g in list(self.geo[a])[-1500:]]}},
                "events": list(self.events),
            }
            return clean(st)

    # ------------------------------------------------------------ commands
    def control(self, cmd, payload):
        if cmd == "START" and not self.running:
            for w in [w for w in self.workers if w.is_alive()]:
                if self.phase == "idle":
                    w.join(timeout=15)             # a RESET run is still stopping its Scilab instances
        with self.lock:
            if cmd == "START":
                if self.live_active():
                    self.running = True
                    self.log("Resumed.")
                    return {"success": True}
                self.run_id += 1
                self.run_keys = tuple(self._planned_keys())
                self._reset_run()
                try:                                       # ONLY the policies that run get a decision maker
                    self.ctrl = {k: self._make_ctrl(k) for k in self.run_keys}
                except Exception as e:
                    return self._fail(f"could not load {', '.join(LABEL[k] for k in self.run_keys)}: {e}")
                self.running = True
                self.phase = "starting"
                w = threading.Thread(target=self._run, args=(self.run_id,), daemon=True)
                self.workers = [w]
                w.start()
            elif cmd == "PAUSE":
                self.running = False
                self.log("PAUSED (Scilab waits for the next command).")
            elif cmd == "RESET":
                self.run_id += 1
                self.running = False
                self.phase = "idle"
                self.message = ""
                self._reset_run()
                self.log("Run RESET.")
            elif cmd == "STRATEGY":
                k = STRATEGY_TO_KEY.get(payload.get("strategy"), "ai")
                if self.live_active() and k not in self.run_keys:
                    return self._fail(f"{LABEL[k]} is not running in this session. RESET, choose it, then START.")
                self.active = k
                if not self.live_active():
                    self.run_keys = tuple(self._planned_keys()) if self.run_mode == "single" else KEYS
                self.log(f"{'Showing' if self.live_active() else 'Selected'} {LABEL[k]}.")
            elif cmd == "RUN_MODE":
                if self.live_active():
                    return self._fail("RESET the run before switching between single policy and COMPARE ALL.")
                mode = payload.get("mode")
                if mode not in ("single", "compare"):
                    return self._fail(f"unknown run mode {mode}")
                self.run_mode = mode
                self.run_keys = tuple(self._planned_keys())
                self._reset_run()
            elif cmd == "CONFIG":
                if self.live_active():
                    return self._fail("RESET the run before changing the configuration.")
                v = payload.get("config", "all")
                if v != "all":
                    v = int(v)
                    if v not in self.available:
                        return self._fail(f"configuration {v} is not prepared")
                self.selection = v
                self._reset_run()
            elif cmd == "FORCE_PPO_FAILURE":
                # B2: applies from the NEXT scan; works before, during and after a run.
                on = bool(payload.get("enabled"))
                self.ppo_forced_off = on
                self.ppo_forced_at = time.strftime("%H:%M:%S")
                self.ppo_forced_by = str(payload.get("source", "dashboard"))[:40]
                ai = self.ctrl.get("ai")
                if ai is not None:
                    ai.forced_off = on                       # B3: only Spectra's decider
                self.log(("PPO FORCED OFF by " + self.ppo_forced_by +
                          " - Spectra now decides with Tier 2 (Whittle)") if on
                         else "PPO restored - Spectra back on Tier 1")
            elif cmd in ("SPEED", "MODE"):
                pass
            else:
                return self._fail(f"unknown command {cmd}")
            return {"success": True}

    def _fail(self, msg):
        self.log("ERROR: " + msg)
        return {"success": False, "error": msg}

    def audit_xlsx(self):
        with self.lock:
            ref = self.run_keys[0]
            done = [c for c in self.available if self.stats[ref].get(c, {}).get("scans", 0) > 0]
            summaries = {}
            for name, cfgs in (("Summary_unseen", [c for c in done if c in self.test_configs]),
                               ("Summary_training", [c for c in done if c not in self.test_configs])):
                if cfgs:
                    rows = []
                    for k in self.run_keys:
                        m = self.metrics(k, cfgs)
                        rows.append({"strategy": LABEL[k], "configurations": ", ".join(map(str, cfgs)),
                                     "receiver": "live Scilab" if self.mode == "scilab" else "recorded sweep",
                                     "transmitters_found": m["found"], "transmitters_detectable": m["detectable"],
                                     "time_to_first_detection_s": round(m["delay_s"], 3),
                                     "detections_per_s": round(m["per_s"], 3), "scans": m["scans"], "hits": m["hits"],
                                     "detection_rate_pct": round(m["rate"], 2), "Pd_pct": round(m["pd"], 2),
                                     "Pfa_pct": round(m["pfa"], 3), "pulses_intercepted": m["pdws"],
                                     "LPD_found": m["lpd"], "LPI_found": m["lpi"],
                                     "mean_dwell_ms": round(m["mean_dwell_ms"], 2), "receiver_retunes": m["retunes"]})
                    summaries[name] = pd.DataFrame(rows)
            logs = {k: pd.DataFrame([{c: r[c] for c in LOG_COLUMNS} for r in self.rows[k]], columns=LOG_COLUMNS)
                    for k in self.run_keys}
            fails = pd.DataFrame(list(self.failovers))
            per = pd.DataFrame(self.comparison()["perConfig"])
            lpi = pd.DataFrame([{"strategy": LABEL[k], **r} for k in self.run_keys for r in self.lpi_rows(k)])
            geo = pd.DataFrame([{"strategy": LABEL[k], "x_km": g[0], "y_km": g[1], "tx_id": g[2], "confidence": g[3]}
                                for k in self.run_keys for g in self.geo[k]])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            for name, df in summaries.items():
                df.to_excel(w, sheet_name=name, index=False)
            if not per.empty:
                per.to_excel(w, sheet_name="Per_configuration", index=False)
            for k, name in (("ai", "Log_Spectra"), ("whittle", "Log_Whittle"), ("base", "Log_RoundRobin")):
                if k in logs:
                    logs[k].to_excel(w, sheet_name=name, index=False)
            if not fails.empty:
                fails.to_excel(w, sheet_name="Failover_events", index=False)
            if not lpi.empty:
                lpi.to_excel(w, sheet_name="LPI_LPD", index=False)
            if not geo.empty:
                geo.to_excel(w, sheet_name="Geolocation", index=False)
            pd.DataFrame({"item": ["logs", "time_to_first_detection_s", "decision"], "meaning": [
                "one sheet per strategy, 10 columns each (same as online_controller.py)",
                "mean over ALL transmitters that became detectable; one never found counts until the end",
                "T1 = Spectra (PPO), T2 = Whittle failover inside the Spectra run; POLICY_* = baseline policy"]}).to_excel(
                w, sheet_name="Notes", index=False)
        return buf.getvalue()


# ---------------------------------------------------------------------------- HTTP
def make_handler(engine, dash_dir):
    dash_dir = Path(dash_dir).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/v1/simulation/state":
                return self._json(engine.state())
            if path == "/api/v1/audit.xlsx":
                return self._send(200, engine.audit_xlsx(),
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                  {"Content-Disposition": 'attachment; filename="spectra_audit.xlsx"'})
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (dash_dir / rel).resolve()
            if dash_dir not in f.parents or not f.is_file():
                return self._send(404, b"Not found", "text/plain")
            return self._send(200, f.read_bytes(), mimetypes.guess_type(str(f))[0] or "application/octet-stream")

        def do_POST(self):
            if urlparse(self.path).path != "/api/v1/simulation/control":
                return self._json({"success": False, "error": "not found"}, 404)
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"success": False, "error": "bad JSON"}, 400)
            return self._json(engine.control(str(payload.get("command", "")).upper(), payload))

    return Handler


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Spectra dashboard server")
    ap.add_argument("--dashboard-dir", default=str(here.parent / "smart_scan_dashboard"))
    ap.add_argument("--scilab-dir", default=str(here.parent / "SmartScan_fast"))
    ap.add_argument("--scilab-exe", default=None, help="path to WScilex-cli.exe (auto-detected)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    args = ap.parse_args()
    if not (Path(args.dashboard_dir) / "index.html").exists():
        raise SystemExit(f"index.html not found in {args.dashboard_dir}. Use --dashboard-dir.")
    print("Loading models ...")
    engine = Engine(args.models_dir, args.scilab_dir, args.scilab_exe)
    print(f"Scilab folder : {engine.scilab_dir}")
    print(f"Scilab program: {engine.scilab_exe or 'NOT found'}")
    print(f"Receivers     : {'3 x LIVE Scilab (background, no windows)' if engine.mode == 'scilab' else 'recorded sweep (replay) - Scilab not available'}")
    print(f"Configurations: {engine.available}  (unseen by the ML: {engine.test_configs})")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine, args.dashboard_dir))
    print(f"Dashboard ready:  http://localhost:{args.port}    (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
