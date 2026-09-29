"""
Tiered decision making with per-decision telemetry (Tasks 2, 3, 4).

  Tier 1  PPO_CUSTOM   LightGBM -> DBN -> Whittle-informed index -> PPO   (primary)
                       The PPO forward pass has a hard per-decision timeout (PPO_TIMEOUT_S).
  Tier 2  WHITTLE      Whittle index, no ML dependency                    (failover)

A tier that fails (exception or timeout) hands over to the next one; the tier used and the
reason are returned with every decision. Force a tier for testing with the environment
variable SPECTRA_FORCE_TIER=1|2 (or force_tier=...).

Uses only the already-trained artifacts. Adaptive dwell is applied to tier-1 (custom) decisions only.
With no failure and no forced tier, decisions are identical to Controller("custom").
"""
import concurrent.futures
import math
import os
import time

import numpy as np

from smartscan_config import (MODELS_DIR, BAND_CENTRE_MHZ, FIXED_DWELL_S, DWELL_BASE_S, DWELL_K, DWELL_MIN_S, DWELL_MAX_S,
                              LPD_WEAK_MARGIN_DB, LPD_DWELL_GAIN, DETECTION_THRESHOLD_DB, SPECTRA_ADAPTIVE_DWELL,
                              DWELL_MODEL, RETUNE_OVERHEAD_S, DWELL_GRID_S,
                              PPO_TIMEOUT_S, FORCE_TIER_ENV, PPO_TIMEOUT_ENV)
from smartscan_core import Controller

TIER_NAMES = {1: "Tier 1 (PPO)", 2: "Tier 2 (Whittle)"}

# C1: plain-language label for every raw decision code. The raw code stays in the record
# (field "decision") for the log/export; the label is what the dashboard shows.
DECISION_LABELS = {
    "PPO_TOP_INDEX": "Track: highest-belief band",
    "PPO_ALT": "Discover: alternative band",
    "WHITTLE_TIMEOUT": "Failover: PPO too slow",
    "WHITTLE_ERROR": "Failover: PPO error",
    "WHITTLE_FORCED": "Failover: forced by operator",
    "WHITTLE_UNAVAILABLE": "Failover: PPO unavailable",
    "POLICY_WHITTLE": "Baseline: Whittle staleness",
    "POLICY_ROUND_ROBIN": "Baseline: sequential sweep",
}


def decision_label(code):
    """Plain-language label for a raw decision code (unknown codes fall back to the code)."""
    base = str(code).split(":")[-1].strip()
    if base.startswith("PPO_ALT"):
        return DECISION_LABELS["PPO_ALT"]
    for key, text in DECISION_LABELS.items():
        if base.startswith(key):
            return text
    return base
_TIER_ALIASES = {"1": 1, "ppo": 1, "ppo_custom": 1, "custom": 1, "2": 2, "whittle": 2}


def binary_entropy_bits(p):
    """H(p) in bits, 0 (certain) .. 1 (p = 0.5)."""
    if p is None or not math.isfinite(p):
        return float("nan")
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def adaptive_dwell_s(p, weak=False, lpd_gain=None):
    """Spectra only: dwell = clip(base * (1 + k * H(p)) * (1 + LPD gain if the band's signal is weak), min, max)."""
    h = binary_entropy_bits(p)
    if not math.isfinite(h):
        return float(FIXED_DWELL_S), h
    d = DWELL_BASE_S * (1.0 + DWELL_K * h)
    if weak:
        d *= 1.0 + (LPD_DWELL_GAIN if lpd_gain is None else lpd_gain)
    return float(min(max(d, DWELL_MIN_S), DWELL_MAX_S)), h


def dwell_for_rate(p_target, rate_per_s):
    """Physics-based dwell: pick T maximising  P(target) * (1 - exp(-r*T)) / (T + overhead).

    r is the MEASURED rate of detectable pulses on that band (pulses per second). A band that
    pulses often is caught quickly, so a short dwell wins; a sparse or weak emitter (typical of
    LPD) needs a longer look or the scan is wasted. Returns (dwell_s, expected P(detect))."""
    if rate_per_s is None or not math.isfinite(rate_per_s) or rate_per_s <= 0:
        return float(FIXED_DWELL_S), float("nan")
    p = float(p_target) if (p_target is not None and math.isfinite(p_target)) else 1.0
    best_t, best_v, best_pd = FIXED_DWELL_S, -1.0, float("nan")
    for T in DWELL_GRID_S:
        pd = 1.0 - math.exp(-rate_per_s * T)
        v = p * pd / (T + RETUNE_OVERHEAD_S)
        if v > best_v:
            best_t, best_v, best_pd = T, v, pd
    return float(min(max(best_t, DWELL_MIN_S), DWELL_MAX_S)), best_pd


def parse_force_tier(value):
    if value is None or str(value).strip() == "":
        return None
    key = str(value).strip().lower()
    if key not in _TIER_ALIASES:
        raise ValueError(f"Unknown tier '{value}'. Use 1 (PPO custom) or 2 (Whittle).")
    return _TIER_ALIASES[key]


class TieredDecider:
    """Drop-in replacement for Controller('custom'): decide(t) -> (band, score), observe(...), reset().
    .last_info holds the decision telemetry: p_target, confidence, dwell_s, decision, tier."""

    def __init__(self, models_dir=MODELS_DIR, force_tier=None, ppo_timeout_s=None, adaptive_dwell=None):
        self.force_tier = parse_force_tier(force_tier if force_tier is not None
                                           else os.environ.get(FORCE_TIER_ENV))
        if ppo_timeout_s is None:
            env = os.environ.get(PPO_TIMEOUT_ENV, "").strip()
            try:
                ppo_timeout_s = float(env) / 1000.0 if env else PPO_TIMEOUT_S
            except ValueError:
                raise ValueError(f"{PPO_TIMEOUT_ENV} must be a number of milliseconds, got '{env}'")
            if ppo_timeout_s <= 0:
                raise ValueError(f"{PPO_TIMEOUT_ENV} must be > 0")
        self.ppo_timeout_s = float(ppo_timeout_s)
        self.adaptive_dwell = SPECTRA_ADAPTIVE_DWELL if adaptive_dwell is None else bool(adaptive_dwell)
        self.lpd_gain = LPD_DWELL_GAIN
        self.primary = None
        self.primary_error = None
        try:
            self.primary = Controller("custom", models_dir)          # LightGBM + DBN + PPO
        except Exception as e:
            self.primary_error = f"{type(e).__name__}: {e}"
        self.whittle = Controller("whittle", models_dir)            # tier 2: no ML dependency
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._warm_up()
        self.last_info = None
        self.last_reason = ""
        self.forced_off = False          # B2: operator "force PPO failure" switch
        self.tier_counts = {1: 0, 2: 0}

    def _warm_up(self):
        """The first PPO call is slow (library start-up). Do it once now, in the worker thread,
        so the per-decision timeout only measures real decisions."""
        if self.primary is None:
            return
        try:
            from smartscan_core import PPO_OBS_DIM
            dummy = np.zeros(PPO_OBS_DIM, dtype=np.float32)
            for _ in range(3):
                self._pool.submit(self._ppo_action, dummy).result(timeout=30)
        except Exception:
            pass

    @property
    def core(self):
        return self.primary.core if self.primary is not None else None

    def reset(self):
        if self.primary is not None:
            self.primary.reset()
        self.whittle.reset()
        self.last_info = None

    def _ppo_action(self, obs):
        action, _ = self.primary.ppo.predict(obs, deterministic=True)
        return int(np.asarray(action).item())

    def _info(self, band, p, tier, decision, t0):
        if tier == 1 and self.adaptive_dwell:
            core = self.primary.core if self.primary is not None else None
            if DWELL_MODEL == "rate" and core is not None:
                dwell, _ = dwell_for_rate(p, core.pulse_rate.rate_for(band))
                h = binary_entropy_bits(p)
            else:
                snr = core.last_det_snr[int(band)] if core is not None else float("nan")
                weak = bool(np.isfinite(snr) and snr < DETECTION_THRESHOLD_DB + LPD_WEAK_MARGIN_DB)
                dwell, h = adaptive_dwell_s(p, weak, self.lpd_gain)
        else:
            dwell, h = float(FIXED_DWELL_S), binary_entropy_bits(p)
        code = decision.split(":", 1)[-1].split(" (")[0].strip()
        return {"band": int(band), "centre_mhz": float(BAND_CENTRE_MHZ[band]),
                "decision_label": decision_label(code), "reason": self.last_reason,
                "p_target": float(p) if p is not None and math.isfinite(p) else float("nan"),
                "confidence": (1.0 - h) if math.isfinite(h) else float("nan"),
                "dwell_s": dwell, "decision": decision, "tier": tier,
                "decision_ms": round((time.perf_counter() - t0) * 1000, 3)}

    def decide(self, t, now_s=None):
        t0 = time.perf_counter()
        why = None
        fused = None
        if self.primary is None:
            why = f"WHITTLE_UNAVAILABLE ({self.primary_error})"
        else:
            try:
                fused = self.primary.core.score(t, now_s)           # LightGBM -> DBN -> index + revisit
            except Exception as e:
                why = f"WHITTLE_ERROR (model: {type(e).__name__}: {e})"
        if self.force_tier == 2 or self.forced_off:
            why = "WHITTLE_FORCED (operator)"

        # ---------------- tier 1: PPO custom ----------------
        if why is None and not self.forced_off:
            try:
                core = self.primary.core
                cand = core.candidates()
                obs = core.ppo_observation(cand)
                fut = self._pool.submit(self._ppo_action, obs)
                a = fut.result(timeout=self.ppo_timeout_s)
                band = int(cand[a])
                tag = "PPO_TOP_INDEX" if a == 0 else f"PPO_ALT_{a}"
                self.last_reason = "top belief" if a == 0 else f"alternative rank {a}"
                info = self._info(band, float(fused[band]), 1,
                                  f"T1:{tag}" + (" (forced)" if self.force_tier == 1 else ""), t0)
                return self._done(info, float(fused[band]))
            except concurrent.futures.TimeoutError:
                why = f"WHITTLE_TIMEOUT (> {self.ppo_timeout_s * 1000:.0f} ms)"
                self._pool.shutdown(wait=False)                     # abandon the late call
                self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            except Exception as e:
                why = f"WHITTLE_ERROR ({type(e).__name__}: {e})"

        # ---------------- tier 2: Whittle index ----------------
        band, omega = self.whittle.decide(t)
        self.last_reason = why.split(" (")[-1].rstrip(")") if " (" in why else why
        info = self._info(band, float(omega), 2, f"T2:{why}", t0)
        score = float(fused[band]) if fused is not None else float(omega)
        return self._done(info, score)

    def _done(self, info, score):
        self.last_info = info
        self.tier_counts[info["tier"]] += 1
        return info["band"], score

    def observe(self, t, band, hit, snr=0.0, pdws=0.0, now_s=None, dwell_s=None):
        # both tiers are updated every dwell, so the backup is always ready
        if self.primary is not None:
            try:
                self.primary.observe(t, band, hit, snr, pdws, now_s, dwell_s)
            except Exception:
                pass
        self.whittle.observe(t, band, hit, snr, pdws)
