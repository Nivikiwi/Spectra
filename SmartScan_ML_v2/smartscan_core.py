"""
Smart Scan core - ONE code path shared by training, PPO, evaluation and the
live controller (so the features used in training are exactly the features
used at runtime).

Pipeline per 10-ms dwell t:
    FeatureTracker (receiver scan history only, no ground truth)
        -> LightGBM   : P(receiver detects something in band b at dwell t)
        -> DBN        : per-band hidden Markov belief "emitter active", Bayes-updated after each scan
        -> Fusion     : calibrated combination of LightGBM and DBN -> fused P(hit)
        -> Policy     : PPO (custom) | Whittle index (baseline) | Round Robin (baseline)
"""
import json
from pathlib import Path

import numpy as np

from smartscan_config import N_BANDS, BAND_CENTRE_MHZ, MODELS_DIR, DWELL_S

AGE_CAP = 2000.0
PPO_K = 8                        # PPO chooses among the 8 bands with the highest Whittle-informed index
PPO_FEATS_PER_CAND = 9
ML_GAMMA = 0.5                   # weight of the LightGBM+DBN probability ADDED to the Whittle index
                                 # (chosen by leave-one-config-out on the training configurations: blending the
                                 #  two beliefs inside the index lost information; adding them keeps both)
                                 # (chosen by 7-fold cross-validation on the training configurations)
PPO_OBS_DIM = PPO_K * PPO_FEATS_PER_CAND + 1
PHASE_WHITTLE_GAIN = 0.0         # how strongly a due-back beam raises the WHITTLE wake-up probability.
                                 # Tested on the training configurations: 0.05-0.6 all scored WORSE
                                 # (71 -> 60-67 transmitters found) because inflating the belief sends
                                 # the receiver back to KNOWN radars. Kept at 0; see CHANGELOG.
PHASE_GAIN = 0.0                 # how strongly the rotation model raises the DBN turn-on probability
                                 # when a band's beam is due back (0 = plain two-state chain)
REVISIT_BETA = 0.05              # weight of the radar-rotation revisit boost (chosen on development config 6)

FEATURE_NAMES = [
    "centre_ghz", "log_age", "never_scanned", "last_hit", "last_snr_db",
    "log_last_pdws", "hit_rate_s", "log_scans", "log_hits",
    "nbr_recent_hit", "nbr_hit_rate", "global_recent_hit_rate",
    "log_since_hit", "log_last_hit_gap", "hit_cycle_phase",
]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_grid(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


# =====================================================================
# Receiver scan history -> features  (observable information only)
# =====================================================================
class FeatureTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.scans = np.zeros(N_BANDS)
        self.hits = np.zeros(N_BANDS)
        self.last_scan = np.full(N_BANDS, -1.0)
        self.last_hit = np.zeros(N_BANDS)
        self.last_snr = np.zeros(N_BANDS)
        self.last_pdws = np.zeros(N_BANDS)
        self.recent = []                       # last 20 scan outcomes
        self.last_hit_t = np.full(N_BANDS, -1.0)   # dwell of the most recent detection
        self.prev_hit_t = np.full(N_BANDS, -1.0)   # dwell of the detection before that

    def features(self, t):
        never = self.last_scan < 0
        age = np.where(never, AGE_CAP, np.minimum(t - self.last_scan, AGE_CAP))
        hit_rate = (self.hits + 0.02) / (self.scans + 1.0)
        recent_hit = self.last_hit * np.exp(-age / 20.0)
        pad_r = np.concatenate([[0.0], recent_hit, [0.0]])
        nbr_recent = np.maximum(pad_r[:-2], pad_r[2:])
        pad_h = np.concatenate([[hit_rate[0]], hit_rate, [hit_rate[-1]]])
        nbr_rate = 0.5 * (pad_h[:-2] + pad_h[2:])
        g = float(np.mean(self.recent)) if self.recent else 0.0
        # Timing: rotating radars re-appear on a cycle -> time since last hit vs last gap
        has_hit = self.last_hit_t >= 0
        since_hit = np.where(has_hit, np.minimum(t - self.last_hit_t, AGE_CAP), AGE_CAP)
        has_gap = self.prev_hit_t >= 0
        gap = np.where(has_gap, np.minimum(self.last_hit_t - self.prev_hit_t, AGE_CAP), AGE_CAP)
        phase = np.where(has_gap, np.clip(since_hit / np.maximum(gap, 1.0), 0.0, 3.0), 3.0)
        X = np.column_stack([
            BAND_CENTRE_MHZ / 1000.0, np.log1p(age), never.astype(float), self.last_hit,
            self.last_snr, np.log1p(self.last_pdws), hit_rate, np.log1p(self.scans),
            np.log1p(self.hits), nbr_recent, nbr_rate, np.full(N_BANDS, g),
            np.log1p(since_hit), np.log1p(gap), phase,
        ]).astype(np.float32)
        return X, age

    def update(self, t, band, hit, snr=0.0, pdws=0.0):
        b = int(band)
        self.scans[b] += 1
        self.hits[b] += float(hit)
        self.last_scan[b] = t
        self.last_hit[b] = float(hit)
        self.last_snr[b] = float(snr) if np.isfinite(snr) else 0.0
        self.last_pdws[b] = float(pdws) if np.isfinite(pdws) else 0.0
        if hit:
            self.prev_hit_t[b] = self.last_hit_t[b]
            self.last_hit_t[b] = t
        self.recent.append(float(hit))
        if len(self.recent) > 20:
            self.recent.pop(0)


# =====================================================================
# DBN: independent 2-state hidden Markov chain per band (factored DBN)
# hidden X_b(t) = 1 if a transmitter is active in band b.
# =====================================================================
class DBN:
    def __init__(self, params):
        self.p01 = np.asarray(params["p01"], float)
        self.p11 = np.asarray(params["p11"], float)
        self.pd = np.asarray(params["pd"], float)
        self.pfa = float(params["pfa"])
        denom = 1.0 - self.p11 + self.p01
        self.stationary = np.clip(np.divide(self.p01, denom, out=np.zeros(N_BANDS), where=denom > 0), 1e-4, 0.999)
        self.reset()

    def reset(self):
        self.belief = self.stationary.copy()

    def predict(self, phase=None):
        """One step forward. `phase` (0..1 per band, from the rotation model) raises the
        turn-on probability of bands whose beam is due back now: a rotating radar is periodic,
        not a memoryless on/off switch, so its belief should PEAK on schedule rather than
        decay uniformly. phase = 0 reproduces the plain two-state chain exactly."""
        p01 = self.p01
        if phase is not None and PHASE_GAIN > 0.0:
            # raise p01 toward 1 in proportion to how strongly the beam is expected now
            p01 = p01 + PHASE_GAIN * np.clip(phase, 0.0, 1.0) * (1.0 - p01)
        self.belief = self.belief * self.p11 + (1.0 - self.belief) * p01
        return self.belief

    def hit_probability(self):
        return self.pd * self.belief + self.pfa * (1.0 - self.belief)

    def update(self, band, hit):
        b = int(band)
        prior = self.belief[b]
        if hit:
            num = self.pd[b] * prior
            den = num + self.pfa * (1.0 - prior)
        else:
            num = (1.0 - self.pd[b]) * prior
            den = num + (1.0 - self.pfa) * (1.0 - prior)
        self.belief[b] = num / den if den > 0 else prior


# =====================================================================
# Baselines
# =====================================================================
class RoundRobin:
    def __init__(self):
        self.next = 0

    def reset(self):
        self.next = 0

    def select(self):
        b = self.next
        self.next = (self.next + 1) % N_BANDS
        return b


class Whittle:
    """Whittle index policy for restless bandits with 2-state Markov bands
    (Liu & Zhao model): each band is ON/OFF (detectable or not), scanning a band
    reveals its state. The index table W_b(omega) is pre-computed in step 3."""

    def __init__(self, params):
        self.p01 = np.asarray(params["p01"], float)
        self.p11 = np.asarray(params["p11"], float)
        self.grid = np.asarray(params["omega_grid"], float)
        self.table = np.asarray(params["index_table"], float)       # (N_BANDS, len(grid))
        denom = 1.0 - self.p11 + self.p01
        self.stationary = np.clip(np.divide(self.p01, denom, out=np.zeros(N_BANDS), where=denom > 0), 0.0, 1.0)
        self.reset()

    def reset(self):
        self.omega = self.stationary.copy()

    def index_of(self, omega):
        """Whittle index of every band for belief vector omega (linear interpolation)."""
        g = self.grid
        pos = np.clip(omega, g[0], g[-1]) * (len(g) - 1)
        i0 = np.clip(np.floor(pos).astype(int), 0, len(g) - 2)
        w = pos - i0
        rows = np.arange(N_BANDS)
        return (1 - w) * self.table[rows, i0] + w * self.table[rows, i0 + 1]

    def select(self):
        return int(np.argmax(self.index_of(self.omega)))

    def update(self, band, hit, phase=None):
        """Belief for the NEXT dwell.

        The plain model says a silent band wakes up with a constant probability p01, which
        describes a random switch. A rotating radar is periodic: it is due back at a known
        moment. `phase` (0..1 per band, measured from this band's own detection bursts) raises
        the wake-up probability when the beam is due, so the index itself understands rotation.
        phase = None reproduces the plain two-state chain exactly (used by the baseline).
        """
        p01 = self.p01
        if phase is not None and PHASE_WHITTLE_GAIN > 0.0:
            p01 = p01 + PHASE_WHITTLE_GAIN * np.clip(phase, 0.0, 1.0) * (1.0 - p01)
        nxt = self.omega * self.p11 + (1 - self.omega) * p01
        b = int(band)
        nxt[b] = self.p11[b] if hit else self.p01[b]
        self.omega = nxt


def whittle_index_table(p01, p11, beta=0.9, n_omega=101, n_lambda=201, iters=400):
    """Numerical Whittle index for one 2-state band (Liu & Zhao model).
    Active: reward omega, state revealed -> next belief p11 (ON) or p01 (OFF).
    Passive: subsidy lambda, next belief T(omega) = omega*p11 + (1-omega)*p01.
    W(omega) = smallest subsidy that makes passive optimal. Returns (grid, W)."""
    om = np.linspace(0.0, 1.0, n_omega)
    lam = np.linspace(0.0, 1.0, n_lambda)[:, None]

    def interp_weights(x):
        pos = np.clip(np.asarray(x, float), 0.0, 1.0) * (n_omega - 1)
        i0 = np.clip(np.floor(pos).astype(int), 0, n_omega - 2)
        return i0, pos - i0

    i11, w11 = interp_weights(p11)
    i01, w01 = interp_weights(p01)
    iT, wT = interp_weights(om * p11 + (1 - om) * p01)

    def q_values(V):
        v11 = ((1 - w11) * V[:, i11] + w11 * V[:, i11 + 1])[:, None]
        v01 = ((1 - w01) * V[:, i01] + w01 * V[:, i01 + 1])[:, None]
        vT = (1 - wT) * V[:, iT] + wT * V[:, iT + 1]
        active = om + beta * (om * v11 + (1 - om) * v01)
        passive = lam + beta * vT
        return active, passive

    V = np.zeros((n_lambda, n_omega))
    for _ in range(iters):
        active, passive = q_values(V)
        V_new = np.maximum(active, passive)
        done = np.max(np.abs(V_new - V)) < 1e-8
        V = V_new
        if done:
            break
    active, passive = q_values(V)
    passive_ok = passive >= active - 1e-10
    k = np.where(passive_ok.any(axis=0), np.argmax(passive_ok, axis=0), n_lambda - 1)
    W = lam[k, 0]
    return om, np.maximum.accumulate(W)            # the index is monotone in omega


# =====================================================================
# Custom pipeline: LightGBM -> DBN -> fusion (-> PPO)
# =====================================================================
class PulseRateTracker:
    """Measured pulse rate per band: how many detectable pulses the RECEIVER sees per second
    while listening there. Pure observation - pulses counted, seconds counted, divide.
    Used to choose how long to dwell (see decision_tiers.dwell_for_band)."""

    ALPHA = 0.3                     # weight of the newest scan in the running average

    def __init__(self):
        self.reset()

    def reset(self):
        self.rate = np.full(N_BANDS, np.nan)     # pulses per second, NaN = never measured
        self.listened = np.zeros(N_BANDS)        # seconds spent on this band

    def observe(self, band, pdws, dwell_s):
        if dwell_s is None or not np.isfinite(dwell_s) or dwell_s <= 0:
            return
        b = int(band)
        r = (float(pdws) if np.isfinite(pdws) else 0.0) / float(dwell_s)
        self.rate[b] = r if not np.isfinite(self.rate[b]) else (1 - self.ALPHA) * self.rate[b] + self.ALPHA * r
        self.listened[b] += float(dwell_s)

    def rate_for(self, band):
        """Rate for one band; falls back to the median of bands that HAVE produced pulses."""
        b = int(band)
        r = self.rate[b]
        if np.isfinite(r) and r > 0:
            return float(r)
        seen = self.rate[np.isfinite(self.rate) & (self.rate > 0)]
        return float(np.median(seen)) if seen.size else float("nan")


class RevisitPredictor:
    """Radar-rotation timing. Rotating radars light up a band on a regular cycle. From Spectra's
    OWN detections it records when each detection burst starts, estimates the cycle, and returns
    a boost (0..1) for bands whose beam is expected back now. Whittle cannot use this."""

    BURST_GAP_S = 0.05           # a hit more than 50 ms after the previous hit starts a new burst
    MIN_PERIOD_S = 0.2           # ignore faster "periods" (continuous emitters)
    MAX_BURSTS = 12

    def __init__(self):
        self.reset()

    def reset(self):
        self.starts = [[] for _ in range(N_BANDS)]
        self.last_hit_s = np.full(N_BANDS, -np.inf)
        self.period = np.zeros(N_BANDS)          # 0 = no regular cycle found (yet)
        self.last_start = np.full(N_BANDS, -np.inf)
        self.predicted_next = np.full(N_BANDS, np.nan)   # when this band's beam is expected back
        self.errors_s = []                       # |predicted - actual| for each verified prediction

    def _estimate(self, b):
        st = self.starts[b]
        if len(st) < 3:
            self.period[b] = 0.0
            return
        gaps = np.diff(st)
        gaps = gaps[gaps >= self.MIN_PERIOD_S]
        if len(gaps) < 2:
            self.period[b] = 0.0
            return
        p = float(np.min(gaps))                  # missed bursts give multiples of the period
        ratio = gaps / p
        regular = np.abs(ratio - np.round(ratio)) <= 0.15
        p_ref = float(np.median(gaps[regular] / np.round(ratio[regular]))) if regular.any() else p
        self.period[b] = p_ref if regular.mean() >= 0.6 else 0.0

    def intercept_time_error_s(self):
        """Average intercept time error: mean |predicted return time - actual return time|
        over every prediction the model made and later verified. A figure of merit required by
        the problem statement; it measures how accurately the model predicts WHEN a band becomes
        active, as opposed to whether it does."""
        return float(np.mean(self.errors_s)) if self.errors_s else float("nan")

    def predictions_made(self):
        return len(self.errors_s)

    def observe(self, band, hit, now_s):
        if not hit or now_s is None or not np.isfinite(now_s):
            return
        b = int(band)
        if now_s - self.last_hit_s[b] > self.BURST_GAP_S:
            # a new burst has started: score the prediction we made for it, before updating
            pred = self.predicted_next[b]
            if np.isfinite(pred) and self.period[b] > 0:
                err = abs(float(now_s) - float(pred))
                if err <= 3.0 * self.period[b]:          # ignore bursts we never predicted
                    self.errors_s.append(err)
                    del self.errors_s[:-2000]
            st = self.starts[b]
            st.append(float(now_s))
            del st[:-self.MAX_BURSTS]
            self.last_start[b] = float(now_s)
            self._estimate(b)
            self.predicted_next[b] = (float(now_s) + self.period[b]) if self.period[b] > 0 else np.nan
        self.last_hit_s[b] = float(now_s)

    def boost(self, now_s):
        if now_s is None or not np.isfinite(now_s):
            return np.zeros(N_BANDS)
        p = self.period
        ok = p > 0
        out = np.zeros(N_BANDS)
        if not ok.any():
            return out
        since = now_s - self.last_start[ok]
        k = np.maximum(np.round(since / p[ok]), 1.0)
        delta = since - k * p[ok]                               # 0 = beam expected exactly now
        width = np.maximum(0.03, 0.06 * p[ok])
        waiting = (now_s - self.last_hit_s[ok]) > self.BURST_GAP_S   # not already inside a burst
        out[ok] = np.exp(-0.5 * (delta / width) ** 2) * waiting
        return out


class CustomCore:
    def __init__(self, models_dir=MODELS_DIR, lgbm_file="lightgbm_model.txt"):
        import lightgbm as lgb
        models_dir = Path(models_dir)
        self.booster = lgb.Booster(model_file=str(models_dir / lgbm_file))
        if self.booster.num_feature() != len(FEATURE_NAMES):
            raise ValueError("LightGBM model feature count does not match FEATURE_NAMES. Re-run step 2.")
        dbn_path = models_dir / "dbn_params.json"
        self.dbn = DBN(json.loads(dbn_path.read_text())) if dbn_path.exists() else None
        fus_path = models_dir / "fusion.json"
        self.fusion = json.loads(fus_path.read_text()) if fus_path.exists() else None
        wh_path = models_dir / "whittle_params.json"
        self.whittle = Whittle(json.loads(wh_path.read_text())) if wh_path.exists() else None
        self.tracker = FeatureTracker()
        self.revisit = RevisitPredictor()
        self.pulse_rate = PulseRateTracker()
        self.revisit_beta = REVISIT_BETA
        self.last_det_snr = np.full(N_BANDS, np.nan)   # SNR of each band's most recent DETECTION
        self.last = {}

    def reset(self):
        self.tracker.reset()
        self.revisit.reset()
        self.pulse_rate.reset()
        self.last_det_snr[:] = np.nan
        if self.dbn is not None:
            self.dbn.reset()
        if self.whittle is not None:
            self.whittle.reset()
        self.last = {}

    def lightgbm_prob(self, t):
        X, age = self.tracker.features(t)
        p = self.booster.predict(X, num_threads=1)
        return np.clip(p, 1e-6, 1 - 1e-6), age

    def score(self, t, now_s=None):
        """Run LightGBM -> DBN -> fusion (-> Whittle-informed index + revisit timing) for scan t.
        now_s = simulation time of the scan (seconds); if unknown, t x 10 ms is used."""
        if now_s is None:
            now_s = t * DWELL_S
        p_lgbm, age = self.lightgbm_prob(t)
        boost = self.revisit.boost(now_s)          # rotation model: is a beam due back now?
        if self.dbn is None or self.fusion is None:
            fused = p_lgbm
            p_dbn = np.full(N_BANDS, np.nan)
        else:
            self.dbn.predict(phase=boost)
            p_dbn = np.clip(self.dbn.hit_probability(), 1e-6, 1 - 1e-6)
            f = self.fusion
            fused = _sigmoid(f["w_lgbm"] * _logit(p_lgbm) + f["w_dbn"] * _logit(p_dbn) + f["bias"])
        # Whittle's strength (value of information) PLUS Spectra's own detection probability:
        # index of the band's own Markov belief, with the LightGBM+DBN probability added on top.
        if self.whittle is not None:
            index = self.whittle.index_of(self.whittle.omega) + ML_GAMMA * fused
        else:
            index = fused
        self.last = {"t": t, "now_s": now_s, "p_lgbm": p_lgbm, "p_dbn": p_dbn, "fused": fused, "age": age,
                     "index": index, "revisit": boost, "score": index + self.revisit_beta * boost}
        return fused

    def candidates(self):
        """The PPO_K bands with the highest score = Whittle-informed index + revisit boost."""
        return np.argsort(-self.last["score"], kind="stable")[:PPO_K]

    def ppo_observation(self, cand):
        L = self.last
        tr = self.tracker
        hit_rate = (tr.hits + 0.02) / (tr.scans + 1.0)
        p_dbn = np.nan_to_num(L["p_dbn"], nan=0.0)
        rows = np.column_stack([
            L["score"][cand], L["revisit"][cand], L["fused"][cand], L["p_lgbm"][cand], p_dbn[cand],
            np.log1p(L["age"][cand]) / np.log1p(AGE_CAP), tr.last_hit[cand], hit_rate[cand],
            BAND_CENTRE_MHZ[cand] / 16100.0,
        ])
        g = float(np.mean(tr.recent)) if tr.recent else 0.0
        return np.concatenate([rows.ravel(), [g]]).astype(np.float32)

    def observe(self, t, band, hit, snr=0.0, pdws=0.0, now_s=None, dwell_s=None):
        if self.dbn is not None:
            self.dbn.update(band, hit)
        if self.whittle is not None:
            # Spectra's own index belief is phase-aware; the standalone Whittle baseline
            # (Controller("whittle")) keeps the plain two-state model and is untouched.
            self.whittle.update(band, hit, phase=self.last.get("revisit"))
        self.tracker.update(t, band, hit, snr, pdws)
        self.revisit.observe(band, hit, t * DWELL_S if now_s is None else now_s)
        self.pulse_rate.observe(band, pdws, dwell_s)
        if hit and snr is not None and np.isfinite(snr) and snr > 0:
            self.last_det_snr[int(band)] = float(snr)


class Controller:
    """policy: 'custom' (LightGBM->DBN->Whittle-informed index->PPO),
    'greedy' (LightGBM->DBN->Whittle-informed index, no PPO; ablation),
    'whittle', 'round_robin'."""

    POLICIES = ("custom", "greedy", "whittle", "round_robin")

    def __init__(self, policy, models_dir=MODELS_DIR):
        if policy not in self.POLICIES:
            raise ValueError(f"Unknown policy {policy}; choose from {self.POLICIES}")
        self.policy = policy
        models_dir = Path(models_dir)
        self.core = CustomCore(models_dir) if policy in ("custom", "greedy") else None
        self.ppo = None
        if policy == "custom":
            from stable_baselines3 import PPO
            ppo_path = models_dir / "ppo_smart_scan.zip"
            if not ppo_path.exists():
                raise FileNotFoundError(f"{ppo_path} not found. Run step 4 first.")
            self.ppo = PPO.load(str(ppo_path), device="cpu")
            got = int(np.prod(self.ppo.observation_space.shape))
            if got != PPO_OBS_DIM:
                raise ValueError(f"PPO model expects {got} inputs but Spectra now provides {PPO_OBS_DIM}. "
                                 f"Re-run the training (step 4) to create a matching PPO model.")
        self.whittle = None
        if policy == "whittle":
            self.whittle = Whittle(json.loads((models_dir / "whittle_params.json").read_text()))
        self.rr = RoundRobin() if policy == "round_robin" else None
        self.reset()

    def reset(self):
        for o in (self.core, self.whittle, self.rr):
            if o is not None:
                o.reset()

    def decide(self, t, now_s=None):
        """Return (band, score_of_chosen_band). now_s = simulation time of the scan (optional)."""
        if self.policy in ("custom", "greedy"):
            fused = self.core.score(t, now_s)
            cand = self.core.candidates()
            if self.policy == "greedy":
                band = int(cand[0])
            else:
                obs = self.core.ppo_observation(cand)
                action, _ = self.ppo.predict(obs, deterministic=True)
                band = int(cand[int(np.asarray(action).item())])
            return band, float(fused[band])
        if self.policy == "whittle":
            band = self.whittle.select()
            return band, float(self.whittle.omega[band])
        band = self.rr.select()
        return band, float("nan")

    def observe(self, t, band, hit, snr=0.0, pdws=0.0, now_s=None, dwell_s=None):
        if self.core is not None:
            self.core.observe(t, band, hit, snr, pdws, now_s, dwell_s)
        if self.whittle is not None:
            self.whittle.update(band, hit)
