#!/usr/bin/env python3
"""RouterBench replay: off-policy tail-cost certification after policy selection.

Constructed logging on a real outcome table (RouterBench 0-shot, 5 LLMs; Hugging Face dataset
withmartian/routerbench, file routerbench_0shot.pkl, downloaded into --data-dir on first use).
  * Outcome Y = per-request dollar cost of the chosen model (dataset value).
    Quality = dataset score of the chosen model (reported, never certified).
  * 30% of unique prompts (dev) fit per-model quality/cost predictors, build the
    nested candidate families (K = 8 < 32 < 128), the logging base router, the
    safe default router and the budget tau.  Everything is then frozen.
  * 70% (eval pool) is the population.  One repetition draws n prompts with
    replacement, A ~ pi0^eps(.|x), and reveals only (x, A, cost_A, quality_A).
    The full table is used only for ground truth.

Methods (delta = 0.05, uniform prior over K => C = log(K/delta)):
  exceedance, exc_betting, exc_empbernstein, cdf_pacbayes, cdf_empbernstein   (common/certs.py)
      constrained selection: maximise IW quality among candidates with bound <= 1-p
  split   : half 1: maximise IW quality among candidates with plain IW exceedance
            estimate <= 1-p (fallback: smallest estimate); half 2: exceedance cert, K=1
  naive   : exceedance bound with C = log(1/delta), i.e. ignores the search (INVALID)

Validity stress tests:
  stress_interp  : up to 1024 deterministic routers that send an independent random
                   subset of prompts to the quality-greedy router and the rest to
                   the cheap base router; subset sizes calibrated on the eval
                   population so every true G_j is in (1-p)*(1, 1.02]; none safe.
  stress_natural : the real K-family with tau re-chosen on the eval population so
                   that min_j G_j = (1-p)*1.005 (none safe).
  Selection = smallest bound among feasible (most aggressive).  Every certificate
  is run both valid (C=log(K/delta)) and naive (C=log(1/delta)); plus split and a
  plug-in Wald UCB (naive_wald, invalid).

Optional (--posterior): fixed sampled mixtures of linear argmax routers
  (see build_posterior). All methods pay log(n_means/delta) for selection from
  this finite list. Gaussian KL values describe the generating distributions
  only; they do not certify the finite Monte Carlo mixtures.

Outputs (in --out): ground_truth_routers.csv (true G, quality, cost and tail/bulk overlaps per
candidate), ground_truth_stress_*.csv, summary_main.csv, headline.csv (n for 50% / 90%
certification), summary_stress.csv, meta.json, raw_*.csv.gz, run_log.txt.

Usage:
  python routerbench_replay.py --smoke --workers 4          # quick check (writes ../results/routerbench/custom)
  python routerbench_replay.py --preset main                # main study  -> ../results/routerbench/main
  python routerbench_replay.py --preset stress              # stress test -> ../results/routerbench/stress
The presets set exactly the arguments used for the shipped results (see PRESETS); any explicit
flag overrides them.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import certs  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results", "routerbench")
DATA_DIR = os.path.join(HERE, "data")
DATA_FILE = "routerbench_0shot.pkl"
MODELS = ["mistralai/mistral-7b-chat", "mistralai/mixtral-8x7b-chat",
          "gpt-3.5-turbo-1106", "claude-v2", "gpt-4-1106-preview"]
SHORT = ["Mistral-7B", "Mixtral-8x7B", "GPT-3.5", "Claude-2", "GPT-4"]
M = len(MODELS)
DELTA = 0.05
KAPPAS = np.geomspace(3.0, 3.0e4, 32)   # quality units per dollar
TEMPS = [0.01, 0.03, 0.1]               # softmax temperatures (quality units)
CERT_METHODS = list(certs.METHODS)
# Certificates added after the original runs draw tie-breaks from their own random stream, so every other
# method sees exactly the random numbers it saw before (results reproduce bit for bit).
ADDED_METHODS = ("norm_betting",)
ROWCHUNK = 32                           # candidate rows per certs call (memory)
LOG = print


def log(*a):
    LOG(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------
# data, features, predictors
# ----------------------------------------------------------------------------
def load_data(data_dir=DATA_DIR):
    path = os.path.join(data_dir, DATA_FILE)
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        hf_hub_download("withmartian/routerbench", DATA_FILE,
                        repo_type="dataset", local_dir=data_dir)
    df = pd.read_pickle(path)
    audit = {"rows_raw": int(len(df)), "evals_raw": int(df.eval_name.nunique())}
    df = df.copy()
    df["prompt_str"] = df.prompt.astype(str)
    ok = np.ones(len(df), bool)
    for m in MODELS:
        q = pd.to_numeric(df[m], errors="coerce")
        c = pd.to_numeric(df[m + "|total_cost"], errors="coerce")
        ok &= (q.notna() & q.between(0, 1) & c.notna() & (c > 0)).values
    audit["rows_missing_or_invalid"] = int((~ok).sum())
    df = df[ok]
    # identical prompt text may appear twice (different sample_id); keep first so
    # that the dev/eval split is by unique prompt and no prompt leaks across
    dup = df.prompt_str.duplicated(keep="first")
    audit["rows_duplicate_prompt_dropped"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    audit["unique_prompts_kept"] = int(len(df))
    Q = np.stack([df[m].astype(float).values for m in MODELS], 1)
    C = np.stack([df[m + "|total_cost"].astype(float).values for m in MODELS], 1)
    return df, Q, C, audit


def make_features(df, dev):
    """Pre-generation context only: eval-task one-hot + log prompt length (bins + z)."""
    ev = df.eval_name.values
    evs = np.array(sorted(np.unique(ev[dev])))       # tasks seen on dev
    E = (ev[:, None] == evs[None, :]).astype(float)
    L = np.log(df.prompt_str.str.len().values.astype(float))
    edges = np.quantile(L[dev], np.linspace(0, 1, 9)[1:-1])
    B = (np.searchsorted(edges, L)[:, None] == np.arange(8)[None, :]).astype(float)
    Lz = (L - L[dev].mean()) / L[dev].std()
    return np.hstack([E, B, Lz[:, None], np.ones((len(df), 1))])


def ridge(X, y, lam=1.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


# ----------------------------------------------------------------------------
# policies (P[j, x, a])
# ----------------------------------------------------------------------------
def family_specs():
    """Nested ordering: first 8 = deterministic, every 4th kappa; first 32 = all
    deterministic kappas; 128 = + softmax(T) variants for T in TEMPS."""
    k8 = list(range(0, 32, 4))
    specs = [(KAPPAS[i], 0.0) for i in k8]
    specs += [(KAPPAS[i], 0.0) for i in range(32) if i not in k8]
    specs += [(KAPPAS[i], T) for T in TEMPS for i in range(32)]
    assert len(specs) == 128
    return specs


def policy(qhat, chat, kap, T):
    s = qhat - kap * chat
    N = s.shape[0]
    if T == 0.0:
        P = np.zeros((N, M))
        P[np.arange(N), s.argmax(1)] = 1.0
        return P
    z = np.exp((s - s.max(1, keepdims=True)) / T)
    return z / z.sum(1, keepdims=True)


def logging_policy(base, eps):
    P0 = np.full((len(base), M), eps / M)
    P0[np.arange(len(base)), base] += 1 - eps
    return P0


# ----------------------------------------------------------------------------
# certificate helpers
# ----------------------------------------------------------------------------
def bound_rows(meth, w, exc, W, C):
    """certs.METHODS[meth] applied row-chunk-wise (all certs reduce over the last
    axis only, so chunking candidate rows is exact)."""
    f = certs.METHODS[meth]
    out = np.empty(w.shape[0])
    for s in range(0, w.shape[0], ROWCHUNK):
        out[s:s + ROWCHUNK] = f(w[s:s + ROWCHUNK], exc, W, C)
    return out


def _multiC_rows(meth, w, exc, W, Cs):
    """Bounds for several complexity values at once: (len(Cs), rows).  Uses
    numpy broadcasting of C inside the certs functions (no math re-implemented);
    cdf_pacbayes collapses its radius with .min() over all axes, so it is called
    once per C.  Checked against scalar calls by selftest_multiC()."""
    f = certs.METHODS[meth]
    Cs = np.asarray(Cs, float)
    if meth == "exceedance":
        return f(w, exc, W, Cs[:, None, None])
    if meth in ("cdf_empbernstein", "exc_empbernstein", "exc_betting", "norm_betting"):
        return f(w, exc, W, Cs[:, None])
    return np.stack([f(w, exc, W, c) for c in Cs])


def multi_bounds(meth, w, exc, W, Cs):
    out = np.empty((len(Cs), w.shape[0]))
    for s in range(0, w.shape[0], ROWCHUNK):
        out[:, s:s + ROWCHUNK] = _multiC_rows(meth, w[s:s + ROWCHUNK], exc, W, Cs)
    return out


def selftest_multiC():
    rng = np.random.default_rng(0)
    for n in (50, 3000):
        w = rng.choice([0.0, 1.25, 5.0, 25.0], size=(70, n)) * rng.random((70, n)) ** 0.3
        exc = (rng.random(n) < 0.2).astype(float)
        Cs = [certs.complexity(k, DELTA) for k in (1, 8, 128)]
        for meth in CERT_METHODS:
            mb = multi_bounds(meth, w, exc, 25.0, Cs)
            for i, c in enumerate(Cs):
                ref = certs.METHODS[meth](w, exc, 25.0, c)
                assert np.allclose(mb[i], ref, rtol=1e-12, atol=1e-12), meth


def select_util(bounds, p, util, rng):
    return int(certs.select_and_certify(bounds, p, utility=util, rng=rng))


def select_min(bounds, p, rng):
    return int(certs.select_and_certify(bounds, p, utility=None, rng=rng))


def wald_ucb(w, exc):
    X = w * exc
    return X.mean(-1) + 1.6449 * X.std(-1, ddof=1) / np.sqrt(w.shape[-1])


# ----------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------
S = {}  # read-only state shared with forked workers


def draw(n, eps_i, rep, tag_id, seed):
    """Common random numbers: contexts and uniforms depend on (seed, tag, n, rep)
    only, so all methods, K, p and eps see the same prompts."""
    P0 = S["P0s"][eps_i]
    rx = np.random.default_rng([seed, tag_id, n, rep, 0])
    ru = np.random.default_rng([seed, tag_id, n, rep, 1])
    idx = rx.integers(0, S["Nev"], n)
    u = ru.random(n)
    A = np.minimum((u[:, None] > np.cumsum(P0[idx], 1)).sum(1), M - 1)
    return idx, A, 1.0 / P0[idx, A]


def run_main(job):
    eps_i, n, r0, r1, seed = job
    P, Cev, Qev = S["P"], S["Cev"], S["Qev"]
    Ks, taus, ps = S["Ks"], S["taus"], S["ps"]
    W = S["W"][eps_i]
    rows = []
    for rep in range(r0, r1):
        idx, A, inv = draw(n, eps_i, rep, 0, seed)
        c, q = Cev[idx, A], Qev[idx, A]
        w = P[:, idx, A]; w *= inv[None, :]               # (128, n) importance weights
        rng = np.random.default_rng([seed, 99, n, rep, eps_i])  # tie-breaking
        rng_added = np.random.default_rng([seed, 97, n, rep, eps_i])
        Vhat = w @ q / n                                   # IW quality estimates
        h = n // 2
        w1, w2 = w[:, :h], w[:, h:]
        V1 = w1 @ q[:h] / h
        for p in ps:
            exc = (c > taus[p]).astype(float)
            G1 = w1 @ exc[:h] / h
            # all bounds for all K (nested prefixes) and the naive C in one pass;
            # W[K] is the same for every K here (asserted in setup)
            Cs = [certs.complexity(K, DELTA) for K in Ks] + [certs.complexity(1, DELTA)]
            Bm = {meth: multi_bounds(meth, w, exc, W[Ks[-1]], Cs) for meth in CERT_METHODS}
            for iK, K in enumerate(Ks):
                WK = W[K]
                for meth in CERT_METHODS:
                    b = Bm[meth][iK, :K]
                    j = select_util(b, p, Vhat[:K], rng_added if meth in ADDED_METHODS else rng)
                    rows.append((meth, p, eps_i, K, n, rep, j, b[j] if j >= 0 else np.nan))
                b = Bm["exceedance"][-1, :K]                 # naive: C = log(1/delta)
                j = select_util(b, p, Vhat[:K], rng)
                rows.append(("naive", p, eps_i, K, n, rep, j, b[j] if j >= 0 else np.nan))
                # split
                feas = G1[:K] <= 1 - p
                if feas.any():
                    js = int(np.where(feas, V1[:K], -np.inf).argmax())
                else:
                    js = int(G1[:K].argmin())
                b2 = float(certs.exc_pb(w2[js], exc[h:], WK, certs.complexity(1, DELTA)))
                rows.append(("split", p, eps_i, K, n, rep, js if b2 <= 1 - p else -1,
                             b2 if b2 <= 1 - p else np.nan))
    return rows


def stress_w(st, p, s, e, idx, A, inv):
    """Importance weights (rows s:e of the stress family) for logged (idx, A)."""
    if "act" in st:                          # deterministic family stored as actions
        w = (st["act"][p][s:e][:, idx] == A[None, :]).astype(float)
    else:
        w = st["P"][p][s:e][:, idx, A]
    w *= inv[None, :]
    return w


def run_stress(job):
    tag, eps_i, n, r0, r1, seed = job
    st = S["stress"][tag]
    Cev, ps = S["Cev"], S["ps"]
    Ks = S["stressKs"][tag]
    Kmax = max(Ks)
    W = S["Wstress"][tag][eps_i]
    tag_id = {"stress_interp": 1, "stress_natural": 2}[tag]
    labs = [(m, m, False) for m in CERT_METHODS] + [("naive_" + m, m, True) for m in CERT_METHODS]
    rows = []
    for rep in range(r0, r1):
        idx, A, inv = draw(n, eps_i, rep, tag_id, seed)
        c = Cev[idx, A]
        rng = np.random.default_rng([seed, 98, tag_id, n, rep, eps_i])
        rng_added = np.random.default_rng([seed, 96, tag_id, n, rep, eps_i])
        h = n // 2
        for p in ps:
            exc = (c > st["tau"][p]).astype(float)
            B = {(lab, K): np.empty(K) for lab, _, _ in labs for K in Ks}
            G1 = np.empty(Kmax); wald = np.empty(Kmax)
            for s in range(0, Kmax, 128):        # candidate row chunks (memory)
                e = min(s + 128, Kmax)
                w = stress_w(st, p, s, e, idx, A, inv)
                G1[s:e] = w[:, :h] @ exc[:h] / h
                wald[s:e] = wald_ucb(w, exc)
                Kin = [K for K in Ks if K > s]
                for Wv in sorted({W[p][K] for K in Kin}):   # W may differ by K
                    Kg = [K for K in Kin if W[p][K] == Wv]
                    Cg = [certs.complexity(K, DELTA) for K in Kg] + [certs.complexity(1, DELTA)]
                    for meth in CERT_METHODS:
                        mb = multi_bounds(meth, w, exc, Wv, Cg)
                        for t, K in enumerate(Kg):
                            e2 = min(e, K)
                            B[(meth, K)][s:e2] = mb[t, :e2 - s]
                            B[("naive_" + meth, K)][s:e2] = mb[-1, :e2 - s]
            for K in Ks:
                for lab, meth, _ in labs:
                    b = B[(lab, K)]
                    j = select_min(b, p, rng_added if meth in ADDED_METHODS else rng)
                    rows.append((tag, lab, p, eps_i, K, n, rep, j, b[j] if j >= 0 else np.nan))
                b = wald[:K]
                j = select_min(b, p, rng)
                rows.append((tag, "naive_wald", p, eps_i, K, n, rep, j, b[j] if j >= 0 else np.nan))
                js = int(G1[:K].argmin())
                wj = stress_w(st, p, js, js + 1, idx[h:], A[h:], inv[h:])[0]
                b2 = float(certs.exc_pb(wj, exc[h:], W[p][K], certs.complexity(1, DELTA)))
                rows.append((tag, "split", p, eps_i, K, n, rep, js if b2 <= 1 - p else -1,
                             b2 if b2 <= 1 - p else np.nan))
    return rows


def run_posterior(job):
    eps_i, n, r0, r1, seed = job
    po = S["post"]
    Cev, Qev, ps = S["Cev"], S["Qev"], S["ps"]
    Wp = S["Wpost"][eps_i]
    rows = []
    for rep in range(r0, r1):
        idx, A, inv = draw(n, eps_i, rep, 0, seed)        # same logs as the main study
        c, q = Cev[idx, A], Qev[idx, A]
        w = po["P"][:, idx, A] * inv[None, :]              # (n_means, n) pooled-policy weights
        Vhat = w @ q / n
        rng = np.random.default_rng([seed, 97, n, rep, eps_i])
        for p in ps:
            exc = (c > S["taus"][p]).astype(float)
            for meth in CERT_METHODS:
                # Each sampled mixture is fixed before the certification log.
                # All methods pay for this finite list; Gaussian KL is not a
                # valid complexity charge for an unprotected MC approximation.
                n_pooled = po["P"].shape[0]
                Cm = [certs.complexity(n_pooled, DELTA)] * n_pooled
                b = np.array([certs.METHODS[meth](w[m], exc, Wp, Cm[m])
                              for m in range(len(Cm))])
                j = select_util(b, p, Vhat, rng)
                rows.append(("posterior_" + meth, p, eps_i, -1, n, rep, j,
                             b[j] if j >= 0 else np.nan))
    return rows


def dispatch(job):
    kind = job[0]
    if kind == "main":
        return run_main(job[1:])
    if kind == "stress":
        return run_stress(job[1:])
    return run_posterior(job[1:])


# ----------------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    k = np.asarray(k, float); n = np.asarray(n, float)
    ph = k / n
    den = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / den
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return np.clip(ctr - hw, 0, 1), np.clip(ctr + hw, 0, 1)


def summarize(raw, keys):
    g = raw.groupby(keys)
    s = g.agg(reps=("certified", "size"), cert=("certified", "sum"),
              false=("false_cert", "sum"), G_sel=("G_sel", "mean"),
              V_sel=("V_sel", "mean"), cost_sel=("cost_sel", "mean"),
              slack=("slack", "mean"), slack_min=("slack", "min"),
              V_deploy=("V_deploy", "mean"), V_deploy_sd=("V_deploy", "std"),
              cost_deploy=("cost_deploy", "mean")).reset_index()
    s["cert_rate"] = s.cert / s.reps
    s["cert_lo"], s["cert_hi"] = wilson(s.cert, s.reps)
    s["false_rate"] = s.false / s.reps
    s["false_lo"], s["false_hi"] = wilson(s.false, s.reps)
    return s


def attach_truth(raw, Gtab, Vtab, Ctab, fb):
    """Gtab[p] / Vtab / Ctab: true values per candidate index."""
    G = np.full(len(raw), np.nan); V = G.copy(); Cc = G.copy()
    for p in raw.p.unique():
        m = (raw.p == p).values
        jj = raw.j.values[m]
        ok = jj >= 0
        g = np.full(m.sum(), np.nan); v = g.copy(); cc = g.copy()
        g[ok] = Gtab[p][jj[ok]]; v[ok] = Vtab[p][jj[ok]] if isinstance(Vtab, dict) else Vtab[jj[ok]]
        cc[ok] = Ctab[p][jj[ok]] if isinstance(Ctab, dict) else Ctab[jj[ok]]
        G[m], V[m], Cc[m] = g, v, cc
    raw["G_sel"], raw["V_sel"], raw["cost_sel"] = G, V, Cc
    raw["certified"] = raw.j >= 0
    raw["false_cert"] = raw.certified & (raw.G_sel > 1 - raw.p + 1e-12)
    raw["slack"] = raw.bound - raw.G_sel
    if fb is not None:
        raw["V_deploy"] = np.where(raw.certified, raw.V_sel, fb["V_true"])
        raw["cost_deploy"] = np.where(raw.certified, raw.cost_sel, fb["cost_true"])
    else:
        raw["V_deploy"] = raw.V_sel
        raw["cost_deploy"] = raw.cost_sel
    return raw


def cross_n(ns, rate, level):
    """Smallest n with cert rate >= level, log-linear interpolation; nan if never."""
    ns = np.asarray(ns, float); rate = np.asarray(rate, float)
    for i in range(len(rate)):
        if rate[i] >= level:
            if i == 0:
                return ns[0], True
            t = (level - rate[i - 1]) / (rate[i] - rate[i - 1])
            return float(np.exp(np.log(ns[i - 1]) + t * (np.log(ns[i]) - np.log(ns[i - 1])))), False
    return np.nan, False


def headline(summ):
    out = []
    for key, sub in summ.groupby(["method", "p", "eps", "K"]):
        sub = sub.sort_values("n")
        n50, l50 = cross_n(sub.n, sub.cert_rate, 0.5)
        n90, l90 = cross_n(sub.n, sub.cert_rate, 0.9)
        out.append(dict(zip(["method", "p", "eps", "K"], key), n50=n50, n50_at_min_n=l50,
                        n90=n90, n90_at_min_n=l90, max_false_rate=sub.false_rate.max()))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------------
def choose_tau(Pdev, Cdev, Vdev, p, grid):
    """Dev-only rule (per p): smallest tau on the grid such that
       (a) >= 1/2 of the K=32 deterministic candidates have G_dev <= (1-p)/2, and
       (b) the quality-greedy K=32 candidate (max dev quality) has G_dev > 1-p.
    Then cheap routers are comfortably feasible, quality-greedy ones are not, and
    the best feasible router sits near the boundary (a genuine trade-off)."""
    g = int(np.argmax(Vdev[:32]))
    for tau in grid:
        Gd = (Pdev[:32] * (Cdev > tau)[None]).sum(2).mean(1)
        if (Gd <= (1 - p) / 2).mean() >= 0.5 and Gd[g] > 1 - p:
            return float(tau), Gd
    raise RuntimeError("no tau satisfies the dev rule for p=%s" % p)


def build_stress_interp(alo, ahi, Cev, Qev, tau, p, n_cand, band, rng):
    """n_cand deterministic routers a_j(x) = r_hi(x) if x in S_j else r_lo(x), with
    S_j a uniformly random subset of the eval pool (fixed by seed) whose size is
    calibrated on the eval population so that G_j = (1-p) * U(band) exactly
    (up to 1/N_eval), i.e. every candidate is UNSAFE but only slightly.  r_lo = the
    frozen cheap base router, r_hi = the frozen quality-greedy router (kappa_min).
    Because S_j are independent random subsets, the IW exceedance estimates of
    different candidates are only weakly correlated (overlap ~ alpha^2), which is
    the regime where ignoring the search (winner's curse) bites hardest."""
    E = (Cev > tau)
    N = len(Cev)
    elo = E[np.arange(N), alo].astype(float); ehi = E[np.arange(N), ahi].astype(float)
    assert elo.mean() < (1 - p) * band[0] and ehi.mean() > (1 - p) * band[1]
    targets = (1 - p) * rng.uniform(band[0], band[1], n_cand)
    act = np.empty((n_cand, N), np.int8)
    for k in range(n_cand):
        order = np.argsort(rng.random(N))
        cum = elo.mean() + np.cumsum((ehi - elo)[order]) / N
        m = int(np.argmax(cum >= targets[k])) + 1          # smallest prefix with G >= target
        sel = np.zeros(N, bool); sel[order[:m]] = True
        act[k] = np.where(sel, ahi, alo)
    ar = np.arange(N)[None, :]
    G = E[ar, act].mean(1)
    assert (G > 1 - p).all()
    return act, G, Qev[ar, act].mean(1), Cev[ar, act].mean(1), dict(
        G_lo=float(elo.mean()), G_hi=float(ehi.mean()),
        alpha_mean=float((act == ahi[None]).mean()))


def setup(args):
    log("loading data")
    df, Q, C, audit = load_data(args.data_dir)
    N = len(df)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    ndev = int(round(0.3 * N))
    dev = np.zeros(N, bool); dev[perm[:ndev]] = True
    ev = ~dev
    audit.update(dev_prompts=int(dev.sum()), eval_prompts=int(ev.sum()),
                 eval_tasks_unseen_on_dev=int(len(set(df.eval_name[ev]) - set(df.eval_name[dev]))))
    X = make_features(df, dev)
    qhat = np.zeros((N, M)); chat = np.zeros((N, M)); fit = {}
    for a in range(M):
        qhat[:, a] = np.clip(X @ ridge(X[dev], Q[dev, a]), 0, 1)
        chat[:, a] = np.exp(X @ ridge(X[dev], np.log(C[dev, a])))
        fit[SHORT[a]] = {
            "quality_R2_eval": float(1 - ((qhat[ev, a] - Q[ev, a]) ** 2).mean() / Q[ev, a].var()),
            "logcost_R2_eval": float(1 - ((np.log(chat[ev, a]) - np.log(C[ev, a])) ** 2).mean()
                                     / np.log(C[ev, a]).var())}
    specs = family_specs()
    P = np.stack([policy(qhat, chat, k, T) for k, T in specs])      # (128, N, M)
    base = (qhat - KAPPAS[-1] * chat).argmax(1)                     # cheap-leaning base router
    Pdev, Pev = P[:, dev], P[:, ev]
    Vdev = (Pdev * Q[dev][None]).sum(2).mean(1)
    grid = np.geomspace(1e-5, 1e-2, 241)
    taus, tau_info = {}, {}
    for p in args.ps:
        tau, Gd = choose_tau(Pdev, C[dev], Vdev, p, grid)
        taus[p] = tau
        feas = Gd <= 1 - p
        tau_info[str(p)] = {"tau": tau, "dev_frac_feasible_K32": float(feas.mean()),
                            "dev_best_feasible_quality_K32": float(Vdev[:32][feas].max()),
                            "dev_greedy_quality_K32": float(Vdev[:32].max()),
                            "dev_G_greedy": float(Gd[int(np.argmax(Vdev[:32]))])}
    log("taus", taus)
    Cev, Qev = C[ev], Q[ev]
    Nev = int(ev.sum())
    # ground truth on the eval population
    gt = pd.DataFrame({"j": np.arange(128), "kappa": [s[0] for s in specs],
                       "temp": [s[1] for s in specs], "V_dev": Vdev,
                       "V_true": (Pev * Qev[None]).sum(2).mean(1),
                       "cost_true": (Pev * Cev[None]).sum(2).mean(1)})
    for p in args.ps:
        gt[f"G_true_p{p}"] = (Pev * (Cev > taus[p])[None]).sum(2).mean(1)
    mix = Pev.mean(1)
    for a in range(M):
        gt["frac_" + SHORT[a]] = mix[:, a]
    bev = base[ev]
    fb = {"router": "logging base router (deterministic argmax, kappa=%g)" % KAPPAS[-1],
          "V_true": float(Qev[np.arange(Nev), bev].mean()),
          "cost_true": float(Cev[np.arange(Nev), bev].mean()),
          "frac": {SHORT[a]: float((bev == a).mean()) for a in range(M)}}
    for p in args.ps:
        fb[f"G_true_p{p}"] = float((Cev[np.arange(Nev), bev] > taus[p]).mean())
    oracle = {}
    for p in args.ps:
        oracle[str(p)] = {}
        for K in args.Ks:
            g = gt.iloc[:K]
            f = g[g[f"G_true_p{p}"] <= 1 - p]
            oracle[str(p)][str(K)] = {"n_feasible": int(len(f)),
                                      "best_feasible_V": float(f.V_true.max()) if len(f) else None,
                                      "best_feasible_j": int(f.V_true.idxmax()) if len(f) else None,
                                      "unconstrained_best_V": float(g.V_true.max()),
                                      "n_within_10pct_of_boundary": int(
                                          ((g[f"G_true_p{p}"] > 0.9 * (1 - p)) &
                                           (g[f"G_true_p{p}"] <= 1.1 * (1 - p))).sum())}
    P0s, Wtab = [], []
    for e in args.eps:
        P0 = logging_policy(bev, e)
        P0s.append(P0)
        r = Pev / P0[None]
        Wtab.append({K: float(r[:K].max()) for K in args.Ks})
        assert len(set(Wtab[-1].values())) == 1, "main loop assumes W equal across nested K"
        # tail- and bulk-effective overlaps per candidate (Theorem 3; appendix "Predicted versus
        # observed log sizes"): V_tau = E_pi0[w^2 1{Y>tau}], V_F = E_pi0[w^2 1{Y<=tau}]
        w2 = Pev ** 2 / P0[None]
        for p in args.ps:
            ex = (Cev > taus[p]).astype(float)
            gt[f"V_tau_p{p}_eps{e}"] = (w2 * ex[None]).sum(2).mean(1)
            gt[f"V_F_p{p}_eps{e}"] = (w2 * (1.0 - ex)[None]).sum(2).mean(1)
    log("W", {e: Wtab[i] for i, e in enumerate(args.eps)})

    # ---- stress families ----
    srng = np.random.default_rng([args.seed, 7])
    stress, Wst, sinfo = {}, {}, {}
    band = args.stress_band
    alo = base[ev]                                   # cheap base router
    ahi = Pev[int(np.argmin([s_[0] if s_[1] == 0 else np.inf for s_ in specs]))].argmax(1)
    st = {"act": {}, "G": {}, "V": {}, "cost": {}, "tau": {}}
    nst = max(args.stress_Ks)
    for p in args.ps:
        act, Gs, Vs, Cs, info = build_stress_interp(alo, ahi, Cev, Qev, taus[p], p, nst, band, srng)
        st["act"][p] = act; st["G"][p] = Gs; st["V"][p] = Vs; st["cost"][p] = Cs
        st["tau"][p] = taus[p]
        info.update(tau=taus[p], n_cand=nst, band=band, G_min=float(Gs.min()),
                    G_max=float(Gs.max()), n_unsafe=int((Gs > 1 - p).sum()))
        sinfo.setdefault("stress_interp", {})[str(p)] = info
    stress["stress_interp"] = st
    # natural: real family, tau re-chosen ON THE EVAL POPULATION so that the
    # closest candidate sits just above the boundary: min_j G_j >= (1-p)*1.005
    st = {"P": {}, "G": {}, "V": {}, "cost": {}, "tau": {}}
    cand = np.unique(Cev)
    for p in args.ps:
        target = (1 - p) * 1.005
        def gmin(t):
            return (Pev * (Cev > t)[None]).sum(2).mean(1).min()
        lo, hi = 0, len(cand) - 1
        while lo < hi:                       # largest tau with min_j G_j >= target
            mid = (lo + hi + 1) // 2
            if gmin(cand[mid]) >= target:
                lo = mid
            else:
                hi = mid - 1
        t = float(cand[lo])
        Gs = (Pev * (Cev > t)[None]).sum(2).mean(1)
        st["P"][p] = Pev; st["G"][p] = Gs; st["V"][p] = gt.V_true.values
        st["cost"][p] = gt.cost_true.values; st["tau"][p] = t
        sinfo.setdefault("stress_natural", {})[str(p)] = {
            "tau": t, "G_min": float(Gs.min()), "G_max": float(Gs.max()),
            "n_unsafe": int((Gs > 1 - p).sum()),
            "n_in_band": int(((Gs > 1 - p) & (Gs <= band[1] * (1 - p))).sum()),
            "n_within_10pct": int(((Gs > 1 - p) & (Gs <= 1.1 * (1 - p))).sum())}
    stress["stress_natural"] = st
    stressKs = {"stress_interp": sorted(args.stress_Ks),
                "stress_natural": sorted(k for k in args.stress_Ks if k <= 128) or [128]}
    for tag, st in stress.items():
        Wst[tag] = []
        for i, e in enumerate(args.eps):
            inv0 = 1.0 / P0s[i]
            d = {}
            for p in args.ps:
                if "act" in st:
                    rowmax = inv0[np.arange(Nev)[None, :], st["act"][p]].max(1)
                else:
                    rowmax = (st["P"][p] * inv0[None]).max((1, 2))
                cm = np.maximum.accumulate(rowmax)
                d[p] = {K: float(cm[K - 1]) for K in stressKs[tag]}
            Wst[tag].append(d)
    log("stress", json.dumps(sinfo))

    # ---- optional finite list of sampled mixtures of linear-argmax routers ----
    post, Wpost, pinfo = None, None, None
    if args.posterior:
        post, pinfo = build_posterior(qhat, chat, dev, ev, Q, C, taus, args)
        Wpost = [float((post["P"] / P0[None]).max()) for P0 in P0s]

    S.update(P=Pev, Cev=Cev, Qev=Qev, Nev=Nev, P0s=P0s, W=Wtab, Ks=args.Ks, ps=args.ps,
             taus=taus, stress=stress, Wstress=Wst, stressKs=stressKs, post=post, Wpost=Wpost)
    meta = {"audit": audit, "predictor_fit": fit, "models": MODELS, "short": SHORT,
            "taus": {str(k): v for k, v in taus.items()}, "tau_info": tau_info,
            "tau_rule": choose_tau.__doc__.strip(),
            "W": {str(e): Wtab[i] for i, e in enumerate(args.eps)},
            "W_stress": {tag: {str(e): {str(p): v for p, v in Wst[tag][i].items()}
                               for i, e in enumerate(args.eps)} for tag in Wst},
            "stress_info": sinfo, "stress_Ks": stressKs, "fallback": fb, "oracle": oracle,
            "eps": args.eps, "Ks": args.Ks, "ns": args.ns, "stress_ns": args.stress_ns,
            "ps": args.ps, "reps": args.reps, "stress_reps": args.stress_reps,
            "seed": args.seed, "delta": DELTA, "n_lam": certs.N_LAM,
            "kappas": KAPPAS.tolist(), "temps": TEMPS, "family_order": [list(s) for s in specs],
            "posterior": pinfo, "W_posterior": Wpost}
    return gt, meta, stress


def build_posterior(qhat, chat, dev, ev, Q, C, taus, args):
    """Linear-softmax router family pi_theta(a|x) = argmax_a theta . psi(x,a) with
    psi(x,a) = (qhat_a(x)/0.1, -kappa0*chat_a(x)/0.1, e_a) (d = 2+M), so theta0 =
    (1, 1, 0) is exactly the kappa0 argmax router; kappa0 = best dev quality with dev
    G <= (1-p)/2 on the kappa grid (strictest p).  Prior N(theta0, s^2 I), s fixed.
    Posteriors N(mu_m, s^2 I) over a predeclared grid of means mu_m (cost-coefficient
    multipliers x per-arm bias shifts of +0.5); KL_m = ||mu_m - theta0||^2/(2 s^2).
    Each candidate is the finite mixture of D fixed draws (common random
    numbers across m), fixed independently of the certification log. This
    sampled mixture is the deployed policy, not an exact Gaussian posterior.
    All certificates therefore pay log(n_means/delta) for the finite list.
    KL_m is retained only as a description of the generating Gaussians; it is
    not used for certification and cannot replace a bound on integration error."""
    D, s = args.post_draws, args.post_sigma
    cbar = np.median(chat[dev])
    N = qhat.shape[0]
    Pdev_mask = dev
    # theta0: on dev only, use p = max(ps) (strictest) to pick kappa0
    p0 = max(args.ps)
    best = None
    for kap in KAPPAS:
        a = (qhat - kap * chat).argmax(1)
        Gd = (C[dev, :][np.arange(dev.sum()), a[dev]] > taus[p0]).mean()
        Vd = Q[dev, :][np.arange(dev.sum()), a[dev]].mean()
        if Gd <= (1 - p0) / 2 and (best is None or Vd > best[1]):
            best = (kap, Vd)
    kap0 = best[0]
    theta0 = np.concatenate([[1.0, 1.0], np.zeros(M)])
    mults = np.geomspace(0.25, 4.0, 9)                  # cost-coefficient multiplier
    shifts = [np.zeros(M)] + [0.5 * np.eye(M)[a] for a in (1, 2, 3, 4)]
    means = [np.concatenate([[1.0, mlt], sh]) for mlt in mults for sh in shifts]
    means = np.array(means)
    KL = ((means - theta0[None]) ** 2).sum(1) / (2 * s * s)
    zr = np.random.default_rng([args.seed, 5]).standard_normal((D, 2 + M))
    Qe, Ce = qhat[ev] / 0.1, kap0 * chat[ev] / 0.1
    Nev = int(ev.sum())
    P = np.zeros((len(means), Nev, M))
    for m, mu in enumerate(means):
        for d in range(D):
            th = mu + s * zr[d]
            sc = th[0] * Qe - th[1] * Ce + th[2:][None]
            P[m, np.arange(Nev), sc.argmax(1)] += 1.0 / D
    Cev, Qev = C[ev], Q[ev]
    info = {"kappa0": float(kap0), "cbar": float(cbar), "sigma": s, "draws": D,
            "n_means": len(means), "KL": KL.tolist(),
            "certificate_scope": "fixed sampled mixtures; all methods use log(n_means/delta)",
            "V_true": (P * Qev[None]).sum(2).mean(1).tolist(),
            "G_true": {str(p): (P * (Cev > taus[p])[None]).sum(2).mean(1).tolist() for p in args.ps},
            "cost_true": (P * Cev[None]).sum(2).mean(1).tolist(),
            "note": build_posterior.__doc__.strip()}
    log("posterior: kappa0=%.3g, KL range [%.2f, %.2f]" % (kap0, KL.min(), KL.max()))
    return {"P": P, "KL": KL}, info


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_list(s, f):
    return [f(v) for v in s.split(",") if v]


def run_jobs(jobs, workers, label):
    t0 = time.time()
    rows = []
    jobs = sorted(jobs, key=lambda j: -j[3] * (j[5] - j[4]) if j[0] == "stress"
                  else -j[2] * (j[4] - j[3]))
    if workers <= 1:
        it = map(dispatch, jobs)
        pool = None
    else:
        pool = Pool(workers)
        it = pool.imap_unordered(dispatch, jobs, chunksize=1)
    for i, r in enumerate(it):
        rows += r
        if (i + 1) % max(1, len(jobs) // 20) == 0 or i + 1 == len(jobs):
            log("%s: %d/%d jobs, %.1fs" % (label, i + 1, len(jobs), time.time() - t0))
    if pool is not None:
        pool.close(); pool.join()
    return rows, time.time() - t0


_MAIN_NS = ",".join(str(int(round(v))) for v in np.geomspace(500, 300000, 30))
PRESETS = {
    "main": dict(reps=300, ns=_MAIN_NS, ps="0.9,0.95,0.99", no_stress=True,
                 out=os.path.join(RESULTS, "main")),
    "stress": dict(reps=25, ns="500", stress_reps=500, ps="0.9,0.95,0.99",
                   out=os.path.join(RESULTS, "stress")),
}


def main():
    ap = argparse.ArgumentParser(description="RouterBench replay (see module docstring).")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="arguments used for the shipped results (explicit flags override)")
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="directory holding (or receiving) routerbench_0shot.pkl")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--stress-reps", type=int, default=None, help="default: --reps")
    ap.add_argument("--ns", type=str, default=",".join(str(int(round(v))) for v in
                                                       np.geomspace(500, 300000, 12)))
    ap.add_argument("--stress-ns", type=str, default="2000,5000,10000,20000,50000")
    ap.add_argument("--stress-Ks", type=str, default="32,128,1024",
                    help="interp family sizes (natural family uses those <= 128)")
    ap.add_argument("--stress-band", type=str, default="1.0,1.02",
                    help="interp candidates have true G in (1-p)*(lo, hi]")
    ap.add_argument("--Ks", type=str, default="8,32,128")
    ap.add_argument("--eps", type=str, default="1.0,0.5,0.2")
    ap.add_argument("--ps", type=str, default="0.9,0.95")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--chunk", type=int, default=25, help="reps per job")
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS, "custom"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-stress", action="store_true")
    ap.add_argument("--posterior", action="store_true")
    ap.add_argument("--post-draws", type=int, default=500)
    ap.add_argument("--post-sigma", type=float, default=0.25)
    pre, _ = ap.parse_known_args()
    if pre.preset:
        ap.set_defaults(**PRESETS[pre.preset])
    args = ap.parse_args()
    if args.smoke:
        args.reps, args.ns, args.stress_ns = 6, "500,5000,50000", "2000,20000"
        args.stress_Ks = "32,128,256"
        args.chunk = 3
        args.post_draws = min(args.post_draws, 50)
    args.stress_reps = args.reps if args.stress_reps is None else args.stress_reps
    args.ns = parse_list(args.ns, int); args.stress_ns = parse_list(args.stress_ns, int)
    args.Ks = parse_list(args.Ks, int); args.eps = parse_list(args.eps, float)
    args.ps = parse_list(args.ps, float)
    args.stress_Ks = parse_list(args.stress_Ks, int)
    args.stress_band = parse_list(args.stress_band, float)
    assert max(args.Ks) <= 128
    os.makedirs(args.out, exist_ok=True)
    global LOG
    logf = open(os.path.join(args.out, "run_log.txt"), "a")

    def _log(*a, **k):
        print(*a, **k); print(*a, file=logf, flush=True)
    LOG = _log
    log("args", vars(args))
    selftest_multiC()
    log("selftest_multiC passed (broadcast-C certs == scalar certs)")
    T0 = time.time()
    gt, meta, stress = setup(args)
    gt.to_csv(os.path.join(args.out, "ground_truth_routers.csv"), index=False)
    for tag, st in stress.items():
        d = pd.DataFrame({"j": np.arange(len(st["G"][args.ps[0]]))})
        for p in args.ps:
            d[f"G_true_p{p}"] = st["G"][p]; d[f"V_true_p{p}"] = st["V"][p]
        d.to_csv(os.path.join(args.out, f"ground_truth_{tag}.csv"), index=False)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    cols = ["method", "p", "eps_i", "K", "n", "rep", "j", "bound"]
    timings = {}

    def chunks(reps):
        return [(r, min(r + args.chunk, reps)) for r in range(0, reps, args.chunk)]

    # ---- main ----
    jobs = [("main", ei, n, a, b, args.seed) for ei in range(len(args.eps))
            for n in args.ns for a, b in chunks(args.reps)]
    rows, timings["main_s"] = run_jobs(jobs, args.workers, "main")
    raw = pd.DataFrame(rows, columns=cols)
    raw["eps"] = np.array(args.eps)[raw.eps_i]
    Gt = {p: gt[f"G_true_p{p}"].values for p in args.ps}
    raw = attach_truth(raw, Gt, gt.V_true.values, gt.cost_true.values, meta["fallback"])
    raw.drop(columns="eps_i").to_csv(os.path.join(args.out, "raw_main.csv.gz"), index=False)
    summ = summarize(raw, ["method", "p", "eps", "K", "n"])
    summ.to_csv(os.path.join(args.out, "summary_main.csv"), index=False)
    hl = headline(summ)
    hl.to_csv(os.path.join(args.out, "headline.csv"), index=False)
    log("main done: %d rows, %.1fs" % (len(raw), timings["main_s"]))

    # ---- stress ----
    if not args.no_stress:
        jobs = [("stress", tag, ei, n, a, b, args.seed) for tag in stress
                for ei in range(len(args.eps)) for n in args.stress_ns
                for a, b in chunks(args.stress_reps)]
        rows, timings["stress_s"] = run_jobs(jobs, args.workers, "stress")
        raw = pd.DataFrame(rows, columns=["tag"] + cols)
        raw["eps"] = np.array(args.eps)[raw.eps_i]
        parts = []
        for tag, st in stress.items():
            r = raw[raw.tag == tag].copy()
            parts.append(attach_truth(r, st["G"], st["V"], st["cost"], None))
        raw = pd.concat(parts)
        raw.drop(columns="eps_i").to_csv(os.path.join(args.out, "raw_stress.csv.gz"), index=False)
        ss = summarize(raw, ["tag", "method", "p", "eps", "K", "n"])
        ss.to_csv(os.path.join(args.out, "summary_stress.csv"), index=False)
        log("stress done: %d rows, %.1fs" % (len(raw), timings["stress_s"]))

    # ---- posterior ----
    if args.posterior:
        jobs = [("post", ei, n, a, b, args.seed) for ei in range(len(args.eps))
                for n in args.ns for a, b in chunks(args.reps)]
        rows, timings["posterior_s"] = run_jobs(jobs, args.workers, "posterior")
        raw = pd.DataFrame(rows, columns=cols)
        raw["eps"] = np.array(args.eps)[raw.eps_i]
        pi = meta["posterior"]
        Gt = {p: np.array(pi["G_true"][str(p)]) for p in args.ps}
        raw = attach_truth(raw, Gt, np.array(pi["V_true"]), np.array(pi["cost_true"]),
                           meta["fallback"])
        raw.drop(columns="eps_i").to_csv(os.path.join(args.out, "raw_posterior.csv.gz"), index=False)
        sp = summarize(raw, ["method", "p", "eps", "K", "n"])
        sp.to_csv(os.path.join(args.out, "summary_posterior.csv"), index=False)
        headline(sp).to_csv(os.path.join(args.out, "headline_posterior.csv"), index=False)

    timings["total_s"] = time.time() - T0
    meta["timings"] = timings
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log("all done", timings)


if __name__ == "__main__":
    main()
