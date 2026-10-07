#!/usr/bin/env python3
"""Step 3 (CPU, multiprocessing): constructed-logging replay on the outcome table, and the
deployment plan for deploy_verify.py.

Outcome Y = generated tokens per request (thinking included); constraint P(Y > tau) <= 1 - p.

Design (everything except the logged sample is frozen on dev before any replay):
  * Partition fixed in prompts.parquet (prepare_prompts.py): dev (30% of the table prompts) /
    eval (70%) / deploy (held out for deploy_verify.py), stratified by dataset.
  * Dev: per-action logistic P(correct|x) and ridge on log Y; candidate routers
        reference router (always cfg.reference_action)
        deterministic argmax_a pred_acc - kappa * pred_Y          (31 kappas)
        softmax_T(pred_acc - kappa * pred_Y)                       (3 T x 32 kappas)
    nested K = 8 (ref + 7 det) < 32 (ref + 31 det) < 128 (+96 softmax).
  * Logging pi0^eps = (1-eps) pi_base + eps/M, pi_base = constant cheap action (config
    logging_base.action) or a cheap kappa-router (logging_base.kappa).
    W_K = max over contexts (all partitions) and the first K candidates of pi/pi0.
  * tau per p chosen on dev (rule printed and saved in meta.json).
  * Replay: n eval prompts WITH replacement (n can exceed the pool: the population is the
    finite empirical eval pool), A ~ pi0, reveal only (Y, correct) of A. Seeds paired across
    methods, K and p.
  * Methods: every certificate in certs.METHODS (exceedance PAC-Bayes = Theorem 1, exceedance
    betting = appendix "The betting certificate", exceedance empirical Bernstein, CDF PAC-Bayes,
    CDF empirical Bernstein) with C = log(K/delta) and constrained selection (max IW accuracy
    among candidates with bound <= 1-p); `split` (select on half 1, certify the one selected
    router on half 2 with the exceedance bound, C = log(1/delta)); and `naive` (exceedance bound
    with C = log(1/delta) despite selecting among K: INVALID, reported for reference only).

  python certify.py --reps 1000 --procs 16                                 # needs runs/open/table/
  python certify.py --table-dir ../results/reasoning/table --reps 1000      # shipped outcome table
  python certify.py --table-dir ../results/reasoning/table --deploy-only    # deploy plan only (~1 min)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import grlib
import certs  # ../common/certs.py (path inserted by grlib)

DELTA = 0.05
TEMPS = [0.01, 0.03, 0.1]   # softmax temperatures (accuracy units)
PS = (0.90, 0.95)
METHODS = list(certs.METHODS) + ["split", "naive"]
ADDED_METHODS = ("norm_betting",)


# ----------------------------------------------------------------------------------------
# predictors and router family (dev only)
# ----------------------------------------------------------------------------------------
def fit_predictors(X, dev, T, Q):
    from sklearn.linear_model import LogisticRegression, Ridge
    N, M = X.shape[0], T.shape[1]
    pacc, pk = np.zeros((N, M)), np.zeros((N, M))
    diag = {}
    for a in range(M):
        y = Q[dev, a]
        if y.min() == y.max():
            pacc[:, a] = y.mean()
        else:
            lr = LogisticRegression(C=1.0, max_iter=5000).fit(X[dev], y)
            pacc[:, a] = lr.predict_proba(X)[:, 1]
        lt = np.log(np.maximum(T[dev, a], 1e-12))
        rg = Ridge(alpha=1.0).fit(X[dev], lt)
        res = lt - rg.predict(X[dev])
        pk[:, a] = np.minimum(np.exp(rg.predict(X) + res.var() / 2), T[dev, a].max())
        brier0 = float(((y - y.mean()) ** 2).mean())
        brier = float(((y - pacc[dev, a]) ** 2).mean())
        diag[a] = {"dev_brier_skill": 1 - brier / max(brier0, 1e-12),
                   "dev_R2_logY": float(1 - res.var() / max(lt.var(), 1e-12))}
    pk /= np.median(T[dev])   # Y in units of the dev median => kappa is dimensionless
    return pacc, pk, diag


def router_probs(pacc, pk, kappa, temp):
    s = pacc - kappa * pk
    N, M = s.shape
    if temp == 0.0:
        P = np.zeros((N, M), np.float32)
        P[np.arange(N), s.argmax(1)] = 1.0
        return P
    z = (s - s.max(1, keepdims=True)) / temp
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def build_family(pacc, pk, ref, kappa_range):
    N, M = pacc.shape
    kd = np.geomspace(kappa_range[0], kappa_range[1], 31)
    ks = np.geomspace(kappa_range[0], kappa_range[1], 32)
    k8 = [0, 5, 10, 15, 20, 25, 30]
    specs = [dict(kind="ref", kappa=None, temp=0.0, action=int(ref))]
    specs += [dict(kind="det", kappa=float(kd[i]), temp=0.0) for i in k8]
    specs += [dict(kind="det", kappa=float(kd[i]), temp=0.0) for i in range(31) if i not in k8]
    specs += [dict(kind="soft", kappa=float(k), temp=T) for T in TEMPS for k in ks]
    assert len(specs) == 128
    P = np.zeros((128, N, M), np.float32)
    for j, s in enumerate(specs):
        if s["kind"] == "ref":
            P[j, :, ref] = 1.0
        else:
            P[j] = router_probs(pacc, pk, s["kappa"], s["temp"])
    return P, specs


def logging_policy(pacc, pk, base_spec, eps):
    """base_spec: int action index (constant cheap router) or float kappa (cheap kappa-router)."""
    N, M = pacc.shape
    if isinstance(base_spec, (int, np.integer)):
        base = np.full(N, int(base_spec))
    else:
        base = (pacc - float(base_spec) * pk).argmax(1)
    P0 = np.full((N, M), eps / M)
    P0[np.arange(N), base] += 1 - eps
    return P0, base


def mixture_quantile(Pm, Tm, p):
    """p-quantile of Y under router probs Pm (N, M) over the empirical prompt pool."""
    w = (Pm / Pm.shape[0]).ravel(); t = Tm.ravel()
    o = np.argsort(t, kind="stable")
    c = np.cumsum(w[o])
    return float(t[o][np.searchsorted(c, p - 1e-12)])


# ----------------------------------------------------------------------------------------
# replay worker
# ----------------------------------------------------------------------------------------
G = {}


def run_cell(job):
    tag, ei, n, r0, reps = job
    P, P0, T, Q = G["P"], G["P0s"][ei], G["T"], G["Q"]
    Wk, Ks, taus, seed = G["W"][ei], G["Ks"], G["taus"][tag], G["seed"]
    Nev, M = T.shape
    cum0 = np.cumsum(P0, 1)
    rows = []
    for r in range(r0, r0 + reps):
        rng = np.random.default_rng([seed, 0 if tag == "main" else 1, ei, n, r])
        idx = rng.integers(0, Nev, n)
        A = np.minimum((rng.random(n)[:, None] > cum0[idx]).sum(1), M - 1)
        y, q = T[idx, A], Q[idx, A].astype(np.float32)
        w = P[:, idx, A] * (1.0 / P0[idx, A]).astype(np.float32)[None, :]   # (128, n)
        V = (w * q).mean(-1)
        half = n // 2
        V1 = (w[:, :half] * q[:half]).mean(-1)
        for p, tau in taus.items():
            exc = (y > tau).astype(np.float32)
            for K in Ks:
                W = Wk[K]
                C = certs.complexity(K, DELTA)
                C1 = certs.complexity(1, DELTA)
                wK = w[:K]
                srng = np.random.default_rng([seed, 7, ei, n, r, K, int(p * 100)])
                # certificates added after the original runs use their own tie-breaking stream, so the
                # other methods' results reproduce bit for bit
                srng_added = np.random.default_rng([seed, 8, ei, n, r, K, int(p * 100)])
                for m, f in certs.METHODS.items():
                    b = f(wK, exc, W, C)
                    j = int(certs.select_and_certify(b, p, utility=V[:K],
                                                     rng=srng_added if m in ADDED_METHODS else srng))
                    rows.append((tag, ei, K, n, p, r, m, j, float(b[j]) if j >= 0 else np.nan,
                                 float(V[j]) if j >= 0 else np.nan))
                b0 = certs.exc_pb(wK, exc, W, C1)
                j = int(certs.select_and_certify(b0, p, utility=V[:K], rng=srng))
                rows.append((tag, ei, K, n, p, r, "naive", j, float(b0[j]) if j >= 0 else np.nan,
                             float(V[j]) if j >= 0 else np.nan))
                # split: screen + select on half 1, certify the single selected router on half 2
                b1 = certs.exc_pb(wK[:, :half], exc[:half], W, C1)
                js = int(certs.select_and_certify(b1, p, utility=V1[:K], rng=srng))
                if js < 0:
                    js = int(b1.argmin())
                b2 = float(certs.exc_pb(wK[js, half:], exc[half:], W, C1))
                ok = b2 <= 1 - p
                rows.append((tag, ei, K, n, p, r, "split", js if ok else -1, b2 if ok else np.nan,
                             float(V[js]) if ok else np.nan))
    return rows


# ----------------------------------------------------------------------------------------
def summarize(raw):
    g = raw.groupby(["tag", "eps", "K", "n", "p", "method"])
    s = g.agg(reps=("certified", "size"), cert=("certified", "sum"), false=("false_cert", "sum"),
              G_sel=("G_sel", "mean"), acc_sel=("acc_sel", "mean"), Ymean_sel=("Ymean_sel", "mean"),
              slack=("slack", "mean"), acc_deploy=("acc_deploy", "mean"),
              acc_deploy_sd=("acc_deploy", "std"), Ymean_deploy=("Ymean_deploy", "mean")).reset_index()
    s["cert_rate"] = s.cert / s.reps
    s["cert_lo"], s["cert_hi"] = grlib.wilson(s.cert, s.reps)
    s["false_rate"] = s.false / s.reps
    s["false_lo"], s["false_hi"] = grlib.wilson(s.false, s.reps)
    return s


def choose_tau(Gdev_fn, acc_dev, specs, p, grid, margin, frac):
    """Dev-only tau rule (a real accuracy/budget trade-off with certifiable headroom):
    among grid values where (i) the accuracy-greedy candidate (smallest kappa) is infeasible,
    G_dev > 1-p, and (ii) the reference router is safe with margin, G_dev <= margin*(1-p),
    take the SMALLEST tau at which the best 'safe-with-margin' K=32 candidate (G_dev <= margin*(1-p))
    recovers >= `frac` of the accuracy gap between the reference router and the greedy one.
    If no tau reaches `frac`, take the tau maximising that recovered fraction."""
    det = [j for j, s in enumerate(specs[:32]) if s["kind"] == "det"]
    kap = np.array([specs[j]["kappa"] for j in det])
    greedy = det[int(kap.argmin())]
    a_ref, a_gr = acc_dev[0], acc_dev[greedy]
    best, best_tau = -np.inf, None
    for tau in grid:
        Gd = Gdev_fn(tau)[:32]
        if Gd[greedy] <= 1 - p or Gd[0] > margin * (1 - p):
            continue
        ok = Gd <= margin * (1 - p)
        rec = (acc_dev[:32][ok].max() - a_ref) / max(a_gr - a_ref, 1e-9)
        if rec >= frac:
            return float(tau), (f"smallest tau with G_dev(greedy) > 1-p, G_dev(reference) <= {margin}(1-p), and "
                                f"best K=32 candidate with G_dev <= {margin}(1-p) recovering >= {frac:.0%} of the "
                                f"reference->greedy dev accuracy gap (recovered {rec:.2f})")
        if rec > best:
            best, best_tau = rec, tau
    if best_tau is None:
        return None, "no tau with greedy infeasible and reference safe on dev; lower p or raise max_tokens"
    return float(best_tau), (f"[fallback] tau maximising the recovered accuracy gap ({best:.2f}) subject to "
                             f"G_dev(greedy) > 1-p and G_dev(reference) <= {margin}(1-p)")


def main():
    ap = grlib.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--table-dir", default=None,
                    help="directory with table.parquet and features.parquet (default: <run-dir>/table); "
                         "the precomputed outcome table is shipped in ../results/reasoning/table")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--stress-reps", type=int, default=None, help="default = --reps")
    ap.add_argument("--eps", default="1.0,0.5,0.2")
    ap.add_argument("--Ks", default="8,32,128")
    ap.add_argument("--ns", default=None, help="comma list; default geometric 250 .. 10 x eval pool")
    ap.add_argument("--n-grid", type=int, default=9)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--kappa-range", default="1e-3,3.0", help="accuracy per unit of (Y / dev-median Y)")
    ap.add_argument("--tau-margin", type=float, default=0.5, help="safe-with-margin factor on 1-p (dev)")
    ap.add_argument("--tau-frac", type=float, default=0.5, help="accuracy-gap fraction to recover (dev)")
    ap.add_argument("--stress-gap", type=float, default=0.002)
    ap.add_argument("--tau-override", default=None, help="e.g. '0.9:2048,0.95:3072' (default: dev rule)")
    ap.add_argument("--ps", default="0.9,0.95,0.99",
                    help="quantile levels; a level whose tau rule fails is skipped (p=0.99 in the paper)")
    ap.add_argument("--deploy-eps", type=float, default=0.5)
    ap.add_argument("--deploy-K", type=int, default=32)
    ap.add_argument("--deploy-p", type=float, default=0.90)
    ap.add_argument("--deploy-n", type=int, default=None, help="log size for the deploy plan (default: eval pool)")
    ap.add_argument("--quick", action="store_true", help="quick check: 40 reps, 5 log sizes")
    ap.add_argument("--deploy-only", action="store_true", help="stop after writing the deploy plan (no replay)")
    args = ap.parse_args()
    global PS
    PS = tuple(float(v) for v in args.ps.split(","))
    if args.quick:
        args.reps = min(args.reps, 40)
        args.n_grid = 5
    args.stress_reps = args.stress_reps or args.reps
    cfg = grlib.load_config(args.config)
    rd = grlib.run_dir(cfg, args)
    od = os.path.join(rd, "certify"); os.makedirs(od, exist_ok=True)
    acts = grlib.action_names(cfg); M = len(acts)
    seed = cfg["seed"]
    outcome = "output_tokens"
    td = args.table_dir or os.path.join(rd, "table")
    tab = pd.read_parquet(os.path.join(td, "table.parquet"))
    feat = pd.read_parquet(os.path.join(td, "features.parquet"))
    # all prompts with features; outcomes only for table prompts (dev/eval)
    feat = feat.set_index("prompt_id")
    tab = tab.set_index("prompt_id")
    ids_tab = tab.index.values
    ids_dep = feat.index[feat["partition"] == "deploy"].values
    ids_all = np.concatenate([ids_tab, ids_dep])
    fcols = [c for c in feat.columns if c.startswith("f_")]
    Xraw = feat.loc[ids_all, fcols].values.astype(float)
    part = np.concatenate([tab.loc[ids_tab, "partition"].values, np.array(["deploy"] * len(ids_dep))])
    dev, ev = part == "dev", part == "eval"
    ds_all = np.concatenate([tab.loc[ids_tab, "dataset"].values, feat.loc[ids_dep, "dataset"].values])
    dsev = ds_all[ev]
    mu, sd = Xraw[dev].mean(0), Xraw[dev].std(0) + 1e-9
    X = (Xraw - mu) / sd
    Ntab = len(ids_tab)
    T = np.zeros((len(ids_all), M), np.float64); Q = np.zeros((len(ids_all), M), np.int8)
    TR = np.zeros((len(ids_all), M), bool)
    T[:Ntab] = np.stack([tab["tokens__" + a].values for a in acts], 1)
    TR[:Ntab] = np.stack([(tab["finish__" + a] == "length").values for a in acts], 1)
    Q[:Ntab] = np.stack([tab["correct__" + a].values for a in acts], 1)
    print(f"prompts: dev={dev.sum()} eval={ev.sum()} deploy={len(ids_dep)}; actions={acts}; Y = generated tokens")

    pacc, pk, diag = fit_predictors(X, dev, T, Q)
    ref = acts.index(cfg["reference_action"])
    krange = [float(v) for v in args.kappa_range.split(",")]
    P, specs = build_family(pacc, pk, ref, krange)
    lb = cfg.get("logging_base", {"action": acts[0]})
    base_spec = acts.index(lb["action"]) if "action" in lb else float(lb["kappa"])
    epss = [float(e) for e in args.eps.split(",")]
    Ks = [int(k) for k in args.Ks.split(",")]

    # ---- logging policies and W --------------------------------------------------------
    P0s, Wtab, base_frac = [], [], None
    for e in epss:
        P0, base = logging_policy(pacc, pk, base_spec, e)
        P0s.append(P0)
        ratio = P / P0[None].astype(np.float32)
        Wtab.append({K: float(ratio[:K].max()) for K in Ks + [args.deploy_K]})
        base_frac = {acts[a]: float((base[ev] == a).mean()) for a in range(M)}
    print("logging base router action mix (eval):", base_frac)
    print("W (max importance weight over family):", {e: Wtab[i] for i, e in enumerate(epss)})

    # ---- tau selection on dev ----------------------------------------------------------
    yd = T[dev]
    grid = np.unique(np.round(np.geomspace(np.quantile(yd[yd > 0], 0.02), yd.max(), 160)))
    Pd, Td = P[:, dev], T[dev]
    acc_dev = (Pd * Q[dev][None]).sum(2).mean(1)

    def Gdev(tau):
        return (Pd * (Td > tau)[None]).sum(2).mean(1)
    taus, tau_info = {}, {}
    over = dict(kv.split(":") for kv in args.tau_override.split(",")) if args.tau_override else {}
    for p in PS:
        if str(p) in over or f"{p:.2f}" in over:
            tau, rule = float(over.get(str(p), over.get(f"{p:.2f}"))), "user override"
        else:
            tau, rule = choose_tau(Gdev, acc_dev, specs, p, grid, args.tau_margin, args.tau_frac)
        if tau is None:
            print(f"WARNING: tau selection failed for p={p}: {rule} -- skipping this level")
            continue
        Gd = Gdev(tau)
        taus[p] = tau
        tau_info[str(p)] = {"tau": tau, "rule": rule, "dev_frac_feasible_K32": float((Gd[:32] <= 1 - p).mean()),
                            "dev_frac_feasible_K128": float((Gd <= 1 - p).mean()),
                            "dev_G_min": float(Gd.min()), "dev_G_max": float(Gd.max())}
    PS = tuple(p for p in PS if p in taus)
    if not PS:
        sys.exit("tau selection failed for every quantile level")
    print("tau:", json.dumps(tau_info, indent=1))

    # ---- ground truth on eval -------------------------------------------------------------
    Pev, Tev, Qev = P[:, ev], T[ev], Q[ev]
    gt = []
    for j, s in enumerate(specs):
        rec = dict(j=j, **s)
        rec["acc_true"] = float((Pev[j] * Qev).sum(1).mean())
        rec["Ymean_true"] = float((Pev[j] * Tev).sum(1).mean())
        rec["acc_dev"] = float((Pd[j] * Q[dev]).sum(1).mean())
        mix = Pev[j].mean(0)
        for a in range(M):
            rec["frac_" + acts[a]] = float(mix[a])
        for p, tau in taus.items():
            rec[f"G_true_p{p}"] = float((Pev[j] * (Tev > tau)).sum(1).mean())
            rec[f"G_dev_p{p}"] = float(Gdev(tau)[j])
            rec[f"q_true_p{p}"] = mixture_quantile(Pev[j], Tev, p)
        gt.append(rec)
    gt = pd.DataFrame(gt)
    # tail- and bulk-effective overlaps per candidate, logging eps and p (paper Theorem 3 and appendix
    # "Predicted versus observed log sizes"): V_tau = E_pi0[w^2 1{Y>tau}], V_F = E_pi0[w^2 1{Y<=tau}].
    for i, e in enumerate(epss):
        P0ev = P0s[i][ev]
        for p, tau in taus.items():
            exc_ev = (Tev > tau).astype(float)
            w2 = Pev ** 2 / P0ev[None]                       # (J, Nev, M): E_{A~pi0}[w^2 f] = sum_a pi^2/pi0 f
            gt[f"V_tau_p{p}_eps{e}"] = (w2 * exc_ev[None]).sum(2).mean(1)
            gt[f"V_F_p{p}_eps{e}"] = (w2 * (1.0 - exc_ev)[None]).sum(2).mean(1)
    tails = []
    for a in range(M):
        for p, tau in taus.items():
            tails.append(dict(action=acts[a], p=p, tau=tau, P_exceed_dev=float((T[dev, a] > tau).mean()),
                              P_exceed_eval=float((Tev[:, a] > tau).mean()),
                              trunc_eval=float(TR[ev][:, a].mean()),
                              acc_eval=float(Qev[:, a].mean()), mean_Y_eval=float(Tev[:, a].mean()),
                              q_p_Y_eval=float(np.quantile(Tev[:, a], p))))
    tails = pd.DataFrame(tails)
    tails.to_csv(os.path.join(od, "action_tails.csv"), index=False)
    print(tails.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    # fallback = reference router (always cfg.reference_action); report whether it is safe
    fb = {"j": 0, "action": acts[ref], "acc_true": float(gt.acc_true[0]), "Ymean_true": float(gt.Ymean_true[0])}
    for p in taus:
        fb[f"G_true_p{p}"] = float(gt[f"G_true_p{p}"][0])
        fb[f"safe_p{p}"] = bool(gt[f"G_true_p{p}"][0] <= 1 - p)
    if not all(fb[f"safe_p{p}"] for p in taus):
        print("WARNING: the reference/fallback router is NOT safe at some p:", fb)

    # ---- stress tau: min_j G_j just above 1-p on eval (no candidate safe) --------------------
    stress = {}
    cand = np.unique(Tev)
    Kmax = max(Ks)

    def gmin(t):
        return (Pev[:Kmax] * (Tev > t)[None]).sum(2).mean(1).min()
    for p in PS:
        target = 1 - p + args.stress_gap
        lo, hi = 0, len(cand) - 1
        if gmin(cand[0]) < target:
            sys.exit("stress tau impossible: even the smallest tau has a safe candidate")
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if gmin(cand[mid]) >= target:
                lo = mid
            else:
                hi = mid - 1
        stress[p] = float(cand[lo])
        gt[f"G_stress_p{p}"] = (Pev * (Tev > stress[p])[None]).sum(2).mean(1)
    sG = {str(p): {"min": float(gt[f"G_stress_p{p}"][:Kmax].min()),
                   "median": float(gt[f"G_stress_p{p}"][:Kmax].median()),
                   "n_within_0.01_of_1-p": int((gt[f"G_stress_p{p}"][:Kmax] <= 1 - p + 0.01).sum()),
                   "n_safe": int((gt[f"G_stress_p{p}"][:Kmax] <= 1 - p).sum())} for p in PS}
    print("stress taus", stress, "G_true range (min, median) over family:", sG)
    gt.to_csv(os.path.join(od, "ground_truth_routers.csv"), index=False)

    Nev = int(ev.sum())
    if args.ns:
        ns = [int(v) for v in args.ns.split(",")]
    else:
        ns = sorted(set(np.round(np.geomspace(250, 10 * Nev, args.n_grid)).astype(int).tolist() + [Nev]))
    meta = dict(actions=acts, models=[a["model"] for a in cfg["actions"]], outcome=outcome,
                caps={a["name"]: grlib.action_max_tokens(cfg, a) for a in cfg["actions"]}, delta=DELTA,
                taus={str(k): v for k, v in taus.items()}, tau_info=tau_info,
                stress_taus={str(k): v for k, v in stress.items()}, stress_G_min_median=sG,
                stress_gap=args.stress_gap, W={str(e): Wtab[i] for i, e in enumerate(epss)},
                eps=epss, Ks=Ks, ns=ns, n_eval_pool=Nev, n_dev=int(dev.sum()), n_deploy=len(ids_dep),
                reps=args.reps, stress_reps=args.stress_reps, seed=seed, kappa_range=krange,
                temps=TEMPS, logging_base=lb, logging_base_mix_eval=base_frac,
                reference=fb, predictor_diag={acts[a]: d for a, d in diag.items()},
                methods=METHODS, n_lam=certs.N_LAM,
                note_with_replacement=("n > eval pool means resampling the finite empirical eval "
                                       "population with replacement; truths are pool quantities"))
    with open(os.path.join(od, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)

    # ---- single realistic logs -> deploy plan (paper: live deployment, p=0.90, eps=0.5, K=32) ----
    # Routers written to deploy_plan.json (deploy_verify.py runs any subset of them):
    #   certified          Exc-PB (Theorem 1) on ONE log of n = eval pool (each eval prompt once)
    #   certified_bet      betting certificate, unstratified, on the same log
    #   certified_bet_prod stratified betting certificate on the same log (the paper's live certificate)
    #   greedy             max IW accuracy among K on the same log, ignoring the constraint (uncertified)
    #   reference          always cfg.reference_action
    #   certified_10x      Exc-PB on a log of 10x the pool, resampled with replacement (valid for the pool only)
    #   think_always       uncertified contrast: always the largest thinking configuration
    di = epss.index(args.deploy_eps) if args.deploy_eps in epss else None
    P0d = P0s[di] if di is not None else logging_policy(pacc, pk, base_spec, args.deploy_eps)[0]
    Wd = float((P[:args.deploy_K] / P0d[None]).max())
    P0e = P0d[ev]
    tau_d = taus[args.deploy_p]
    Cd = certs.complexity(args.deploy_K, DELTA)
    tcols = ["acc_true", "Ymean_true", f"G_true_p{args.deploy_p}", f"q_true_p{args.deploy_p}"]

    def info(j, bound, V, nd, kind):
        return dict(j=int(j), kind=kind, n_log=int(nd), bound=None if bound is None else float(bound),
                    Vhat=float(V[j]), spec=specs[j], eval_truth={c: float(gt.loc[j, c]) for c in tcols})
    routers = {}
    for name, nd in (("certified", args.deploy_n or Nev), ("certified_10x", 10 * (args.deploy_n or Nev))):
        rng = np.random.default_rng([seed, 99, nd])
        # n <= pool: distinct prompts from a stratified split do not form an i.i.d. mixture log.
        # Unstratified calculations below are diagnostics only; use the stratified calculations
        # for the stated within-dataset i.i.d. population model. n > pool: i.i.d. resampling
        # with replacement supports a certificate for the finite evaluation pool only.
        idx = rng.permutation(Nev)[:nd] if nd <= Nev else rng.integers(0, Nev, nd)
        diagnostic_only = nd <= Nev
        scope = ("diagnostic only; no population certificate for this stratified single-pass log"
                 if diagnostic_only else "eval pool only (resampled with replacement)")
        A = np.minimum((rng.random(nd)[:, None] > np.cumsum(P0e[idx], 1)).sum(1), M - 1)
        w = Pev[:args.deploy_K, idx, A] / P0e[idx, A][None]
        V = (w * Qev[idx, A]).mean(-1)
        b = certs.exc_pb(w, (Tev[idx, A] > tau_d).astype(float), Wd, Cd)
        jc = int(certs.select_and_certify(b, args.deploy_p, utility=V, rng=np.random.default_rng(seed)))
        if jc >= 0:
            kind = (f"unstratified Exc-PB threshold crossing; {scope}" if diagnostic_only
                    else f"exceedance-certified; certificate covers: {scope}")
            routers[name] = info(jc, b[jc], V, nd, kind)
        else:
            kind = ("no unstratified Exc-PB diagnostic crossing -> reference fallback"
                    if diagnostic_only else "NOTHING CERTIFIED -> reference fallback")
            routers[name] = info(0, None, V, nd, kind)
            print(f"No Exc-PB threshold crossing on the n={nd} deploy log; {name} = reference router")
        if name == "certified":
            # same log, betting certificate for the finite candidate list (appendix "The betting certificate")
            bb = certs.exc_bet(w, (Tev[idx, A] > tau_d).astype(float), Wd, Cd)
            jb = int(certs.select_and_certify(bb, args.deploy_p, utility=V, rng=np.random.default_rng(seed)))
            if jb >= 0:
                kind = (f"unstratified Exc-Bet threshold crossing; {scope}" if diagnostic_only
                        else f"betting-certified; certificate covers: {scope}")
                routers["certified_bet"] = info(jb, bb[jb], V, nd, kind)
            else:
                kind = ("no unstratified Exc-Bet diagnostic crossing -> reference fallback"
                        if diagnostic_only else "NOTHING CERTIFIED (betting) -> reference fallback")
                routers["certified_bet"] = info(0, None, V, nd, kind)
            jg = int(V.argmax())
            routers["greedy"] = info(jg, b[jg], V, nd, "accuracy-greedy, UNCERTIFIED")
            # Stratified certificates. The partition is stratified by dataset, so this log has fixed
            # per-dataset counts and is i.i.d. only within each dataset. The target is the declared mixture
            # G = q G_gsm8k + (1-q) G_math, with q the GSM8K share of the pool.
            # (a) union-bound variant: bound each (router, dataset) at delta/(2K) and combine,
            #     U_j = q U_{j,gsm8k} + (1-q) U_{j,math}; valid after selection over K, but loose
            #     (reported in the paper as certifying nothing).
            ds_log = dsev[idx]
            q = float((ds_all == "gsm8k").mean())
            exc_log = (Tev[idx, A] > tau_d).astype(float)
            strat = {}
            for meth, f in (("bet", certs.exc_bet), ("pb", certs.exc_pb), ("eb", certs.exc_eb)):
                Us = []
                for dname in ("gsm8k", "math"):
                    mk = ds_log == dname
                    Us.append(f(w[:, mk], exc_log[mk], Wd, np.log(2 * args.deploy_K / DELTA)))
                U = q * Us[0] + (1 - q) * Us[1]
                js = int(certs.select_and_certify(U, args.deploy_p, utility=V, rng=np.random.default_rng(seed)))
                strat[meth] = dict(j=js, bound=None if js < 0 else float(U[js]),
                                   U_gsm8k=None if js < 0 else float(Us[0][js]),
                                   U_math=None if js < 0 else float(Us[1][js]),
                                   min_bound=float(U.min()), argmin=int(U.argmin()))
                if js >= 0:
                    routers[f"certified_{meth}_strat"] = info(js, U[js], V, nd,
                        f"stratified {meth}-certified (q={q:.4f}); within-dataset iid population model")
            # (b) product variant (appendix "The betting certificate", stratified version): the two strata's
            #     betting wealths are independent, so their product at the true means is an e-value; one
            #     confidence budget log(K/delta) for both strata. This is the certificate used live.
            mk = ds_log == "gsm8k"
            Up = certs.exc_bet_stratified([(w[:, mk], exc_log[mk][None].repeat(len(w), 0)),
                                           (w[:, ~mk], exc_log[~mk][None].repeat(len(w), 0))],
                                          (q, 1 - q), Wd, Cd)
            js = int(certs.select_and_certify(Up, args.deploy_p, utility=V, rng=np.random.default_rng(seed)))
            strat["bet_prod"] = dict(j=js, bound=None if js < 0 else float(Up[js]), min_bound=float(Up.min()),
                                     argmin=int(Up.argmin()), bound_greedy=float(Up[jg]), bound_ref=float(Up[0]))
            if js >= 0:
                routers["certified_bet_prod"] = info(js, Up[js], V, nd,
                    f"stratified product-betting certified (q={q:.4f}); within-dataset iid population model")
            print("stratified deploy certificates (q=%.4f):" % q, json.dumps(strat, indent=1))
            with open(os.path.join(od, "deploy_stratified.json"), "w") as fh:
                json.dump(dict(q=q, n_gsm8k=int((ds_log == "gsm8k").sum()), n_math=int((ds_log == "math").sum()),
                               delta=DELTA, K=args.deploy_K, W=Wd, tau=tau_d, p=args.deploy_p, results=strat,
                               unstratified_bet=dict(j=jb, bound=None if jb < 0 else float(bb[jb]))), fh, indent=1)
            routers["reference"] = info(0, b[0], V, nd, "reference (always %s)" % acts[ref])
    # uncertified contrast: always reason with the largest model (q30b_think if present, else the
    # thinking action with the largest mean Y); fixed before any deploy generation.
    think_acts = [i for i, n in enumerate(acts) if n.endswith("_think")]
    if think_acts:
        a_th = acts.index("q30b_think") if "q30b_think" in acts else max(think_acts, key=lambda i: Tev[:, i].mean())
        routers["think_always"] = dict(
            j=-1, kind=f"contrast: always {acts[a_th]}, UNCERTIFIED", n_log=0, bound=None, Vhat=float("nan"),
            spec={"kind": "fixed", "action": int(a_th)}, fixed_action=int(a_th),
            eval_truth={"acc_true": float(Qev[:, a_th].mean()), "Ymean_true": float(Tev[:, a_th].mean()),
                        f"G_true_p{args.deploy_p}": float((Tev[:, a_th] > tau_d).mean()),
                        f"q_true_p{args.deploy_p}": float(np.quantile(Tev[:, a_th], args.deploy_p, method="inverted_cdf"))})
    plan = dict(outcome=outcome, eps=args.deploy_eps, K=args.deploy_K, p=args.deploy_p, tau=tau_d, W=Wd,
                delta=DELTA, routers=routers)
    dep = ~(dev | ev)
    probs = pd.DataFrame({"prompt_id": ids_all[dep]})
    for name, r in routers.items():
        for a in range(M):
            probs[f"{name}__{acts[a]}"] = (float(a == r["fixed_action"]) if "fixed_action" in r
                                          else P[r["j"]][dep][:, a])
    grlib.write_parquet(probs, os.path.join(od, "deploy_router_probs.parquet"))
    with open(os.path.join(od, "deploy_plan.json"), "w") as f:
        json.dump(plan, f, indent=1, default=float)
    print(f"deploy plan (p={args.deploy_p}, tau={tau_d:.4g}, eps={args.deploy_eps}, K={args.deploy_K}, W={Wd:.0f}):")
    for name, r in routers.items():
        print(f"  {name:14s} j={r['j']:3d} n_log={r['n_log']:7d} bound={r['bound']} spec={r['spec']} "
              f"eval-pool truth: acc={r['eval_truth']['acc_true']:.3f} G={r['eval_truth'][tcols[2]]:.4f}")

    if args.deploy_only:
        return
    # ---- replay ------------------------------------------------------------------------
    G.update(P=Pev.astype(np.float32), P0s=[p0[ev] for p0 in P0s], T=Tev, Q=Qev, W=Wtab, Ks=Ks,
             taus={"main": taus, "stress": stress}, seed=seed)
    for tag, reps in (("main", args.reps), ("stress", args.stress_reps)):
        t0 = time.time()
        jobs = []
        for ei in range(len(epss)):
            for n in ns:
                ch = max(1, reps // max(1, int(np.ceil(n / 5000)))) if n > 5000 else reps
                ch = min(ch, max(5, reps // 4))
                for s0 in range(0, reps, ch):
                    jobs.append((tag, ei, n, s0, min(ch, reps - s0)))
        jobs.sort(key=lambda j: -j[2] * j[4])
        with Pool(args.procs) as pool:
            res = pool.map(run_cell, jobs, chunksize=1)
        raw = pd.DataFrame([r for rr in res for r in rr],
                           columns=["tag", "eps_i", "K", "n", "p", "rep", "method", "j", "bound", "Vhat"])
        raw["eps"] = [epss[i] for i in raw.eps_i]
        gcol = "G_true" if tag == "main" else "G_stress"
        raw["G_sel"] = np.nan; raw["acc_sel"] = np.nan; raw["Ymean_sel"] = np.nan
        for p in PS:
            m = (raw.p == p).values
            jj = raw.j.values[m]
            ok = jj >= 0
            for col, src in (("G_sel", f"{gcol}_p{p}"), ("acc_sel", "acc_true"), ("Ymean_sel", "Ymean_true")):
                v = np.full(len(jj), np.nan); v[ok] = gt[src].values[jj[ok]]
                raw.loc[m, col] = v
        raw["certified"] = raw.j >= 0
        raw["false_cert"] = raw.certified & (raw.G_sel > 1 - raw.p)
        raw["slack"] = raw.bound - raw.G_sel
        raw["acc_deploy"] = np.where(raw.certified, raw.acc_sel, fb["acc_true"])
        raw["Ymean_deploy"] = np.where(raw.certified, raw.Ymean_sel, fb["Ymean_true"])
        raw.drop(columns=["eps_i"]).to_csv(os.path.join(od, f"raw_{tag}.csv.gz"), index=False)
        summ = summarize(raw)
        summ.to_csv(os.path.join(od, f"summary_{tag}.csv"), index=False)
        print(f"{tag}: {len(raw)} rows in {time.time() - t0:.1f}s")

    # ---- headline -----------------------------------------------------------------------
    sm = pd.read_csv(os.path.join(od, "summary_main.csv"))
    ss = pd.read_csv(os.path.join(od, "summary_stress.csv"))
    K0 = 32 if 32 in Ks else Ks[0]
    h = sm[(sm.K == K0) & (np.isclose(sm.p, 0.95))]
    print(f"\nHEADLINE certification rate (K={K0}, p=0.95; rows eps, cols method):")
    for n in [ns[0], Nev, ns[-1]]:
        t = h[h.n == n].pivot(index="eps", columns="method", values="cert_rate")
        print(f"n={n}\n{t.round(3).to_string()}")
    t = h.groupby(["method"]).agg(max_false=("false_rate", "max")).T
    print("max false-cert rate over (eps, n), main:\n", t.round(4).to_string())
    hs = ss[np.isclose(ss.p, 0.95)]
    t = hs.groupby(["method"]).agg(false_rate=("false_rate", "max"), false_hi=("false_hi", "max")).T
    print("STRESS (p=0.95) max false-cert rate over (eps, K, n):\n", t.round(4).to_string())
    print(f"outputs in {od}")


if __name__ == "__main__":
    main()
