#!/usr/bin/env python3
"""Step 4 (GPU): live deployment check  log -> select -> certify -> deploy -> verify.

Takes routers from certify/deploy_plan.json (paper: p=0.90, eps=0.5, K=32) and runs them on the
2,000 held-out `deploy` prompts (never used for dev, logging or the outcome table), generating
fresh responses. Routers used in the paper (default --routers):
  certified      Exc-PB on one log of n = eval pool; certifies nothing there, so it is the
                 reference router (fallback)
  greedy         accuracy-greedy router on the same log; on this log it coincides with the router
                 certified by the (stratified) betting certificate (certified_bet, certified_bet_prod)
  certified_10x  Exc-PB on a log of 10 x eval pool (resampled with replacement)
  think_always   uncertified contrast: always the largest thinking configuration
Routers are sampled with a per-router seed, so the subset and order do not affect each other.

Reports realized P(Y > tau) (Wilson CI) vs the certified bound, realized p-quantile vs tau,
accuracy and mean tokens (paper table "Live deployment").

  python deploy_verify.py --config configs/open.yaml
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import grlib


def main():
    ap = grlib.add_engine_args(grlib.add_common_args(argparse.ArgumentParser()))
    ap.add_argument("--routers", default="certified,certified_10x,greedy,think_always",
                    help="comma list of router names in certify/deploy_plan.json")
    ap.add_argument("--max-prompts", "--limit", type=int, default=None, help="only the first N deploy prompts")
    args = ap.parse_args()
    cfg = grlib.load_config(args.config)
    rd = grlib.run_dir(cfg, args)
    cd = os.path.join(rd, "certify")
    od = os.path.join(rd, "deploy"); os.makedirs(od, exist_ok=True)
    plan = json.load(open(os.path.join(cd, "deploy_plan.json")))
    probs = pd.read_parquet(os.path.join(cd, "deploy_router_probs.parquet"))
    pr = pd.read_parquet(os.path.join(rd, "prompts.parquet"))
    pr = pr[pr["partition"] == "deploy"].merge(probs, on="prompt_id").sort_values("prompt_id")
    if args.max_prompts:
        pr = pr.iloc[:args.max_prompts]
    acts = grlib.action_names(cfg)
    routers = args.routers.split(",")
    reqs = []
    for rname in routers:
        assert rname in plan["routers"], f"unknown router {rname}; plan has {list(plan['routers'])}"
        Pm = pr[[f"{rname}__{a}" for a in acts]].values.astype(float)
        Pm /= Pm.sum(1, keepdims=True)
        rng = np.random.default_rng([cfg["seed"], 2024, grlib.stable_hash(rname) % 2 ** 31])  # seed by name, not position
        A = np.minimum((rng.random(len(pr))[:, None] > np.cumsum(Pm, 1)).sum(1), len(acts) - 1)
        r = pr.assign(action=[acts[a] for a in A], router=rname)
        r["req_id"] = r.prompt_id + "|" + r.action + "|" + rname
        reqs.append(r)
    req = pd.concat(reqs, ignore_index=True)
    print(f"deploy prompts={len(pr)} routers={routers} requests={len(req)}  tau={plan['tau']} p={plan['p']}")
    sent = grlib.run_requests(req, cfg, os.path.join(od, "gen"), args, salt="deploy",
                              obs_glob=os.path.join(rd, "gen", "*", "part-*.parquet"))
    if not sent:
        return

    df = grlib.read_parts(os.path.join(od, "gen", "part-*.parquet"))
    df["router"] = df.req_id.str.split("|").str[-1]
    tau, p = plan["tau"], plan["p"]
    outcome = plan.get("outcome", "output_tokens")
    df["Y"] = df.output_tokens.astype(float)
    deploy_gpu_h = float((df.gen_seconds_approx * df.get("n_gpus", 1)).sum() / 3600)
    rep = {"plan": plan, "routers": {}}
    rows = []
    for rname in routers:
        d = df[df.router == rname]
        n = len(d)
        k = int((d.Y > tau).sum())
        lo, hi = grlib.wilson(k, n)
        acc_lo, acc_hi = grlib.wilson(d.correct.sum(), n)
        pr_ = plan["routers"][rname]
        bound, evt = pr_["bound"], pr_["eval_truth"]
        rec = dict(router=rname, j=pr_["j"], n_log=pr_["n_log"], n=n, P_exceed=k / max(n, 1), P_exceed_lo=float(lo), P_exceed_hi=float(hi),
                   bound=bound, target=1 - p, meets_SLA=bool(k / max(n, 1) <= 1 - p),
                   q_p_realized=float(np.quantile(d.Y, p, method="inverted_cdf")) if n else np.nan, tau=tau, outcome=outcome,
                   mean_Y=float(d.Y.mean()),
                   accuracy=float(d.correct.mean()), acc_lo=float(acc_lo), acc_hi=float(acc_hi),
                   mean_tokens=float(d.output_tokens.mean()), trunc_rate=float((d.finish_reason == "length").mean()),
                   eval_pool_G=evt.get(f"G_true_p{p}"), eval_pool_acc=evt.get("acc_true"),
                   action_mix=d.action.value_counts(normalize=True).round(3).to_dict())
        rep["routers"][rname] = rec
        rows.append(rec)
    rep["deploy_gpu_hours_approx"] = deploy_gpu_h
    with open(os.path.join(od, "deploy_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    t = pd.DataFrame(rows).drop(columns=["action_mix", "outcome"])
    print(f"\nDEPLOY VERIFICATION (fresh prompts, fresh generations; Y = {outcome}, tau = {tau:.6g}, "
          f"deploy GPU-hours ~ {deploy_gpu_h:.2f})")
    print(t.round(4).to_string(index=False))
    for r in rows:
        print(f"  {r['router']} ({plan['routers'][r['router']]['kind']}): action mix {r['action_mix']}")
    print(f"certified routers: realized P(Y>tau) should be <= 1-p = {1 - p:.2f} (up to sampling error of "
          f"n={len(pr)}), i.e. realized q_{p} <= tau.")
    print(f"wrote {od}/deploy_report.json")


if __name__ == "__main__":
    main()
