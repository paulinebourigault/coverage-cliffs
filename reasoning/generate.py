#!/usr/bin/env python3
"""Step 1 (GPU, vLLM): one independent sample per (prompt, action) for dev+eval prompts.

Processes one model at a time (fresh subprocess per model id; all its actions, e.g. thinking
on/off, are batched together), checkpoints every --chunk-size requests to parquet, and resumes
by skipping done (prompt_id, action) pairs. Shard with --shard i --num-shards k (round-robin
over the sorted prompt ids; every shard writes to its own directory under gen/).

  python generate.py --config configs/open.yaml                      # single GPU
  python generate.py --config configs/open.yaml --shard 0 --num-shards 4
"""
import argparse
import os
import time

import pandas as pd

import grlib


def main():
    ap = grlib.add_engine_args(grlib.add_common_args(argparse.ArgumentParser()))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--actions", default=None, help="comma list (default: all in config)")
    ap.add_argument("--max-prompts", "--limit", type=int, default=None,
                    help="only the first N dev+eval prompts (pilot); default all")
    ap.add_argument("--include-deploy", action="store_true",
                    help="also generate the deploy partition (not used by the study)")
    args = ap.parse_args()
    cfg = grlib.load_config(args.config)
    rd = grlib.run_dir(cfg, args)
    pr = pd.read_parquet(os.path.join(rd, "prompts.parquet"))
    if not args.include_deploy:
        pr = pr[pr["partition"] != "deploy"]
    pr = pr.sort_values("prompt_id").reset_index(drop=True)
    if args.max_prompts:
        pr = pr.iloc[:args.max_prompts]
    pr = pr.iloc[args.shard::args.num_shards]
    acts = args.actions.split(",") if args.actions else grlib.action_names(cfg)
    req = pd.concat([pr.assign(action=a) for a in acts], ignore_index=True)
    req["req_id"] = req.prompt_id + "|" + req.action
    out = os.path.join(rd, "gen", f"shard{args.shard:03d}of{args.num_shards:03d}")
    print(f"shard {args.shard}/{args.num_shards}: {len(pr)} prompts x {len(acts)} actions "
          f"= {len(req)} requests -> {out}")
    t0 = time.time()
    sent = grlib.run_requests(req, cfg, out, args, salt="gen",
                              obs_glob=os.path.join(rd, "gen", "*", "part-*.parquet"))
    df = grlib.read_parts(os.path.join(out, "part-*.parquet"))
    if not sent or df.empty:
        return
    dt = time.time() - t0
    s = df.groupby("action").agg(n=("output_tokens", "size"), mean_out=("output_tokens", "mean"),
                                 acc=("correct", "mean"),
                                 trunc=("finish_reason", lambda f: (f == "length").mean()))
    print(s.round(3).to_string())
    print(f"shard total rows {len(df)}/{len(req)}; wall {dt / 60:.1f} min this invocation; "
          f"{df.output_tokens.sum()} output tokens in shard")


if __name__ == "__main__":
    main()
