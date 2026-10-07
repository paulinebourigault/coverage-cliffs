#!/usr/bin/env python3
"""Step 0 (CPU): build the prompt pool and its fixed partition (dev / eval / deploy).

GSM8K and MATH (train + test) are deduplicated, subsampled proportionally to dataset size, and
split, stratified by dataset, into deploy / dev / eval. The result is deterministic given the
config seed and the dataset versions; the printed digest identifies the partition (machines
running different shards must see the same digest). The deploy partition is never generated in
the outcome table; it is used only by deploy_verify.py.

  python prepare_prompts.py --config configs/open.yaml
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

import grlib


def _load_hf(spec):
    """Try each hf_id in turn; per-subject configs if the repo has them, else the default one."""
    from datasets import load_dataset, get_dataset_config_names
    ids = spec["hf_id"] if isinstance(spec["hf_id"], list) else [spec["hf_id"]]
    want = spec.get("hf_config")
    want = want if isinstance(want, list) else [want]
    last = None
    for hid in ids:
        try:
            try:
                avail = set(get_dataset_config_names(hid))
            except Exception:
                avail = set()
            cfgs = [c for c in want if c in avail] or ([None] if not avail or "default" in avail else [sorted(avail)[0]])
            parts = []
            for c in cfgs:
                for sp in spec["splits"]:
                    ds = load_dataset(hid, c, split=sp) if c else load_dataset(hid, split=sp)
                    df = ds.to_pandas()
                    df["source_split"] = sp
                    df["subset"] = c or ""
                    parts.append(df)
            out = pd.concat(parts, ignore_index=True)
            print(f"loaded {spec['name']} from {hid} configs={cfgs}: {len(out)} rows")
            return out, hid
        except Exception as e:
            print(f"  could not load {hid}: {type(e).__name__}: {str(e)[:200]}")
            last = e
    raise RuntimeError(f"no source worked for {spec['name']}: {last}")


def load_real(cfg):
    rows = []
    for spec in cfg["prompts"]["datasets"]:
        df, hid = _load_hf(spec)
        if spec["name"] == "gsm8k":
            for r in df.itertuples():
                gold = r.answer.split("####")[-1].strip().replace(",", "")
                rows.append(dict(dataset="gsm8k", problem=r.question, gold=gold, level=-1,
                                 subject="", source=f"{hid}:{r.source_split}"))
        elif spec["name"] == "math":
            for r in df.to_dict("records"):
                gold = grlib.last_boxed(r["solution"])
                if gold is None:
                    continue
                lv = str(r.get("level", ""))
                lv = int(lv.split()[-1]) if lv.split() and lv.split()[-1].isdigit() else -1
                rows.append(dict(dataset="math", problem=r["problem"], gold=gold.strip(), level=lv,
                                 subject=r.get("type", r.get("subset", "")),
                                 source=f"{hid}:{r['source_split']}"))
        else:
            raise ValueError("add a loader for dataset " + spec["name"])
    return pd.DataFrame(rows)


def main():
    ap = grlib.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--deploy-size", type=int, default=None)
    args = ap.parse_args()
    cfg = grlib.load_config(args.config)
    pc = cfg["prompts"]
    maxp = args.max_prompts or pc["max_prompts"]
    dsz = pc["deploy_size"] if args.deploy_size is None else args.deploy_size
    out = grlib.run_dir(cfg, args)

    df = load_real(cfg)
    n0 = len(df)
    df["prompt_id"] = [d + "-" + hashlib.sha1(p.strip().encode()).hexdigest()[:12]
                       for d, p in zip(df.dataset, df.problem)]
    df = df.drop_duplicates("prompt_id").sort_values("prompt_id").reset_index(drop=True)
    ndup = n0 - len(df)
    rng = np.random.default_rng(cfg["seed"])
    if len(df) > maxp:  # proportional stratified subsample by dataset
        keep = []
        for ds, idx in sorted(df.groupby("dataset").indices.items()):
            keep.append(rng.choice(idx, int(round(maxp * len(idx) / len(df))), replace=False))
        df = df.iloc[np.sort(np.concatenate(keep))].reset_index(drop=True)
    # partition: stratified by dataset; deploy first, then dev/eval
    part = np.empty(len(df), object)
    for ds, idx in df.groupby("dataset").indices.items():
        idx = rng.permutation(idx)
        nd = int(round(dsz * len(idx) / len(df)))
        rest = idx[nd:]
        ndev = int(round(pc["dev_frac"] * len(rest)))
        part[idx[:nd]] = "deploy"
        part[rest[:ndev]] = "dev"
        part[rest[ndev:]] = "eval"
    df["partition"] = part
    df["prompt_chars"] = df.problem.str.len()
    grlib.write_parquet(df, os.path.join(out, "prompts.parquet"))
    digest = hashlib.sha1("".join(df.prompt_id + df.partition).encode()).hexdigest()[:16]
    meta = {"n_prompts": len(df), "duplicates_dropped": int(ndup),
            "counts": df.groupby(["dataset", "partition"]).size().unstack(fill_value=0).to_dict(),
            "digest": digest, "seed": cfg["seed"]}
    with open(os.path.join(out, "prompts_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(df.groupby(["dataset", "partition"]).size().unstack(fill_value=0))
    print(f"wrote {out}/prompts.parquet  n={len(df)}  digest={digest}")


if __name__ == "__main__":
    main()
