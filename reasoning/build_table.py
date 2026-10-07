#!/usr/bin/env python3
"""Step 2 (CPU): merge generation shards -> full outcome table (prompt x action), audit, and
pre-generation router features.

Outputs in <run>/table/:
  table.parquet     one row per prompt with complete outcomes for every action:
                    tokens__<a>, correct__<a>, finish__<a>, input_tokens__<a>, plus dataset,
                    partition and MATH level
  long.parquet      merged per-(prompt, action) rows (deduplicated)
  features.parquet  prompt_id, partition, dataset, f_* features for ALL prompts (incl. deploy)
  audit.json        missing pairs, truncation, token quantiles, accuracy per action/dataset,
                    measured generation throughput per model

Features (context available before generation): dataset one-hot, log prompt chars, log word
count, TF-IDF(1-2 grams) -> TruncatedSVD(32) fitted on the dev partition only; optionally a
sentence-transformer embedding (--embed-model, PCA->32 on dev).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import grlib


def features(pr, embed_model=None, n_svd=32, seed=0):
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    dev = (pr["partition"] == "dev").values
    F = {}
    for ds in sorted(pr.dataset.unique()):
        F[f"f_ds_{ds}"] = (pr.dataset == ds).astype(float).values
    F["f_log_chars"] = np.log1p(pr.problem.str.len().values)
    F["f_log_words"] = np.log1p(pr.problem.str.split().str.len().values)
    F["f_n_digits"] = np.log1p(pr.problem.str.count(r"\d").values)
    F["f_n_latex"] = np.log1p(pr.problem.str.count(r"\\|\$").values)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    vec.fit(pr.problem[dev])
    T = vec.transform(pr.problem)
    k = min(n_svd, T.shape[1] - 1)
    svd = TruncatedSVD(k, random_state=seed).fit(T[dev])
    Z = svd.transform(T)
    for i in range(k):
        F[f"f_tfidf{i:02d}"] = Z[:, i]
    if embed_model:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(embed_model)
        E = st.encode(pr.problem.tolist(), batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        pca = PCA(32, random_state=seed).fit(E[dev])
        Ez = pca.transform(E)
        for i in range(32):
            F[f"f_emb{i:02d}"] = Ez[:, i]
    out = pd.DataFrame(F)
    out.insert(0, "dataset", pr.dataset.values)
    out.insert(0, "partition", pr["partition"].values)
    out.insert(0, "prompt_id", pr.prompt_id.values)
    return out


def qstats(x):
    q = np.quantile(x, [0.5, 0.9, 0.95, 0.99]) if len(x) else [np.nan] * 4
    return {"mean": float(np.mean(x)) if len(x) else np.nan, "p50": float(q[0]), "p90": float(q[1]),
            "p95": float(q[2]), "p99": float(q[3]), "max": float(np.max(x)) if len(x) else np.nan}


def main():
    ap = grlib.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--embed-model", default=None, help="e.g. sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="build the table from complete prompts even if pairs are missing")
    args = ap.parse_args()
    cfg = grlib.load_config(args.config)
    rd = grlib.run_dir(cfg, args)
    od = os.path.join(rd, "table"); os.makedirs(od, exist_ok=True)
    acts = grlib.action_names(cfg)
    pr = pd.read_parquet(os.path.join(rd, "prompts.parquet"))
    df = grlib.read_parts(os.path.join(rd, "gen", "*", "part-*.parquet"))
    assert not df.empty, "no generation parts found under gen/"
    n_raw = len(df)
    df = df.drop_duplicates(["prompt_id", "action"], keep="first")
    df = df[df.action.isin(acts)].copy()
    acfg = {a["name"]: a for a in cfg["actions"]}
    if "n_gpus" not in df:
        df["n_gpus"] = 1
    thr = {}  # measured aggregate throughput per model (descriptive only)
    for mid, d in df.groupby("model"):
        secs = d.gen_seconds_approx.sum()
        ng = int(d.n_gpus.median())
        thr[mid] = {"out_tok_per_s": float(d.output_tokens.sum() / max(secs, 1e-9)), "n_gpus": ng,
                    "gpu_hours": float(secs * ng / 3600)}
    target = pr[pr["partition"] != "deploy"]
    expected = len(target) * len(acts)
    have = df.set_index(["prompt_id", "action"]).index
    full = pd.MultiIndex.from_product([target.prompt_id, acts])
    missing = full.difference(have)
    cnt = df.groupby("prompt_id").action.nunique()
    complete = set(cnt[cnt == len(acts)].index) & set(target.prompt_id)
    audit = {"rows_raw": n_raw, "duplicates_dropped": int(n_raw - len(df)),
             "expected_pairs": expected, "present_pairs": int(len(df)), "missing_pairs": int(len(missing)),
             "missing_by_action": pd.Series([m[1] for m in missing]).value_counts().to_dict() if len(missing) else {},
             "complete_prompts": len(complete),
             "cap": cfg["max_tokens"], "measured_throughput": thr, "per_action": {}, "per_action_dataset": {}}
    if len(missing) and not args.allow_incomplete:
        print(f"WARNING: {len(missing)} missing (prompt, action) pairs; rerun generate.py to resume, "
              "or pass --allow-incomplete. Proceeding with complete prompts only.")
    for a in acts:
        d = df[df.action == a]
        tr = d.finish_reason == "length"
        audit["per_action"][a] = {"n": int(len(d)), "accuracy": float(d.correct.mean()),
                                  "trunc_rate": float(tr.mean()),
                                  "acc_if_not_trunc": float(d.correct[~tr].mean()),
                                  "out_tokens": qstats(d.output_tokens.values),
                                  "max_tokens": grlib.action_max_tokens(cfg, acfg[a]),
                                  "in_tokens_mean": float(d.input_tokens.mean()),
                                  "total_out_tokens": int(d.output_tokens.sum()),
                                  "gen_seconds_approx_sum": float(d.gen_seconds_approx.sum())}
        for ds, dd in d.groupby("dataset"):
            audit["per_action_dataset"][f"{a}|{ds}"] = {
                "n": int(len(dd)), "accuracy": float(dd.correct.mean()),
                "trunc_rate": float((dd.finish_reason == "length").mean()),
                "mean_out": float(dd.output_tokens.mean()), "p95_out": float(dd.output_tokens.quantile(.95))}
    # wide table
    d = df[df.prompt_id.isin(complete)]
    wide = d.pivot(index="prompt_id", columns="action",
                   values=["output_tokens", "correct", "finish_reason", "input_tokens"])
    tab = pd.DataFrame(index=wide.index)
    for a in acts:
        tab["tokens__" + a] = wide["output_tokens"][a].astype(int)
        tab["correct__" + a] = wide["correct"][a].astype(int)
        tab["finish__" + a] = wide["finish_reason"][a].astype(str)
        tab["input_tokens__" + a] = wide["input_tokens"][a].astype(int)
    tab = tab.reset_index().merge(pr[["prompt_id", "dataset", "partition", "level"]], on="prompt_id")
    grlib.write_parquet(tab, os.path.join(od, "table.parquet"))
    grlib.write_parquet(df.drop(columns=[c for c in ["_text_for_grading"] if c in df]),
                        os.path.join(od, "long.parquet"))
    feat = features(pr, args.embed_model, seed=cfg["seed"])
    grlib.write_parquet(feat, os.path.join(od, "features.parquet"))
    audit["table_prompts_by_partition"] = tab["partition"].value_counts().to_dict()
    audit["n_features"] = int(sum(c.startswith("f_") for c in feat.columns))
    T = np.stack([tab["tokens__" + a].values for a in acts], 1)
    Q = np.stack([tab["correct__" + a].values for a in acts], 1)
    audit["oracle_accuracy_any_action"] = float(Q.max(1).mean())
    audit["frac_prompts_cheapest_correct"] = float(Q[:, 0].mean())
    with open(os.path.join(od, "audit.json"), "w") as f:
        json.dump(audit, f, indent=1, default=float)
    # print summary
    print(f"\nrows={len(df)} expected={expected} missing={len(missing)} complete_prompts={len(complete)}")
    rows = []
    for a in acts:
        s = audit["per_action"][a]
        rows.append(dict(action=a, n=s["n"], acc=s["accuracy"], trunc=s["trunc_rate"],
                         mean=s["out_tokens"]["mean"], p50=s["out_tokens"]["p50"], p90=s["out_tokens"]["p90"],
                         p95=s["out_tokens"]["p95"], p99=s["out_tokens"]["p99"],
                         Mtok=s["total_out_tokens"] / 1e6))
    print(pd.DataFrame(rows).round(6).to_string(index=False))
    print("measured throughput per model:")
    print(pd.DataFrame(thr).T.round(4).to_string())
    print("per action x dataset:")
    print(pd.DataFrame(audit["per_action_dataset"]).T.round(3).to_string())
    print(f"oracle accuracy (best action per prompt): {audit['oracle_accuracy_any_action']:.3f}")
    print(f"wrote {od}/table.parquet ({len(tab)} prompts), features.parquet ({audit['n_features']} feats), audit.json")


if __name__ == "__main__":
    main()
