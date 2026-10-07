"""Recompute the held-out deployment measurements (paper table "Held-out evaluation") from the released
per-request records results/reasoning/deploy/deploy_generations.parquet.

Each record is one fresh generation on one of the 2,000 held-out problems; req_id = "<prompt>|<action>|<router>".
Routers (see results/reasoning/certify/deploy_plan.json):
  greedy        = router 5, the router certified by the stratified Exc-Bet certificate on the 9,800-problem log
  certified     = reference router used as the fallback when Exc-PB certifies nothing on that log
  certified_10x = router certified by Exc-PB on the 98,000-request pool-resampled log
  think_always  = always Qwen3-30B-A3B with thinking (uncertified contrast)
Budget tau = 2,161 generated tokens at p = 0.90.

usage: python deploy_from_records.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 2161

g = pd.read_parquet(os.path.join(HERE, "..", "results", "reasoning", "deploy", "deploy_generations.parquet"))
g["router"] = g.req_id.str.split("|").str[-1]
rows = []
for r, d in g.groupby("router"):
    rows.append(dict(router=r, n=len(d), P_exceed=(d.output_tokens > TAU).mean(),
                     q90=np.quantile(d.output_tokens, 0.9, method="inverted_cdf"), accuracy=d.correct.mean(),
                     mean_tokens=d.output_tokens.mean()))
print(pd.DataFrame(rows).round(4).to_string(index=False))
