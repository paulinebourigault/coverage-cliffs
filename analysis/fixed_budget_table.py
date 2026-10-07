"""Reproduce main Table 2 from the released RouterBench replication summaries.

No experiments are run. Standard errors are method-wise Monte Carlo standard
errors, not paired-difference intervals or population-quality certificates.
"""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
d = pd.read_csv(root / 'results/routerbench/main/summary_main.csv')
methods = ['exc_empbernstein', 'norm_betting', 'cdf_empbernstein', 'cdf_pacbayes']
rows = []
for p, n in [(0.95, 3640), (0.99, 13676)]:
    for method in methods:
        part = d[np.isclose(d.p, p) & np.isclose(d.eps, 1) & (d.K == 32)
                 & (d.n == n) & (d.method == method)]
        if len(part) != 1:
            raise ValueError(f'Expected one summary for {(p,n,method)}')
        row = part.iloc[0]
        rows.append(dict(p=p, n=n, method=method, quality=row.V_deploy,
                         mc_standard_error=row.V_deploy_sd / np.sqrt(row.reps),
                         cert_rate=row.cert_rate, false_rate=row.false_rate,
                         repetitions=int(row.reps)))
result = pd.DataFrame(rows)
print(result.to_string(index=False))
out = root / 'figures'
out.mkdir(exist_ok=True)
result.to_csv(out / 'fixed_budget_summary.csv', index=False)
