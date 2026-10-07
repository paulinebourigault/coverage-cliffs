# Reasoning-budget study

Code for the paper's reasoning-budget replay and held-out benchmark generation study
(Section 5; Appendix "Reasoning-budget study: outcomes and certificates").

A router picks one of seven open-weight reasoning configurations for each math problem. The
outcome Y is the number of generated tokens (thinking included), and the budget constraint is
P(Y > tau) <= 1 - p. We build a full outcome table (every configuration answers every problem
once), replay constructed logging policies on it, select routers from the logs and certify their
token budgets with the certificates in `../common/certs.py`. Finally, selected routers are evaluated with fresh generations on 2,000 held-out benchmark problems.

| action | model (Hugging Face id) | mode |
|---|---|---|
| `q1.7b_nothink` | `Qwen/Qwen3-1.7B` | `enable_thinking=False`, T=0.7, top_p=0.8, top_k=20 |
| `q1.7b_think` | `Qwen/Qwen3-1.7B` | `enable_thinking=True`, T=0.6, top_p=0.95, top_k=20 |
| `q8b_nothink` | `Qwen/Qwen3-8B` | as above (reference router: always this action) |
| `q8b_think` | `Qwen/Qwen3-8B` | as above |
| `q30b_nothink` | `Qwen/Qwen3-30B-A3B` | as above |
| `q30b_think` | `Qwen/Qwen3-30B-A3B` | as above |
| `r1d_qwen7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | T=0.6, top_p=0.95 |

All settings are in `configs/open.yaml`: generation cap 8,192 tokens (truncated requests form an
atom of Y at the cap; a reasoning trace truncated before `</think>` has no answer and is graded
incorrect), seed, prompt template, dataset sources and partition sizes.

**Prompts.** GSM8K (train + test) and MATH (all 7 subjects, train + test), deduplicated, with a
proportional subsample of 16,000 problems split, stratified by dataset, into 4,200 dev, 9,800 eval
(together the 14,000-problem outcome table) and 2,000 deployment problems that are only used by
`deploy_verify.py`. **Grading.** GSM8K: final number (`\boxed{}` first, else the last number).
MATH: last `\boxed{}`, compared with `math_verify`.

## Pipeline

| step | script | hardware | output (in `runs/open/`) |
|---|---|---|---|
| 0 | `prepare_prompts.py` | CPU | `prompts.parquet`, `prompts_meta.json` (fixed dev / eval / deploy partition) |
| 1 | `generate.py` (+ `gen_worker.py`) | GPU, vLLM | `gen/shard*/part-*.parquet` (one sample per problem and action; resumable) |
| 2 | `build_table.py` | CPU | `table/table.parquet`, `table/features.parquet`, `table/audit.json` |
| 3 | `certify.py` | CPU | `certify/` (replay study, stress test, deploy plan) |
| 4 | `deploy_verify.py` | GPU, vLLM | `deploy/deploy_report.json` (fresh generations on held-out benchmark problems) |
| 5 | `make_figures.py` | CPU | `figures/` (diagnostic plots, not in the paper) |

`grlib.py` holds the shared code (config, grading, request runner). Every script takes
`--config` (default `configs/open.yaml`) and `--run-dir` (default `runs/<run_name>` next to the
scripts, i.e. `runs/open/`).

## Reproducing the paper's numbers

### Without a GPU (from the shipped outcome table)

The outcome table and router features produced by steps 0-2 are shipped in
`../results/reasoning/table/`. `certify.py` reads them with `--table-dir`:

```bash
cd reasoning
# replay study, stress test and deploy plan (about 10-25 min on 16 cores for 1000 repetitions)
python certify.py --table-dir ../results/reasoning/table --reps 1000 --procs 16
# deploy plan only (about 1 min)
python certify.py --table-dir ../results/reasoning/table --deploy-only
# diagnostic figures from the outputs above
python make_figures.py --table-dir ../results/reasoning/table
```

Outputs go to `runs/open/certify/` (or `--run-dir`); the paper's outputs are shipped in
`../results/reasoning/certify/` and `../results/reasoning/deploy/` for comparison.

The defaults are the paper's settings: delta = 0.05; eps in {1, 0.5, 0.2}; K in {8, 32, 128};
quantile levels p in {0.90, 0.95, 0.99}, where the dev rule gives tau = 2,161 tokens at p = 0.90
and 3,509 at p = 0.95 and fails at p = 0.99 (that level is skipped with a warning); deploy plan at
p = 0.90, eps = 0.5, K = 32. Expected deploy-plan output:

* `certified` (unstratified Exc-PB diagnostic, one log of n = 9,800): no threshold crossing,
  falls back to the reference router; see the appendix on posterior certificates for Exc-PB;
* `certified_bet` (unstratified Exc-Bet diagnostic): router j = 5, value 0.0970. This does not
  supply a population certificate for the stratified single-pass log;
* stratified betting certificate (`deploy_stratified.json`, `bet_prod`): router j = 5, bound
  0.0968; the union-bound variant (`bet`) certifies nothing (smallest bound 0.111);
* `certified_10x` (Exc-PB on a 98,000-request log resampled from the eval pool): router j = 26,
  bound 0.062.

### Full pipeline

Requirements: Python >= 3.10, `pip install -r ../requirements.txt` plus
`pip install vllm datasets math-verify` (vLLM >= 0.8.5 and transformers >= 4.51 for Qwen3;
install vLLM into a fresh environment and let it pull its own PyTorch). Set `HF_HOME` to a disk
with about 100 GB free for the model weights; all models are public on the Hugging Face Hub.

```bash
cd reasoning
CFG=configs/open.yaml
python prepare_prompts.py --config $CFG                 # downloads GSM8K and MATH; prints the partition digest
python generate.py --config $CFG --dry-run              # optional: per-action output-token estimate
python generate.py --config $CFG                        # GPU; rerun the same command to resume
python build_table.py --config $CFG
python certify.py --config $CFG --reps 1000 --procs 16
python deploy_verify.py --config $CFG                   # GPU; routers certified, certified_10x, greedy, think_always
python make_figures.py --config $CFG
```

Generation can be split across GPUs with `--shard i --num-shards k` (one process per GPU, e.g.
`CUDA_VISIBLE_DEVICES=i`); `build_table.py` merges all shard directories. Useful `generate.py`
flags: `--limit N` (pilot on the first N problems), `--max-num-seqs`, `--gpu-memory-utilization`,
`--chunk-size` (checkpoint granularity), `--tensor-parallel-size`.

**Hardware and time.** The paper's run used one 80-94 GB GPU (Qwen3-30B-A3B needs about 61 GB of
bf16 weights). The outcome table has about 207 M generated tokens; generation took about 17
GPU-hours (measured, excluding model loading), and the held-out benchmark evaluation about
1.2 GPU-hours. vLLM sampling is not bit-reproducible across hardware and versions; regenerated
outcomes need not reproduce the shipped values. Model and dataset revision hashes are not preserved
in the released records (see `../ASSETS.md`).

## Outputs and the paper

| paper | file |
|---|---|
| Table "Outcome table" (accuracy, tokens, truncation per configuration) | `table/audit.json` (`per_action`), `certify/action_tails.csv` |
| Table "Logged requests needed to certify" (appendix "Reasoning-budget study: outcomes and certificates") | `certify/summary_main.csv` (`tag = main`, `K = 32`): log size at which `cert_rate` reaches 0.9, interpolated linearly in log n between grid points |
| False-certification rates (main study and stress test) | `certify/summary_main.csv`, `certify/summary_stress.csv` (`false_rate`, Wilson CIs) |
| tau values and the dev rule | `certify/meta.json` (`taus`, `tau_info`) |
| Overlaps V_tau, V_F (appendix "Predicted versus observed log sizes") | `certify/ground_truth_routers.csv` (`V_tau_p*_eps*`, `V_F_p*_eps*`); predicted vs observed log sizes: `../analysis/predict_vs_observed.py` |
| Held-out benchmark evaluation: certificates from one log | `certify/deploy_plan.json`, `deploy/deploy_stratified.json` |
| Held-out benchmark table (realized exceedance, quantile, accuracy) | `deploy/deploy_report.json` |

The stratified betting certificate (product of the per-dataset betting wealths, one confidence
budget) is the stratified version described in the appendix "The betting certificate"
(`certs.exc_bet_stratified`). The held-out table's row "Exc-Bet certified, stratified" reports the
fresh responses of the `greedy` router, which is the same router (j = 5) on this log.

The historical `deploy/deploy_report.json` preserves the labels and plan used when those responses
were collected. Its `greedy` measurements are reused because that deterministic router equals the
stratified-certified router. Current `certify/deploy_plan.json` distinguishes valid stratified
certification from unstratified diagnostics; no historical response measurements were changed.

## Notes

* The logging policy is constructed (pi0 = (1 - eps) pi_base + eps/M over the outcome table, with
  pi_base = always `q1.7b_nothink`), so propensities are known exactly.
* Replay logs larger than the 9,800-problem eval pool resample it with replacement; the
  corresponding certificates and false-certification rates refer to that finite pool.
* The table holds one sample per (problem, action); its "true" exceedances are pool averages of
  that sample. `deploy_verify.py` uses fresh problems and fresh samples.
