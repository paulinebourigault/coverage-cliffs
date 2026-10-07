# Coverage Cliffs in Off-Policy Quantile Certification

Code and archived summaries for the paper's experiments, figures and tables, and `tailcert`,
a small NumPy-based package implementing the certificates for reuse. Fresh-generation measurements
are supplied as summaries and per-request records; rerunning generations requires external model and dataset assets and
need not reproduce exact values. The prompts (`results/reasoning/prompts.parquet`, with partitions and loaded sources) and the per-request held-out generations (`results/reasoning/deploy/deploy_generations.parquet`) are bundled; `python analysis/deploy_from_records.py` recomputes every held-out measurement. Immutable model revisions and the inference-software versions were not recorded.

## Layout

| Directory | Contents | Paper |
|---|---|---|
| `tailcert/` | Reusable package: `certify`, `select_and_certify`, `tail_overlap`, `required_n`, `tail_aware_logging`; tests | Sec. 4; Apps. C--E |
| `common/certs.py` | Shared certificate implementations (Exc-PB, Exc-Bet, Exc-EB, Norm-Bet, CDF-PB, CDF-EB, stratified Exc-Bet) | Secs. 3--4; App. D |
| `sims/` | Information removal and recovery (`sim_information.py`), shared-action affine routers (`sim_shared_actions.py`), rare-action boundary and oracle ceiling (`sim_tail_boundary.py`), PAC-Bayes-kl variant | Fig. "Information removal"; App. "Paired information study and shared actions", "Simulation details" |
| `routerbench/` | Constructed-logging replay on the RouterBench outcome table (5 LLMs) | Sec. 5, Tables 1--2; App. "RouterBench certification curves" |
| `reasoning/` | Reasoning-budget study: generation with 7 open-weight configurations (GPU), replay and held-out benchmark evaluation | Sec. 5; App. "Reasoning-budget study: outcomes and certificates" (see [reasoning/README.md](reasoning/README.md)) |
| `analysis/` | Predicted vs. observed log sizes, tail-aware logging, main-text figures | Fig. "Retrospective explanation"; Apps. "Predicted versus observed log sizes", "Tail-aware logging" |
| `tests/` | Count-based certificates equal the observation-level ones; validity of the normalization-aware test | — |
| `results/` | Archived experiment summaries and outcome tables used in the figures and tables | — |

Figures and derived tables are written to `figures/` (created on first use).

## Setup

```bash
pip install -r requirements.txt          # CPU parts
pip install -e tailcert && python tailcert/tests/test_tailcert.py && python tests/test_count_certificates.py && python tests/test_stratified_certificate.py
```

The RouterBench outcome table (`routerbench_0shot.pkl`, Hugging Face dataset `withmartian/routerbench`)
is downloaded automatically on first use. The reasoning-study outcome table (14,000 problems × 7
configurations) is shipped in `results/reasoning/table/`, so everything except new generations runs on CPU.

## Reproducing the paper from shipped results (minutes, CPU)

```bash
python sims/plot_information.py                         # information removal and recovery figure
python analysis/fixed_budget_table.py                   # main fixed-budget deployment table
python sims/plot_tail_boundary.py                       # rare-action boundary figure and the n/n* summary
python analysis/make_main_figures.py                    # retrospective prediction figure and optional routing overview
python routerbench/make_figures.py                      # appendix RouterBench figure and table
python analysis/predict_vs_observed.py --dataset routerbench
python analysis/predict_vs_observed.py --dataset reasoning
cd reasoning && python certify.py --run-dir runs/open --table-dir ../results/reasoning/table --deploy-only
#   -> held-out benchmark certificate: stratified Exc-Bet certifies router j=5 with bound 0.0968
```

## Rerunning the experiments

| Experiment | Command | Hardware (approximate) |
|---|---|---|
| Information study, shared actions | `python sims/sim_information.py` ; `python sims/sim_shared_actions.py` | 16 CPU cores, < 15 min |
| Exact reconstruction check | `python analysis/check_reconstruction.py` | 1 CPU core, 1 min |
| Rare-action boundary | `python sims/sim_tail_boundary.py` ; `python sims/sim_kl_variant.py` | 32 CPU cores, a few hours |
| RouterBench replay | `python routerbench/routerbench_replay.py --preset main` ; `--preset stress` | 32 CPU cores, ~2 h each |
| Reasoning replay | `cd reasoning && python certify.py --run-dir runs/open --table-dir ../results/reasoning/table` | 16 CPU cores, ~1 h |
| Reasoning generation and deployment | see [reasoning/README.md](reasoning/README.md) | one 80–94 GB GPU, ~17 GPU-hours |
| Tail-aware logging | `python analysis/tail_aware_logging.py` ; `python analysis/tail_aware_reasoning.py` | 8–10 CPU cores, < 1 h |

All runs use fixed seeds. Certificates are computed at δ = 0.05.

## Using the certificates

The package name `tailcert` covers the exceedance implementations: its default is Exc-Bet, with Exc-EB and Exc-PB available through the method argument. The paper's TailCert direct CDF-PB construction is implemented for fixed candidates as `common.certs.cdf_pb`; it returns the complemented bound in exceedance units.

```python
import tailcert as tc
tc.certify(w, y, tau=2000, p=0.90, W=20)                               # one fixed target policy
tc.select_and_certify(w_K, y, tau=2000, p=0.90, utility=acc_K, W=20)   # best certified of K candidates
tc.required_n(tc.tail_overlap(w_pilot, y_pilot, 2000), gamma=0.02, W=20, K=32)   # plan the log size
```

Requirements for validity: known logging propensities, a known almost-sure bound `W` on the importance weights (required explicitly; not an observed maximum),
and candidates, threshold and quantile level fixed before the log is collected.

## Assets and provenance

See `ASSETS.md` for creator citations, configured source identifiers, verified publisher license declarations, and explicitly unrecorded provenance. The code license below does not replace third-party terms.

## License

MIT.
