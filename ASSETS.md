# Empirical assets and provenance

## Run records

- **MATH and GSM8K sources.** The original `prompts.parquet` is now bundled as `results/reasoning/prompts.parquet` (the unused `mock_difficulty` column removed). Its `source` column shows that every MATH problem was loaded from `EleutherAI/hendrycks_math` (3,750 test and 5,643 train problems) and every GSM8K problem from `openai/gsm8k` (979 test, 5,628 train). Dataset revision hashes were not recorded.
- **RouterBench file identity.** The `routerbench_0shot.pkl` used for every RouterBench run has SHA-256 `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d`, identical to the publisher's committed file below. The dataset-license question is unchanged.
- **Held-out generation records.** `results/reasoning/deploy/deploy_generations.parquet` holds the 8,000 fresh generations (2,000 held-out problems x 4 routers): response text, extracted answer, grade, finish reason and token counts. `analysis/deploy_from_records.py` reproduces every held-out measurement in `deploy_report.json` and the paper exactly.
- **Hardware.** Generation ran on one NVIDIA H100 NVL (94 GB) on a rented cloud instance; simulations and replays ran on rented cloud CPU instances. Versions of vLLM, Transformers, PyTorch and CUDA were not recorded.

Model and dataset revision hashes and the inference-software versions were not recorded.


These notes identify the assets configured for the experiments and report their publishers' stated licenses. They do not determine rights beyond those statements. The supplement's own `LICENSE` does not replace third-party terms.

| Asset and creator | Use in this study | Primary source and stated license |
|---|---|---|
| RouterBench — Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer and Shriyash Kaustubh Upadhyay | `withmartian/routerbench`, file `routerbench_0shot.pkl`; five-model outcome-table replay | [Paper](https://arxiv.org/abs/2403.12031), [official dataset card](https://huggingface.co/datasets/withmartian/routerbench/blob/main/README.md), [code LICENSE](https://github.com/withmartian/routerbench/blob/main/LICENSE). The code repository declares MIT. The reviewed official dataset card does **not** state a dataset license; a separate license declaration for the outcome table remains unconfirmed. |
| GSM8K — Karl Cobbe and coauthors / OpenAI | `openai/gsm8k`, `main` configuration, train and test splits | [Paper](https://arxiv.org/abs/2110.14168), [official dataset card](https://huggingface.co/datasets/openai/gsm8k/blob/main/README.md). The card explicitly declares MIT for the dataset. |
| MATH — Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song and Jacob Steinhardt | Seven subject configurations; train and test splits; loader tries three repositories in order | [Paper](https://arxiv.org/abs/2103.03874), [original repository LICENSE](https://github.com/hendrycks/math/blob/main/LICENSE). The [EleutherAI mirror card](https://huggingface.co/datasets/EleutherAI/hendrycks_math/blob/main/README.md) and [Digital Learning mirror card](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval/blob/main/README.md) both declare MIT. The latter attributes the original authors and links the original license. |
| Qwen3 — Qwen Team | `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-30B-A3B`, each with thinking on and off | [Technical report](https://arxiv.org/abs/2505.09388). The official [1.7B card](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/README.md), [8B card](https://huggingface.co/Qwen/Qwen3-8B/blob/main/README.md) and [30B-A3B card](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/README.md) each declare Apache-2.0 and link a model-specific LICENSE. |
| DeepSeek-R1-Distill-Qwen-7B — DeepSeek-AI | Seventh generation configuration | [Paper](https://arxiv.org/abs/2501.12948), [official model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/blob/main/README.md). The card declares MIT for the code and weights and separately notes that this model derives from the Apache-2.0 Qwen2.5 series. |
| vLLM — Woosuk Kwon and coauthors; vLLM contributors | Model inference engine | [Paper](https://arxiv.org/abs/2309.06180), [repository](https://github.com/vllm-project/vllm), [LICENSE](https://github.com/vllm-project/vllm/blob/main/LICENSE). Apache-2.0. |

## What the supplied run records establish

`reasoning/configs/open.yaml` records exact model names, sampling settings, the generation cap, dataset split choices and seed. `results/reasoning/prompts_meta.json` records the partition counts and digest `a75452c964182321`. The outcome table contains 14,000 problems; the feature table also represents the 2,000 held-out problems. These records do not pin model or dataset revision hashes.

The two supplied Parquet files contain embedded writer metadata identifying `pyarrow 25.0.1` and `pandas 3.0.6`. This establishes the libraries recorded when these tables were written, not the versions used for model inference, prompt loading, grading, or every analysis step. The unpinned lower bounds in `requirements.txt` are installation requirements, not a record of the original environment.

| Supplied artifact | SHA-256 |
|---|---|
| `results/reasoning/table/table.parquet` | `1cd4716313dc1cacc4abbc0094c7535c64787ec64205f2fc86d4bf0cc73a8f41` |
| `results/reasoning/table/features.parquet` | `bb2c33b21a62861fa011c327e49f2b27bac911eab226589d696ee0fedbabc5c0` |

The configured MATH loader tries `EleutherAI/hendrycks_math`, `DigitalLearningGmbH/MATH-lighteval`, then `hendrycks/competition_math`; the `source` column of `results/reasoning/prompts.parquet` shows that the first, `EleutherAI/hendrycks_math`, was used for every MATH problem.

## RouterBench source verification

The publisher's [file history](https://huggingface.co/datasets/withmartian/routerbench/commits/main/routerbench_0shot.pkl) identifies the initial upload of `routerbench_0shot.pkl` on 4 March 2024, at commit [`a4dcf98b60f1faf85c572ee5f20cc0069ca0501a`](https://huggingface.co/datasets/withmartian/routerbench/commit/a4dcf98b60f1faf85c572ee5f20cc0069ca0501a). Its committed pointer records 99,567,659 bytes and SHA-256 `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d`, matching the current official file page. The local copy used for every RouterBench run has the same SHA-256.

We checked the official dataset card, file listing and card history, the code repository's README and LICENSE, and the original paper. No separate license declaration for the outcome table was located in those sources. The present notes therefore do not assert that the code repository's MIT license covers the separate dataset or every constituent benchmark prompt and model response. The raw RouterBench table is not redistributed in this supplement; the replay loader fetches it from the publisher.
