# Lightweight PII Detection from a Few Seeds via Rubric-Guided Contrastive Synthesis

This repository contains the reproduction code for the paper:

```text
Lightweight PII Detection from a Few Seeds via Rubric-Guided Contrastive Synthesis
```

The code builds a small seed pool from labeled training data, synthesizes new
training texts with an OpenAI-compatible vLLM server, trains a RoBERTa classifier
on the synthesized data, and evaluates it on the official test split.

No benchmark data or generated outputs are included in this repository.

## Method

The paper's proposed method is exposed as:

```text
rubric_guided_contrastive_synthesis
```

It uses the following fixed pipeline.

1. Sample a seed pool with 5 Sensitive and 5 Non-sensitive examples.
2. Build 25 support subsets, each with 3 Sensitive and 3 Non-sensitive examples.
3. Ask the LLM to induce one rubric from each subset.
4. Use the rubrics to generate matched Sensitive / Non-sensitive text pairs.
5. Reject exact copies, ROUGE-L near copies, malformed texts, and pairs rejected by a separate LLM verifier.
6. Train RoBERTa on accepted synthetic texts and evaluate on the test split.

Default synthesis parameters:

| Setting | Value |
|---|---:|
| rubric induction calls | 25 |
| rubrics per call | 1 |
| pair attempts per target pair | 5 |
| ROUGE-L reject threshold | 0.95 |
| generation temperature | 1.0 |
| generation top_p | 0.9 |
| generation top_k | unset |
| verifier temperature | 0.0 |

## Environment

Install dependencies with `uv`.

```bash
uv sync
```

For local vLLM serving, install the optional serving dependencies.

```bash
uv sync --extra serve
```

The default generator is `Qwen/Qwen3.5-9B`. Start an OpenAI-compatible vLLM
server in a separate terminal:

```bash
uv run vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

The benchmark CLI connects to `http://localhost:8000/v1` by default. Use
`VLLM_URL` or `--vllm-url` to point to a different OpenAI-compatible endpoint.

## Data

Prepare zip files for SPeDaC1 and SPY outside this repository, then unzip them
under `data/`.

Required layout:

```text
data/spedac1/train.jsonl
data/spedac1/valid.jsonl
data/spedac1/test.jsonl
data/spedac1/label_map.yaml        # optional

data/spy/train.jsonl
data/spy/valid.jsonl
data/spy/test.jsonl
data/spy/label_map.yaml            # optional
```

Each JSONL row must contain at least:

```json
{"text": "...", "label": "Sensitive"}
```

The labels must be exactly `Sensitive` or `Non-sensitive`.

## Reproduction

Run from the repository root.

### 1. Prepare seed pools

```bash
bash scripts/00_prepare.sh
```

This validates the data and writes reproducible seed pools to
`outputs/seed_pools/`.

### 2. Generate synthetic training data

Keep vLLM running while this step executes.

```bash
DATASETS=spedac1,spy \
METHODS=rubric_guided_contrastive_synthesis \
N_SAMPLES=2000 \
CONCURRENCY=4 \
bash scripts/01_generate.sh
```

To use a different model or endpoint:

```bash
MODEL=Qwen/Qwen3.5-9B \
VLLM_URL=http://localhost:8000/v1 \
DATASETS=spedac1,spy \
METHODS=rubric_guided_contrastive_synthesis \
N_SAMPLES=2000 \
bash scripts/01_generate.sh
```

### 3. Train and evaluate RoBERTa

Stop vLLM first if it shares the same GPUs used for RoBERTa training.

Single GPU:

```bash
DATASETS=spedac1,spy \
METHODS=rubric_guided_contrastive_synthesis \
bash scripts/02_train_eval.sh
```

Two GPUs:

```bash
DATASETS=spedac1,spy \
METHODS=rubric_guided_contrastive_synthesis \
bash scripts/02_train_eval_2gpu.sh
```

### 4. Aggregate results

```bash
DATASETS=spedac1,spy \
METHODS=rubric_guided_contrastive_synthesis \
bash scripts/03_aggregate.sh
```

The aggregate files are written under the selected output root:

```text
outputs/runs/summary.csv
outputs/runs/summary.json
outputs/runs/summary.md
```

## Baselines

The following training-data baselines are included:

```text
official_train
duplicate
oneshot
fewshot
paraphrase
```

Run all classifier-training methods:

```bash
DATASETS=spedac1,spy \
METHODS=official_train,duplicate,oneshot,fewshot,paraphrase,rubric_guided_contrastive_synthesis \
N_SAMPLES=2000 \
bash scripts/01_generate.sh

DATASETS=spedac1,spy \
METHODS=official_train,duplicate,oneshot,fewshot,paraphrase,rubric_guided_contrastive_synthesis \
bash scripts/02_train_eval_2gpu.sh

DATASETS=spedac1,spy \
METHODS=official_train,duplicate,oneshot,fewshot,paraphrase,rubric_guided_contrastive_synthesis \
bash scripts/03_aggregate.sh
```

`official_train` does not require synthetic generation, so it is skipped during
the generation stage.

## Useful CLI Examples

Dry-run the generation plan:

```bash
uv run paper-bench --stage generate --datasets spedac1 --methods rubric_guided_contrastive_synthesis --dry-run
```

Use a smaller smoke-test sample size:

```bash
DATASETS=spedac1 \
METHODS=rubric_guided_contrastive_synthesis \
N_SAMPLES=4 \
CONCURRENCY=1 \
ATTEMPT_CAP=40 \
bash scripts/01_generate.sh
```

Use a custom output root:

```bash
OUTPUT_ROOT=outputs/qwen_n2000 \
DATASETS=spedac1,spy \
METHODS=rubric_guided_contrastive_synthesis \
bash scripts/01_generate.sh
```

## Anonymity and Release Hygiene

Before publishing through Anonymous GitHub, run:

```bash
uv run python scripts/check_anonymity.py
```

The check scans release-facing code and documentation for local absolute paths,
private IP literals, common personal email patterns, and local user names. It
does not scan ignored benchmark data or generated outputs.

Also inspect the staged files before pushing:

```bash
git status --short --ignored
git diff --cached --stat
```

Expected ignored paths include `data/spedac1/`, `data/spy/`, `outputs/`, and
`.venv/`.
