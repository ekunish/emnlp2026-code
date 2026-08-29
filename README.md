# Training Lightweight PII Text Detectors via Rubric-Guided Contrastive Data Synthesis

Official code for the EMNLP 2026 Industry Track paper
**“Training Lightweight PII Text Detectors via Rubric-Guided Contrastive Data
Synthesis.”**

The pipeline uses a small labeled seed pool to synthesize English
Sensitive/Non-sensitive sentences with an offline LLM, then trains
`roberta-base` (approximately 125M parameters) for sentence-level inference.
No LLM call is required when the trained detector is deployed.

## What is included

- the proposed rubric-guided contrastive synthesis method;
- duplicate, paraphrase, one-shot, and few-shot synthesis baselines;
- the camera-ready fixed-task Self-Instruct adaptation;
- RoBERTa training and evaluation.

Benchmark text, generated corpora, result tables, item-level predictions,
induced rubrics, human ratings, model checkpoints, and credentials are not
distributed. The paper and its appendix report the experimental results. See
[data/README.md](data/README.md) for the benchmark-data requirements.

## Proposed method

The public method name is `rubric_guided_contrastive_synthesis`. For each seed
pool, it:

1. samples five Sensitive and five Non-sensitive examples;
2. induces 25 dataset-specific label-decision rubrics from 3+3 subsets;
3. generates minimally different Sensitive/Non-sensitive pairs using the full
   5+5 pool and one rubric;
4. applies a separate LLM verifier;
5. rejects normalized exact matches and candidates with ROUGE-L F1 at least
   0.95 against the seeds or previously accepted generations;
6. trains RoBERTa on the accepted sentences.

The main experiment targets 2,000 balanced examples per run. This common count
was fixed before the main comparison to balance generation time and classifier
training data while keeping methods, datasets, and generators comparable; it is
not claimed to be a per-dataset optimum.

## Matched Self-Instruct adaptation

The camera-ready baseline is `self_instruct_faithful_neutral`. It adapts
Self-Instruct to one fixed binary classification task without adding the
proposed method's paired generation:

1. initialize a shared pool with six generic human-written privacy-decision
   instructions that are identical across datasets;
2. prompt with all six seed instructions and up to two accepted model-generated
   instructions;
3. accept a new instruction only when its ROUGE-L F1 is below 0.7 against every
   instruction already in the pool, then reuse it in later iterations;
4. select the target label before generation and independently synthesize one
   sentence for that label from a sampled pooled instruction and the same 5+5
   labeled seed examples;
5. use the same verifier, normalized exact-match filter, instance-level ROUGE-L
   threshold of 0.95, generators, sampling settings, and 2,000-example target as
   the proposed method.

Generated sentences are not reused as demonstrations. The full prompts are in
`src/pii_bench/self_instruct_faithful_neutral.py`.

## Installation

Python 3.10 or later and [`uv`](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync
```

For local vLLM serving:

```bash
uv sync --extra serve
```

Start an OpenAI-compatible endpoint in a separate terminal. For example:

```bash
uv run vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

The default endpoint is `http://localhost:8000/v1`. Override it with
`VLLM_URL` or `--vllm-url`.

## Data layout

Place licensed copies of the processed benchmarks under `data/`:

```text
data/spedac1/{train,valid,test}.jsonl
data/spy/{train,valid,test}.jsonl
```

Each row must contain `text` and one of the exact labels `Sensitive` or
`Non-sensitive`:

```json
{"text": "...", "label": "Sensitive"}
```

The paper evaluates English sentence-level classification. SPY preparation and
licensing remain the responsibility of the user; benchmark data are not
redistributed here.

## Reproduce an experiment

Run from the repository root.

```bash
bash scripts/00_prepare.sh

DATASETS=spedac1,spy \
METHODS=rubric_guided_contrastive_synthesis,self_instruct_faithful_neutral \
N_SAMPLES=2000 \
CONCURRENCY=4 \
bash scripts/01_generate.sh

# Stop vLLM first if RoBERTa needs the same GPUs.
DATASETS=spedac1,spy \
METHODS=rubric_guided_contrastive_synthesis,self_instruct_faithful_neutral \
bash scripts/02_train_eval_2gpu.sh

DATASETS=spedac1,spy \
METHODS=rubric_guided_contrastive_synthesis,self_instruct_faithful_neutral \
bash scripts/03_aggregate.sh
```

All classifier-training methods can be selected with:

```text
official_train,duplicate,oneshot,fewshot,paraphrase,
self_instruct_faithful_neutral,rubric_guided_contrastive_synthesis
```

Dry-run the execution plan before starting generation:

```bash
uv run paper-bench \
  --stage generate \
  --datasets spedac1 \
  --methods rubric_guided_contrastive_synthesis,self_instruct_faithful_neutral \
  --dry-run
```

## Development checks

```bash
uv run ruff check .
uv run pytest
```

## License

Code is released under the Apache License 2.0. Upstream benchmark data remain
subject to their original terms.
