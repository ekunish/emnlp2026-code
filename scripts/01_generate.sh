#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASETS="${DATASETS:-all}"
METHODS="${METHODS:-oneshot,fewshot,duplicate,paraphrase}"
SEEDS="${SEEDS:-42 123 456}"
N_SAMPLES="${N_SAMPLES:-2000}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
CONFIG="${CONFIG:-config.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
CONCURRENCY="${CONCURRENCY:-1}"
ATTEMPT_CAP="${ATTEMPT_CAP:-20000}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-}"

cd "$PAPER_REPO"
cmd=(uv run paper-bench
  --config "$CONFIG" \
  --stage generate \
  --datasets "$DATASETS" \
  --methods "$METHODS" \
  --seeds $SEEDS \
  --n-samples "$N_SAMPLES" \
  --model "$MODEL" \
  --vllm-url "$VLLM_URL" \
  --concurrency "$CONCURRENCY" \
  --attempt-cap "$ATTEMPT_CAP")
if [[ -n "$OUTPUT_ROOT" ]]; then
  cmd+=(--output-root "$OUTPUT_ROOT")
fi
if [[ -n "$TEMPERATURE" ]]; then
  cmd+=(--temperature "$TEMPERATURE")
fi
if [[ -n "$TOP_P" ]]; then
  cmd+=(--top-p "$TOP_P")
fi
if [[ -n "$TOP_K" ]]; then
  cmd+=(--top-k "$TOP_K")
fi
if [[ -n "$PRESENCE_PENALTY" ]]; then
  cmd+=(--presence-penalty "$PRESENCE_PENALTY")
fi
cmd+=("$@")
"${cmd[@]}"
