#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASETS="${DATASETS:-all}"
METHODS="${METHODS:-all}"
SEEDS="${SEEDS:-42 123 456}"
CONFIG="${CONFIG:-config.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
BACKBONE="${BACKBONE:-roberta-base}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1.0e-5}"
DROPOUT="${DROPOUT:-0.3}"
MAX_LEN="${MAX_LEN:-256}"
EVAL_VALID_SIZE="${EVAL_VALID_SIZE:-2000}"
EVAL_TEST_SIZE="${EVAL_TEST_SIZE:-0}"
EVAL_SUBSAMPLE_SEED="${EVAL_SUBSAMPLE_SEED:-0}"
PATIENCE="${PATIENCE:-0}"
MONITOR="${MONITOR:-valid_acc}"
CALIBRATE_THRESHOLD="${CALIBRATE_THRESHOLD:-0}"

cd "$PAPER_REPO"
cmd=(uv run paper-bench
  --config "$CONFIG" \
  --stage train_eval \
  --datasets "$DATASETS" \
  --methods "$METHODS" \
  --seeds $SEEDS \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --dropout "$DROPOUT" \
  --max-len "$MAX_LEN" \
  --patience "$PATIENCE" \
  --monitor "$MONITOR" \
  --eval-valid-size "$EVAL_VALID_SIZE" \
  --eval-test-size "$EVAL_TEST_SIZE" \
  --eval-subsample-seed "$EVAL_SUBSAMPLE_SEED")
if [[ -n "$OUTPUT_ROOT" ]]; then
  cmd+=(--output-root "$OUTPUT_ROOT")
fi
if [[ "$CALIBRATE_THRESHOLD" == "1" || "$CALIBRATE_THRESHOLD" == "true" ]]; then
  cmd+=(--calibrate-threshold)
fi
cmd+=("$@")
"${cmd[@]}"
