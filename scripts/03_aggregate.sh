#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASETS="${DATASETS:-all}"
METHODS="${METHODS:-all}"
SEEDS="${SEEDS:-42 123 456}"
CONFIG="${CONFIG:-config.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

cd "$PAPER_REPO"
cmd=(uv run paper-bench
  --config "$CONFIG" \
  --stage aggregate \
  --datasets "$DATASETS" \
  --methods "$METHODS" \
  --seeds $SEEDS)
if [[ -n "$OUTPUT_ROOT" ]]; then
  cmd+=(--output-root "$OUTPUT_ROOT")
fi
cmd+=("$@")
"${cmd[@]}"
