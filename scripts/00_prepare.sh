#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASETS="${DATASETS:-all}"
CONFIG="${CONFIG:-config.yaml}"

cd "$PAPER_REPO"
uv run paper-bench \
  --config "$CONFIG" \
  --stage prepare \
  --datasets "$DATASETS" \
  "$@"
