#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASETS="${DATASETS:-all}"
METHODS="${METHODS:-all}"
SEEDS="${SEEDS:-42 123 456}"
CONFIG="${CONFIG:-config.yaml}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
BACKBONE="${BACKBONE:-roberta-base}"
LR="${LR:-1.0e-5}"
DROPOUT="${DROPOUT:-0.3}"
MAX_LEN="${MAX_LEN:-256}"
EVAL_VALID_SIZE="${EVAL_VALID_SIZE:-2000}"
EVAL_TEST_SIZE="${EVAL_TEST_SIZE:-0}"
EVAL_SUBSAMPLE_SEED="${EVAL_SUBSAMPLE_SEED:-0}"
PATIENCE="${PATIENCE:-0}"
MONITOR="${MONITOR:-valid_acc}"
CALIBRATE_THRESHOLD="${CALIBRATE_THRESHOLD:-0}"

timestamp="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="${LOG_DIR:-$PAPER_REPO/outputs/logs/train_eval_2gpu/$timestamp}"
PLAN_PATH="$LOG_DIR/plan.tsv"
mkdir -p "$LOG_DIR"

cd "$PAPER_REPO"

CONFIG="$CONFIG" DATASETS="$DATASETS" METHODS="$METHODS" SEEDS="$SEEDS" PLAN_PATH="$PLAN_PATH" \
  uv run python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from pii_bench.config import load_config


def parse_csv(value: str, allowed: list[str]) -> list[str]:
    allowed = list(dict.fromkeys(allowed))
    if value == "all":
        return allowed
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - set(allowed))
    if unknown:
        raise SystemExit(f"Unknown values {unknown}; allowed: {allowed}")
    return items


config = load_config(os.environ["CONFIG"])
datasets = parse_csv(os.environ["DATASETS"], list(config.datasets))
methods = parse_csv(
    os.environ["METHODS"],
    [*config.train_methods, *sorted(config.raw.get("custom_methods", {}))],
)
seeds = [int(seed) for seed in os.environ["SEEDS"].split()]

tasks: list[tuple[str, str, int]] = []
if "official_train" in methods:
    for dataset in datasets:
        for seed in seeds:
            tasks.append((dataset, "official_train", seed))

for dataset in datasets:
    for method in methods:
        if method == "official_train":
            continue
        for seed in seeds:
            tasks.append((dataset, method, seed))

plan_path = Path(os.environ["PLAN_PATH"])
with plan_path.open("w", encoding="utf-8") as handle:
    handle.write("gpu\tdataset\tmethod\tseed\n")
    for index, (dataset, method, seed) in enumerate(tasks):
        handle.write(f"{index % 2}\t{dataset}\t{method}\t{seed}\n")

print(f"[plan] wrote {plan_path} ({len(tasks)} runs)")
PY

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  echo "[plan-only] $PLAN_PATH"
  exit 0
fi

run_gpu() {
  local gpu="$1"
  local log_path="$LOG_DIR/gpu${gpu}.log"
  {
    echo "[gpu${gpu}] start"
    while IFS=$'\t' read -r assigned_gpu dataset method seed; do
      if [[ "$assigned_gpu" == "gpu" || "$assigned_gpu" != "$gpu" ]]; then
        continue
      fi
      echo "[gpu${gpu}] dataset=${dataset} method=${method} seed=${seed}"
      env_args=(
        CUDA_VISIBLE_DEVICES="$gpu"
        CONFIG="$CONFIG"
        DATASETS="$dataset"
        METHODS="$method"
        SEEDS="$seed"
        EPOCHS="$EPOCHS"
        BATCH_SIZE="$BATCH_SIZE"
        BACKBONE="$BACKBONE"
        LR="$LR"
        DROPOUT="$DROPOUT"
        MAX_LEN="$MAX_LEN"
        PATIENCE="$PATIENCE"
        MONITOR="$MONITOR"
        CALIBRATE_THRESHOLD="$CALIBRATE_THRESHOLD"
        EVAL_VALID_SIZE="$EVAL_VALID_SIZE"
        EVAL_TEST_SIZE="$EVAL_TEST_SIZE"
        EVAL_SUBSAMPLE_SEED="$EVAL_SUBSAMPLE_SEED"
      )
      if [[ -n "$OUTPUT_ROOT" ]]; then
        env_args+=(OUTPUT_ROOT="$OUTPUT_ROOT")
      fi
      env "${env_args[@]}" bash "$SCRIPT_DIR/02_train_eval.sh"
    done < "$PLAN_PATH"
    echo "[gpu${gpu}] done"
  } 2>&1 | tee "$log_path"
}

echo "[start] logs=$LOG_DIR"
run_gpu 0 &
pid0=$!
run_gpu 1 &
pid1=$!

status0=0
status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?

echo "[done] gpu0_status=$status0 gpu1_status=$status1 logs=$LOG_DIR"
if (( status0 != 0 || status1 != 0 )); then
  exit 1
fi
