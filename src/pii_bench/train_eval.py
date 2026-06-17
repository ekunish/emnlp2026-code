"""Training and result checks for paper benchmark reproduction."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .config import DatasetConfig
from .utils import repo_root


def result_path(run_dir: Path) -> Path:
    return run_dir / "results" / "metrics.json"


def _remove_model_artifacts(results_dir: Path) -> None:
    for pattern in [
        "*.safetensors",
        "*.bin",
        "*.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]:
        for path in results_dir.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


def _annotate_metrics(path: Path, *, method: str, seed: int, train_data: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = raw.setdefault("config", {})
    config["method"] = method
    config["run_type"] = "trained_classifier"
    config["seed"] = seed
    config["train_data"] = str(train_data)
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train_and_evaluate(
    *,
    dataset: DatasetConfig,
    method: str,
    train_data: Path,
    run_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    max_len: int,
    backbone: str,
    patience: int,
    eval_valid_size: int,
    eval_test_size: int,
    eval_subsample_seed: int,
    monitor: str,
    calibrate_threshold: bool,
    keep_predictions: bool,
    keep_model_weights: bool,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    results_dir = run_dir / "results"
    result = result_path(run_dir)
    if result.exists() and not overwrite:
        print(f"[skip] trained: {result}")
        return

    train_script = repo_root() / "src" / "pii_bench" / "trainer.py"
    cmd = [
        "uv",
        "run",
        "python",
        str(train_script),
        "--train-data",
        str(train_data),
        "--valid-data",
        str(dataset.valid),
        "--in-test",
        str(dataset.test),
        "--label-map",
        str(dataset.label_map),
        "--output-dir",
        str(results_dir),
        "--backbone",
        backbone,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--dropout",
        str(dropout),
        "--max-len",
        str(max_len),
        "--lambda",
        "0",
        "--patience",
        str(patience),
        "--seed",
        str(seed),
        "--eval-valid-size",
        str(eval_valid_size),
        "--eval-test-size",
        str(eval_test_size),
        "--eval-subsample-seed",
        str(eval_subsample_seed),
        "--monitor",
        monitor,
    ]
    if calibrate_threshold:
        cmd.append("--calibrate-threshold")
    if keep_predictions:
        cmd.append("--save-predictions")
    if keep_model_weights:
        cmd.append("--save-best")

    print("[train]", " ".join(cmd))
    if dry_run:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "train_command.json").open("w", encoding="utf-8") as f:
        json.dump({"cmd": cmd}, f, indent=2)
    subprocess.run(cmd, cwd=repo_root(), check=True)  # noqa: S603
    _annotate_metrics(result, method=method, seed=seed, train_data=train_data)
    if not keep_model_weights:
        _remove_model_artifacts(results_dir)


def check_evaluation(run_dir: Path) -> None:
    result = result_path(run_dir)
    if not result.exists():
        raise FileNotFoundError(f"Missing result file: {result}")
    raw = json.loads(result.read_text(encoding="utf-8"))
    best = raw.get("best", {})
    if "in_test_acc" not in best:
        raise ValueError(f"Missing best.in_test_acc in {result}")
    print(f"[eval] {run_dir}: acc={best['in_test_acc']:.4f} f1={best.get('in_test_f1')}")
