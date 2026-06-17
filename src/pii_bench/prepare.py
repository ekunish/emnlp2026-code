"""Validate benchmark splits and extract reproducible seed pools."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import yaml

from .config import LABELS, BenchmarkConfig, DatasetConfig
from .utils import read_jsonl, write_jsonl


def validate_items(items: Iterable[dict], path: Path) -> list[dict]:
    valid_items: list[dict] = []
    for line_no, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        text = item.get("text")
        label = item.get("label")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_no}: missing non-empty text")
        if label not in LABELS:
            raise ValueError(f"{path}:{line_no}: unknown label {label!r}; expected {LABELS}")
        copied = dict(item)
        copied["text"] = text.strip()
        copied["label"] = str(label)
        valid_items.append(copied)
    if not valid_items:
        raise ValueError(f"{path}: no JSONL items found")
    return valid_items


def label_counts(items: list[dict]) -> dict[str, int]:
    counts = Counter(str(item["label"]) for item in items)
    return {label: counts.get(label, 0) for label in LABELS}


def load_and_validate(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required benchmark split: {path}")
    return validate_items(read_jsonl(path), path)


def write_label_map(dataset: DatasetConfig) -> None:
    dataset.label_map.parent.mkdir(parents=True, exist_ok=True)
    if dataset.source_label_map.exists():
        raw = yaml.safe_load(dataset.source_label_map.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{dataset.source_label_map}: expected mapping label -> id")
        label_map = raw
    else:
        label_map = {"Non-sensitive": 0, "Sensitive": 1}
    dataset.label_map.write_text(yaml.safe_dump(label_map, sort_keys=False), encoding="utf-8")


def extract_seed_pool(
    *,
    train_items: list[dict],
    dataset: DatasetConfig,
    seed: int,
    per_label: int,
    overwrite: bool,
) -> dict:
    pool_dir = dataset.seed_pool(seed, per_label)
    meta_path = pool_dir / "seed_pool_meta.json"
    if meta_path.exists() and not overwrite:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    by_label = {label: [item for item in train_items if item["label"] == label] for label in LABELS}
    for label, items in by_label.items():
        if len(items) < per_label:
            raise ValueError(
                f"{dataset.name}: not enough {label} train items for seed pool: "
                f"{len(items)} < {per_label}"
            )

    rng = random.Random(seed)
    pool_dir.mkdir(parents=True, exist_ok=True)
    chosen_by_label = {}
    for label in LABELS:
        chosen = rng.sample(by_label[label], per_label)
        chosen_by_label[label] = chosen
        write_jsonl(chosen, pool_dir / f"{label}.jsonl")

    meta = {
        "dataset": dataset.name,
        "seed": seed,
        "per_label": per_label,
        "total": per_label * len(LABELS),
        "source_train": str(dataset.train),
        "label_counts": {label: len(items) for label, items in chosen_by_label.items()},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def prepare_dataset(
    *,
    config: BenchmarkConfig,
    dataset: DatasetConfig,
    overwrite: bool = False,
) -> dict:
    train = load_and_validate(dataset.train)
    valid = load_and_validate(dataset.valid)
    test = load_and_validate(dataset.test)

    write_label_map(dataset)

    pools = [
        extract_seed_pool(
            train_items=train,
            dataset=dataset,
            seed=seed,
            per_label=config.seed_pool_per_label,
            overwrite=overwrite,
        )
        for seed in config.seeds
    ]

    meta = {
        "dataset": dataset.name,
        "display_name": dataset.display_name,
        "split_dir": str(dataset.split_dir),
        "seed_pool_root": str(dataset.seed_pool_root),
        "label_map": str(dataset.label_map),
        "splits": {
            "train": {"n": len(train), "label_counts": label_counts(train)},
            "valid": {"n": len(valid), "label_counts": label_counts(valid)},
            "test": {"n": len(test), "label_counts": label_counts(test)},
        },
        "seed_pools": pools,
    }
    dataset.prepare_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return meta


def prepare_all(
    *,
    config: BenchmarkConfig,
    datasets: list[str],
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    for dataset_name in datasets:
        dataset = config.datasets[dataset_name]
        print(f"[prepare] dataset={dataset_name} splits={dataset.split_dir}")
        if dry_run:
            continue
        prepare_dataset(config=config, dataset=dataset, overwrite=overwrite)
