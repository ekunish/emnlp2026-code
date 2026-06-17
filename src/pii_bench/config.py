"""Configuration loading for public paper reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .utils import repo_root, resolve_repo_path

LABELS = ("Non-sensitive", "Sensitive")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display_name: str
    split_dir: Path
    seed_pool_root: Path

    @property
    def train(self) -> Path:
        return self.split_dir / "train.jsonl"

    @property
    def valid(self) -> Path:
        return self.split_dir / "valid.jsonl"

    @property
    def test(self) -> Path:
        return self.split_dir / "test.jsonl"

    @property
    def source_label_map(self) -> Path:
        return self.split_dir / "label_map.yaml"

    @property
    def label_map(self) -> Path:
        return self.seed_pool_root / "label_map.yaml"

    @property
    def prepare_meta(self) -> Path:
        return self.seed_pool_root / "prepare_meta.json"

    def seed_pool(self, seed: int, per_label: int) -> Path:
        return self.seed_pool_root / f"seed_{seed}" / f"pool_{per_label * 2}"


@dataclass(frozen=True)
class BenchmarkConfig:
    raw: dict[str, Any]
    seeds: list[int]
    paths: dict[str, Any]
    seed_pool: dict[str, Any]
    methods: dict[str, list[str]]
    generation: dict[str, Any]
    direct_judge: dict[str, Any]
    trainer: dict[str, Any]
    artifacts: dict[str, Any]
    datasets: dict[str, DatasetConfig]

    @property
    def seed_pool_per_label(self) -> int:
        return int(self.seed_pool.get("per_label", 10))

    @property
    def train_methods(self) -> list[str]:
        return list(self.methods.get("train", []))

    @property
    def direct_judge_methods(self) -> list[str]:
        return list(self.methods.get("direct_judge", []))

    @property
    def all_methods(self) -> list[str]:
        return [*self.train_methods, *self.direct_judge_methods]


def default_config_path() -> Path:
    return repo_root() / "config.yaml"


def _resolve_dataset_dir(base_path: str | Path, dataset_name: str) -> Path:
    return resolve_repo_path(Path(base_path) / dataset_name)


def load_config(path: str | Path | None = None) -> BenchmarkConfig:
    config_path = Path(path) if path else default_config_path()
    if not config_path.is_absolute():
        config_path = resolve_repo_path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    paths = raw.get("paths", {})
    dataset_dir = paths.get("dataset_dir", "data")
    seed_pool_dir = paths.get("seed_pool_dir", "outputs/seed_pools")

    datasets = {}
    for name, ds in raw.get("datasets", {}).items():
        datasets[name] = DatasetConfig(
            name=name,
            display_name=ds.get("display_name", name),
            split_dir=_resolve_dataset_dir(dataset_dir, name),
            seed_pool_root=_resolve_dataset_dir(seed_pool_dir, name),
        )

    return BenchmarkConfig(
        raw=raw,
        seeds=[int(seed) for seed in raw.get("seeds", [42, 123, 456])],
        paths=paths,
        seed_pool=raw.get("seed_pool", {}),
        methods=raw.get("methods", {}),
        generation=raw.get("generation", {}),
        direct_judge=raw.get("direct_judge", {}),
        trainer=raw.get("trainer", {}),
        artifacts=raw.get("artifacts", {}),
        datasets=datasets,
    )
