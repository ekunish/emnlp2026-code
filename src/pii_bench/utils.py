"""Small shared utilities."""

from __future__ import annotations

import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    items = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(items: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(items: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_seed_pool(pool_dir: Path) -> list[dict]:
    items: list[dict] = []
    for path in sorted(pool_dir.glob("*.jsonl")):
        items.extend(read_jsonl(path))
    if not items:
        raise FileNotFoundError(f"No JSONL seed examples found under {pool_dir}")
    return items


def percent(value: float | None) -> float | None:
    return None if value is None else value * 100.0
