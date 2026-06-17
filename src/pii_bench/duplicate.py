"""Simple duplicate baseline."""

from __future__ import annotations

import random
from pathlib import Path

from .utils import load_seed_pool, write_jsonl


def build_duplicate(seed_pool_dir: Path, out_path: Path, seed: int, n_samples: int = 2000) -> dict:
    seed_items = load_seed_pool(seed_pool_dir)
    duplicated = []
    for idx in range(n_samples):
        item = seed_items[idx % len(seed_items)]
        duplicated.append({"text": item["text"], "label": item["label"]})
    random.Random(seed).shuffle(duplicated)
    write_jsonl(duplicated, out_path)
    return {
        "method": "duplicate",
        "seed_pool": str(seed_pool_dir),
        "num_seed_items": len(seed_items),
        "target_samples": n_samples,
        "num_written": len(duplicated),
    }
