"""Aggregate paper reproduction runs."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .config import BenchmarkConfig
from .direct_judge import DIRECT_JUDGE_METHODS
from .train_eval import result_path
from .utils import percent

DIRECT_SEED = 0


def _read_result(run_dir: Path) -> dict | None:
    path = result_path(run_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sd(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def _method_seeds(method: str, seeds: list[int]) -> list[int]:
    return [DIRECT_SEED] if method == "first_person_rule" else seeds


def _run_type(method: str) -> str:
    return "direct_judge" if method in DIRECT_JUDGE_METHODS else "trained_classifier"


def aggregate_results(
    *,
    config: BenchmarkConfig,
    output_root: Path,
    datasets: list[str],
    methods: list[str],
    seeds: list[int],
) -> None:
    rows: list[dict] = []
    for dataset_name in datasets:
        ds = config.datasets[dataset_name]
        for method in methods:
            accs = []
            f1s = []
            seen_seeds = []
            for seed in _method_seeds(method, seeds):
                run_dir = output_root / dataset_name / method / f"seed_{seed}"
                result = _read_result(run_dir)
                if not result:
                    continue
                best = result.get("best", {})
                if best.get("in_test_acc") is not None:
                    accs.append(percent(float(best["in_test_acc"])))
                    seen_seeds.append(seed)
                if best.get("in_test_f1") is not None:
                    f1s.append(percent(float(best["in_test_f1"])))
            rows.append(
                {
                    "dataset": dataset_name,
                    "display_name": ds.display_name,
                    "method": method,
                    "run_type": _run_type(method),
                    "seeds": " ".join(str(s) for s in seen_seeds),
                    "n": len(accs),
                    "accuracy_mean": statistics.mean(accs) if accs else None,
                    "accuracy_sd": _sd(accs),
                    "macro_f1_mean": statistics.mean(f1s) if f1s else None,
                    "macro_f1_sd": _sd(f1s),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "summary.csv"
    fieldnames = [
        "dataset",
        "display_name",
        "method",
        "run_type",
        "seeds",
        "n",
        "accuracy_mean",
        "accuracy_sd",
        "macro_f1_mean",
        "macro_f1_sd",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_root / "summary.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = output_root / "summary.md"
    lines = [
        "# Paper Reproduction Summary",
        "",
        "| Dataset | Method | Run type | n | Accuracy | Macro F1 | Seeds |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["accuracy_mean"] is None:
            acc = "-"
        elif row["n"] > 1:
            acc = f"{row['accuracy_mean']:.2f} +/- {row['accuracy_sd']:.2f}"
        else:
            acc = f"{row['accuracy_mean']:.2f} (n=1)"
        if row["macro_f1_mean"] is None:
            f1 = "-"
        elif row["n"] > 1:
            f1 = f"{row['macro_f1_mean']:.2f} +/- {row['macro_f1_sd']:.2f}"
        else:
            f1 = f"{row['macro_f1_mean']:.2f} (n=1)"
        lines.append(
            f"| {row['display_name']} | `{row['method']}` | {row['run_type']} | {row['n']} | "
            f"{acc} | {f1} | {row['seeds']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[aggregate] wrote {csv_path}")
    print(f"[aggregate] wrote {json_path}")
    print(f"[aggregate] wrote {md_path}")
