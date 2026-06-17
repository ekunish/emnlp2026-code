"""CLI for public paper reproduction."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import math
import shutil
from pathlib import Path

from .aggregate import aggregate_results
from .config import LABELS, BenchmarkConfig, load_config
from .direct_judge import DIRECT_JUDGE_METHODS, evaluate_first_person_rule, evaluate_seed_pool_judge
from .duplicate import build_duplicate
from .generate import generate_fewshot_like, generate_paraphrase
from .prepare import prepare_all
from .train_eval import check_evaluation, train_and_evaluate
from .utils import read_jsonl, repo_root

TRAIN_METHODS = {"official_train", "oneshot", "fewshot", "duplicate", "paraphrase"}
SYNTHETIC_METHODS = {"oneshot", "fewshot", "duplicate", "paraphrase"}
DIRECT_SEED = 0


def parse_csv_list(
    value: str,
    allowed: list[str],
    default_items: list[str] | None = None,
) -> list[str]:
    if value == "all":
        return default_items or allowed
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - set(allowed))
    if unknown:
        raise SystemExit(f"Unknown values {unknown}; allowed: {allowed}")
    return items


def discover_output_datasets(output_root: Path, allowed: list[str]) -> list[str]:
    """Return configured datasets that already have output directories."""
    allowed_set = set(allowed)
    if not output_root.exists():
        return []
    return [
        dataset_dir.name
        for dataset_dir in sorted(p for p in output_root.iterdir() if p.is_dir())
        if dataset_dir.name in allowed_set
    ]


def discover_output_methods(
    output_root: Path,
    datasets: list[str],
    allowed: list[str],
) -> list[str]:
    """Return configured methods that already have output directories."""
    allowed_set = set(allowed)
    discovered: list[str] = []
    seen: set[str] = set()
    for dataset_name in datasets:
        dataset_dir = output_root / dataset_name
        if not dataset_dir.exists():
            continue
        for method_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            method = method_dir.name
            if method in allowed_set and method not in seen:
                discovered.append(method)
                seen.add(method)
    return discovered


def resolve_output_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def run_dir(output_root: Path, dataset: str, method: str, seed: int) -> Path:
    return output_root / dataset / method / f"seed_{seed}"


def training_data_path(output_root: Path, dataset: str, method: str, seed: int) -> Path:
    return run_dir(output_root, dataset, method, seed) / "training_data.jsonl"


def generated_data_path(output_root: Path, dataset: str, method: str, seed: int) -> Path:
    if method == "duplicate":
        return training_data_path(output_root, dataset, method, seed)
    return run_dir(output_root, dataset, method, seed) / "generated" / "synthetic_data.jsonl"


def write_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_meta(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def copy_training_data(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copyfile(src, dst)


def unlink_if_present(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def allows_incomplete_generation(method: str, meta: dict) -> bool:
    if "allow_incomplete_training_data" in meta:
        return bool(meta.get("allow_incomplete_training_data"))
    return method.endswith("_100x") and meta.get("generation_status") == "incomplete_card_budget"


def has_trainable_labels(path: Path) -> bool:
    labels = {str(item.get("label")) for item in read_jsonl(path)}
    return set(LABELS).issubset(labels)


def seeds_for_method(method: str, seeds: list[int]) -> list[int]:
    if method == "first_person_rule":
        return [DIRECT_SEED]
    return seeds


def load_custom_method(config: BenchmarkConfig, method: str):
    custom_methods = config.raw.get("custom_methods", {})
    target = custom_methods.get(method)
    if not target:
        return None
    module_name, _, func_name = target.partition(":")
    if not module_name or not func_name:
        raise ValueError(f"custom method {method!r} must be 'module:function', got {target!r}")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


async def maybe_call(func, **kwargs):
    result = func(**kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def generate_method(
    *,
    args: argparse.Namespace,
    config: BenchmarkConfig,
    dataset_name: str,
    method: str,
    seed: int,
) -> None:
    dataset = config.datasets[dataset_name]
    per_label = config.seed_pool_per_label
    pool = dataset.seed_pool(seed, per_label)
    out_jsonl = generated_data_path(args.output_root, dataset_name, method, seed)
    train_path = training_data_path(args.output_root, dataset_name, method, seed)
    meta_path = out_jsonl.parent / "generation_meta.json"

    if train_path.exists() and not args.overwrite:
        training_count = count_jsonl_lines(train_path)
        if training_count >= args.n_samples:
            print(f"[skip] generated: {train_path}")
            return
        existing_meta = read_meta(meta_path)
        if existing_meta and allows_incomplete_generation(method, existing_meta):
            min_required = int(existing_meta.get("min_required_items", 1))
            if training_count >= min_required and has_trainable_labels(train_path):
                print(
                    f"[skip] accepted incomplete generated: {train_path} "
                    f"rows={training_count}/{args.n_samples} "
                    f"status={existing_meta.get('generation_status')}"
                )
                return

    print(f"[generate] dataset={dataset_name} method={method} seed={seed} pool={pool}")
    if args.dry_run:
        return
    if args.overwrite:
        unlink_if_present(out_jsonl)
        unlink_if_present(train_path)
        unlink_if_present(meta_path)

    if method == "duplicate":
        meta = build_duplicate(pool, out_jsonl, seed=seed, n_samples=args.n_samples)
    elif method == "oneshot":
        meta = await generate_fewshot_like(
            seed_pool_dir=pool,
            out_path=out_jsonl,
            method=method,
            k=1,
            n_samples=args.n_samples,
            seed=seed,
            model=args.model,
            vllm_url=args.vllm_url,
            concurrency=args.concurrency,
            attempt_cap=args.attempt_cap,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif method == "fewshot":
        meta = await generate_fewshot_like(
            seed_pool_dir=pool,
            out_path=out_jsonl,
            method=method,
            k=5,
            n_samples=args.n_samples,
            seed=seed,
            model=args.model,
            vllm_url=args.vllm_url,
            concurrency=args.concurrency,
            attempt_cap=args.attempt_cap,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif method == "paraphrase":
        pool_size = len(read_jsonl(pool / "Sensitive.jsonl")) + len(
            read_jsonl(pool / "Non-sensitive.jsonl")
        )
        meta = await generate_paraphrase(
            seed_pool_dir=pool,
            out_path=out_jsonl,
            per_seed=max(1, math.ceil(args.n_samples / pool_size)),
            target_samples=args.n_samples,
            seed=seed,
            model=args.model,
            vllm_url=args.vllm_url,
            concurrency=args.concurrency,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.paraphrase_batch_size,
            attempt_factor=args.paraphrase_attempt_factor,
            request_timeout=args.paraphrase_request_timeout,
        )
    else:
        custom = load_custom_method(config, method)
        if custom is None:
            raise ValueError(f"Unknown synthetic method: {method}")
        custom_config = dict(config.raw)
        custom_generation = dict(custom_config.get("generation", {}))
        custom_generation.update(
            {
                "n_samples": args.n_samples,
                "model": args.model,
                "vllm_url": args.vllm_url,
                "concurrency": args.concurrency,
                "attempt_cap": args.attempt_cap,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "presence_penalty": args.presence_penalty,
            }
        )
        custom_config["generation"] = custom_generation
        meta = await maybe_call(
            custom,
            dataset=dataset,
            seed_pool_dir=pool,
            out_path=out_jsonl,
            train_path=train_path,
            n_samples=args.n_samples,
            seed=seed,
            model=args.model,
            vllm_url=args.vllm_url,
            concurrency=args.concurrency,
            config=custom_config,
        )

    generated_count = count_jsonl_lines(out_jsonl)
    if generated_count < args.n_samples:
        if not allows_incomplete_generation(method, meta):
            write_meta(meta_path, meta)
            raise RuntimeError(f"{method} wrote fewer than requested items: {out_jsonl}")
        generated_rows = read_jsonl(out_jsonl)
        generated_labels = {str(item.get("label")) for item in generated_rows}
        if generated_count == 0 or not set(LABELS).issubset(generated_labels):
            write_meta(meta_path, meta)
            raise RuntimeError(
                f"{method} wrote insufficient trainable data "
                f"({generated_count} rows, labels={sorted(generated_labels)}): {out_jsonl}"
            )
        print(
            f"[warn] incomplete generation accepted: {out_jsonl} "
            f"rows={generated_count}/{args.n_samples} "
            f"status={meta.get('generation_status')}"
        )
    copy_training_data(out_jsonl, train_path)
    write_meta(meta_path, meta)


async def train_or_direct_method(
    *,
    args: argparse.Namespace,
    config: BenchmarkConfig,
    dataset_name: str,
    method: str,
    seed: int,
) -> None:
    dataset = config.datasets[dataset_name]
    rdir = run_dir(args.output_root, dataset_name, method, seed)

    if method == "first_person_rule":
        evaluate_first_person_rule(
            dataset=dataset,
            run_dir=rdir,
            seed=seed,
            eval_test_size=args.eval_test_size,
            eval_subsample_seed=args.eval_subsample_seed,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        return

    if method in {"oneshot_judge", "fewshot_judge"}:
        await evaluate_seed_pool_judge(
            method=method,
            dataset=dataset,
            seed_pool_dir=dataset.seed_pool(seed, config.seed_pool_per_label),
            run_dir=rdir,
            seed=seed,
            model=args.judge_model,
            vllm_url=args.judge_vllm_url,
            concurrency=args.judge_concurrency,
            max_retries=args.judge_max_retries,
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
            eval_test_size=args.eval_test_size,
            eval_subsample_seed=args.eval_subsample_seed,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        return

    if method == "official_train":
        train_data = dataset.train
    else:
        train_data = training_data_path(args.output_root, dataset_name, method, seed)
        if not args.dry_run and not train_data.exists():
            raise FileNotFoundError(
                f"Missing training data for {dataset_name}/{method}/seed_{seed}: {train_data}"
            )

    train_and_evaluate(
        dataset=dataset,
        method=method,
        train_data=train_data,
        run_dir=rdir,
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        max_len=args.max_len,
        backbone=args.backbone,
        patience=args.patience,
        eval_valid_size=args.eval_valid_size,
        eval_test_size=args.eval_test_size,
        eval_subsample_seed=args.eval_subsample_seed,
        monitor=args.monitor,
        calibrate_threshold=args.calibrate_threshold,
        keep_predictions=args.keep_predictions,
        keep_model_weights=args.keep_model_weights,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


def eval_method(args: argparse.Namespace, dataset_name: str, method: str, seed: int) -> None:
    rdir = run_dir(args.output_root, dataset_name, method, seed)
    print(f"[eval] dataset={dataset_name} method={method} seed={seed}")
    if args.dry_run:
        return
    check_evaluation(rdir)


async def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    args.output_root = resolve_output_root(args.output_root)

    allowed_methods = [*config.all_methods, *sorted(config.raw.get("custom_methods", {}))]
    allowed_datasets = list(config.datasets)
    default_datasets = None
    if args.stage == "aggregate":
        discovered_datasets = discover_output_datasets(args.output_root, allowed_datasets)
        if discovered_datasets:
            default_datasets = discovered_datasets
    datasets = parse_csv_list(args.datasets, allowed_datasets, default_datasets)
    if args.stage == "direct_judge":
        default_methods = config.direct_judge_methods
    elif args.stage == "aggregate":
        default_methods = discover_output_methods(args.output_root, datasets, allowed_methods)
        if not default_methods:
            default_methods = allowed_methods
    else:
        default_methods = config.train_methods
    methods = parse_csv_list(args.methods, allowed_methods, default_methods)
    seeds = args.seeds

    if args.stage in {"prepare", "all"}:
        prepare_all(
            config=config,
            datasets=datasets,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    if args.stage in {"generate", "all"}:
        for dataset_name in datasets:
            for method in methods:
                if (
                    method not in SYNTHETIC_METHODS
                    and method not in config.raw.get("custom_methods", {})
                ):
                    continue
                for seed in seeds:
                    await generate_method(
                        args=args,
                        config=config,
                        dataset_name=dataset_name,
                        method=method,
                        seed=seed,
                    )

    if args.stage in {"train_eval", "direct_judge", "all"}:
        for dataset_name in datasets:
            for method in methods:
                if args.stage == "direct_judge" and method not in DIRECT_JUDGE_METHODS:
                    continue
                if args.stage == "train_eval" and method in DIRECT_JUDGE_METHODS:
                    continue
                for seed in seeds_for_method(method, seeds):
                    await train_or_direct_method(
                        args=args,
                        config=config,
                        dataset_name=dataset_name,
                        method=method,
                        seed=seed,
                    )

    if args.stage in {"eval", "all"}:
        for dataset_name in datasets:
            for method in methods:
                for seed in seeds_for_method(method, seeds):
                    eval_method(args, dataset_name, method, seed)

    if args.stage in {"aggregate", "all"}:
        if args.dry_run:
            print("[aggregate] dry-run")
        else:
            aggregate_results(
                config=config,
                output_root=args.output_root,
                datasets=datasets,
                methods=methods,
                seeds=seeds,
            )


def main() -> None:
    config = load_config()
    generation = config.generation
    judge = config.direct_judge
    trainer = config.trainer
    artifacts = config.artifacts
    parser = argparse.ArgumentParser(description="Run paper benchmark reproduction.")
    parser.add_argument("--config", default=str(repo_root() / "config.yaml"))
    parser.add_argument(
        "--stage",
        choices=["prepare", "generate", "train_eval", "direct_judge", "eval", "aggregate", "all"],
        default="all",
    )
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names or 'all'.")
    parser.add_argument("--methods", default="all", help="Comma-separated method names or 'all'.")
    parser.add_argument("--seeds", nargs="+", type=int, default=config.seeds)
    parser.add_argument(
        "--output-root",
        default=config.paths.get("output_root", "outputs/runs"),
    )

    parser.add_argument("--n-samples", type=int, default=int(generation.get("n_samples", 2000)))
    parser.add_argument("--model", default=generation.get("model", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--vllm-url", default=generation.get("vllm_url", "http://localhost:8000/v1"))
    parser.add_argument("--concurrency", type=int, default=int(generation.get("concurrency", 1)))
    parser.add_argument(
        "--attempt-cap",
        type=int,
        default=int(generation.get("attempt_cap", 20000)),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(generation.get("temperature", 1.0)),
    )
    parser.add_argument("--top-p", type=float, default=float(generation.get("top_p", 0.9)))
    parser.add_argument(
        "--top-k",
        type=int,
        default=generation.get("top_k"),
        help="Optional vLLM top_k sampling value for custom generation methods.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=float(generation.get("presence_penalty", 0.0)),
        help="Presence penalty for custom generation methods.",
    )
    parser.add_argument(
        "--paraphrase-batch-size",
        type=int,
        default=int(generation.get("paraphrase_batch_size", 10)),
    )
    parser.add_argument(
        "--paraphrase-attempt-factor",
        type=float,
        default=float(generation.get("paraphrase_attempt_factor", 0.5)),
    )
    parser.add_argument(
        "--paraphrase-request-timeout",
        type=float,
        default=float(generation.get("paraphrase_request_timeout", 30.0)),
    )

    parser.add_argument(
        "--judge-model",
        default=judge.get("model", generation.get("model", "Qwen/Qwen3.5-9B")),
    )
    parser.add_argument(
        "--judge-vllm-url",
        default=judge.get(
            "vllm_url",
            generation.get("vllm_url", "http://localhost:8000/v1"),
        ),
    )
    parser.add_argument("--judge-concurrency", type=int, default=int(judge.get("concurrency", 1)))
    parser.add_argument("--judge-max-retries", type=int, default=int(judge.get("max_retries", 2)))
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=float(judge.get("temperature", 0.0)),
    )
    parser.add_argument("--judge-top-p", type=float, default=float(judge.get("top_p", 0.9)))

    early_stopping = trainer.get("early_stopping", {})
    patience = int(early_stopping.get("patience", 0)) if early_stopping.get("enabled", False) else 0
    parser.add_argument("--backbone", default=trainer.get("backbone", "roberta-base"))
    parser.add_argument("--epochs", type=int, default=int(trainer.get("epochs", 15)))
    parser.add_argument("--batch-size", type=int, default=int(trainer.get("batch_size", 8)))
    parser.add_argument("--lr", type=float, default=float(trainer.get("lr", 1e-5)))
    parser.add_argument("--dropout", type=float, default=float(trainer.get("dropout", 0.3)))
    parser.add_argument("--max-len", type=int, default=int(trainer.get("max_len", 256)))
    parser.add_argument("--patience", type=int, default=patience)
    parser.add_argument(
        "--monitor",
        choices=["valid_acc", "valid_f1"],
        default=trainer.get("monitor", "valid_acc"),
    )
    parser.add_argument(
        "--eval-valid-size",
        type=int,
        default=int(trainer.get("eval_valid_size", 2000)),
    )
    parser.add_argument("--eval-test-size", type=int, default=int(trainer.get("eval_test_size", 0)))
    parser.add_argument(
        "--eval-subsample-seed",
        type=int,
        default=int(trainer.get("eval_subsample_seed", 0)),
    )
    parser.add_argument(
        "--calibrate-threshold",
        action=argparse.BooleanOptionalAction,
        default=bool(trainer.get("calibrate_threshold", False)),
        help=(
            "Tune the binary Sensitive probability threshold on validation Macro F1 "
            "and apply it to test."
        ),
    )
    parser.add_argument(
        "--keep-predictions",
        action=argparse.BooleanOptionalAction,
        default=bool(artifacts.get("keep_predictions", True)),
    )
    parser.add_argument(
        "--keep-model-weights",
        action=argparse.BooleanOptionalAction,
        default=bool(artifacts.get("keep_model_weights", False)),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
