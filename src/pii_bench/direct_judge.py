"""No-training direct classification baselines."""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from .config import LABELS, DatasetConfig
from .utils import load_seed_pool, read_jsonl, write_jsonl

DIRECT_JUDGE_METHODS = {"oneshot_judge", "fewshot_judge", "first_person_rule"}
FIRST_PERSON_RE = re.compile(r"\b(I|my|me|mine|myself|we|our|ours|ourselves)\b", re.IGNORECASE)


def stratified_sample(items: list[dict], max_size: int, seed: int) -> list[dict]:
    if max_size <= 0 or len(items) <= max_size:
        return list(items)

    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(str(item["label"]), []).append(item)

    rng = random.Random(seed)
    labels = sorted(groups)
    quotas: dict[str, int] = {}
    assigned = 0
    fractions: list[tuple[float, str]] = []
    for label in labels:
        rng.shuffle(groups[label])
        raw_quota = max_size * len(groups[label]) / len(items)
        quota = int(raw_quota)
        if quota == 0 and max_size >= len(labels):
            quota = 1
        quota = min(quota, len(groups[label]))
        quotas[label] = quota
        fractions.append((raw_quota - int(raw_quota), label))
        assigned += quota

    for _, label in sorted(fractions, reverse=True):
        if assigned >= max_size:
            break
        if quotas[label] < len(groups[label]):
            quotas[label] += 1
            assigned += 1

    selected: list[dict] = []
    for label in labels:
        selected.extend(groups[label][: quotas[label]])
    rng.shuffle(selected)
    return selected


def strict_parse_label(raw: str | None) -> str | None:
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped in LABELS else None


def build_judge_prompt(test_text: str, examples: list[dict]) -> str:
    lines = [
        "Classify one PII-detection text as Sensitive or Non-sensitive.",
        "Respond with exactly one label: Sensitive or Non-sensitive.",
        "",
    ]
    for item in examples:
        lines.append(f"<<<{item['text']}>>>: {item['label']}")
    lines.extend(["", f"<<<{test_text}>>>:"])
    return "\n".join(lines)


def pick_examples(seed_pool: list[dict], k: int, rng: random.Random) -> list[dict]:
    examples: list[dict] = []
    for label in LABELS:
        candidates = [item for item in seed_pool if item["label"] == label]
        if not candidates:
            raise ValueError(f"seed pool has no {label} examples")
        examples.extend(rng.sample(candidates, min(k, len(candidates))))
    return examples


async def classify_one(
    *,
    client,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    temperature: float,
    top_p: float,
) -> tuple[str | None, bool, str, int]:
    last_raw = ""
    for attempt in range(max_retries + 1):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=8,
                    timeout=30.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                last_raw = response.choices[0].message.content or ""
            except Exception as exc:
                last_raw = f"ERROR: {exc}"
        parsed = strict_parse_label(last_raw)
        if parsed is not None:
            return parsed, False, last_raw, attempt + 1
        await asyncio.sleep(0.2)
    return None, True, last_raw, max_retries + 1


def write_direct_results(
    *,
    run_dir: Path,
    method: str,
    seed: int,
    predictions: list[dict],
    config: dict,
) -> None:
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions, results_dir / "predictions.jsonl")

    y_true = [item["true_label"] for item in predictions]
    y_pred = [
        item["pred_label"] if item["pred_label"] in LABELS else "__PARSE_FAILED__"
        for item in predictions
    ]
    metrics = {
        "best": {
            "in_test_acc": accuracy_score(y_true, y_pred),
            "in_test_f1": f1_score(y_true, y_pred, labels=list(LABELS), average="macro"),
            "parse_failed": sum(1 for item in predictions if item["parse_failed"]),
            "n_eval": len(predictions),
        },
        "history": [],
        "config": {
            "method": method,
            "run_type": "direct_judge",
            "seed": seed,
            **config,
        },
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def evaluate_first_person_rule(
    *,
    dataset: DatasetConfig,
    run_dir: Path,
    seed: int,
    eval_test_size: int,
    eval_subsample_seed: int,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    result = run_dir / "results" / "metrics.json"
    if result.exists() and not overwrite:
        print(f"[skip] direct_judge: {result}")
        return
    print(f"[direct_judge] dataset={dataset.name} method=first_person_rule seed={seed}")
    if dry_run:
        return

    items = stratified_sample(read_jsonl(dataset.test), eval_test_size, eval_subsample_seed)
    predictions = []
    for item in items:
        pred = "Sensitive" if FIRST_PERSON_RE.search(item["text"]) else "Non-sensitive"
        predictions.append(
            {
                "text": item["text"],
                "true_label": item["label"],
                "pred_label": pred,
                "correct": pred == item["label"],
                "parse_failed": False,
                "raw_response": pred,
            }
        )
    write_direct_results(
        run_dir=run_dir,
        method="first_person_rule",
        seed=seed,
        predictions=predictions,
        config={
            "eval_test_size": eval_test_size,
            "eval_subsample_seed": eval_subsample_seed,
        },
    )


async def evaluate_seed_pool_judge(
    *,
    method: str,
    dataset: DatasetConfig,
    seed_pool_dir: Path,
    run_dir: Path,
    seed: int,
    model: str,
    vllm_url: str,
    concurrency: int,
    max_retries: int,
    temperature: float,
    top_p: float,
    eval_test_size: int,
    eval_subsample_seed: int,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    result = run_dir / "results" / "metrics.json"
    if result.exists() and not overwrite:
        print(f"[skip] direct_judge: {result}")
        return
    print(f"[direct_judge] dataset={dataset.name} method={method} seed={seed}")
    if dry_run:
        return

    k = 1 if method == "oneshot_judge" else 5
    seed_pool = load_seed_pool(seed_pool_dir)
    rng = random.Random(seed)
    test_items = stratified_sample(read_jsonl(dataset.test), eval_test_size, eval_subsample_seed)
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=vllm_url, api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)

    prompts: list[tuple[dict, list[dict], str]] = []
    for item in test_items:
        examples = pick_examples(seed_pool, k, rng)
        prompts.append((item, examples, build_judge_prompt(item["text"], examples)))

    async def classify_indexed(index: int, prompt: str) -> tuple[int, str | None, bool, str, int]:
        pred, failed, raw, attempts = await classify_one(
            client=client,
            model=model,
            prompt=prompt,
            semaphore=semaphore,
            max_retries=max_retries,
            temperature=temperature,
            top_p=top_p,
        )
        return index, pred, failed, raw, attempts

    tasks = [
        asyncio.create_task(classify_indexed(index, prompt))
        for index, (_, _, prompt) in enumerate(prompts)
    ]
    from tqdm.asyncio import tqdm_asyncio

    outputs = await tqdm_asyncio.gather(*tasks, desc=method)
    by_index = {
        index: (pred, failed, raw, attempts)
        for index, pred, failed, raw, attempts in outputs
    }

    predictions = []
    for index, (item, examples, prompt) in enumerate(prompts):
        pred, failed, raw, attempts = by_index[index]
        predictions.append(
            {
                "text": item["text"],
                "true_label": item["label"],
                "pred_label": pred,
                "correct": pred == item["label"],
                "parse_failed": failed,
                "raw_response": raw,
                "attempts": attempts,
                "support": examples,
                "prompt": prompt,
            }
        )

    write_direct_results(
        run_dir=run_dir,
        method=method,
        seed=seed,
        predictions=predictions,
        config={
            "seed_pool": str(seed_pool_dir),
            "k_per_label": k,
            "model": model,
            "vllm_url": vllm_url,
            "concurrency": concurrency,
            "max_retries": max_retries,
            "temperature": temperature,
            "top_p": top_p,
            "eval_test_size": eval_test_size,
            "eval_subsample_seed": eval_subsample_seed,
            "parse_policy": "strict_exact_label_only_no_fallback",
            "label_counts": dict(Counter(item["label"] for item in seed_pool)),
        },
    )
