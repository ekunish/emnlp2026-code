"""LLM-based generation for oneshot, fewshot, and paraphrase baselines."""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from .utils import append_jsonl, load_seed_pool, read_jsonl, write_jsonl

TASK_DESC = (
    "Task: generate one PII-detection training text for the {target_label} class. "
    "PII-detection treats text as Sensitive when it contains directly-identifying "
    "personal information (email, phone, address, ID, URL, username) or "
    "Special-Category attributes (health, ethnicity, religion, political opinion, "
    "criminal record). Otherwise Non-sensitive.\n\n"
    "Below are reference examples. Generate ONE new {target_label} text in similar "
    "style. Output ONLY the text between <<< and >>>."
)

BATCH_PARAPHRASE_PROMPT = (
    "Generate {n} diverse rewrites of the following sentence in different words while "
    "preserving its full meaning, tone, and any personal identifiers "
    "(email/phone/address/URL/name/ID/username) exactly as they appear. Do not modify, "
    "redact, or invent identifiers. Output ONLY a JSON array of strings.\n\n"
    "Original: {text}"
)

MARKER_PREFIX_RE = re.compile(
    r"^\s*(?:(?:Sensitive|Non-sensitive)\s*:\s*|:\s*)?<<<\s*",
    re.IGNORECASE,
)
LABEL_PREFIX_RE = re.compile(r"^\s*(?:Sensitive|Non-sensitive)\s*:\s*", re.IGNORECASE)
PROMPT_ARTIFACT_RE = re.compile(
    r"(?:<<<|>>>|^\s*(?:Sensitive|Non-sensitive)\s*:|"
    r"Classify one PII-detection text|Respond with exactly one label)",
    re.IGNORECASE,
)


def _labels(seed_examples: list[dict]) -> list[str]:
    labels = sorted({item["label"] for item in seed_examples})
    if "Sensitive" in labels and "Non-sensitive" in labels:
        return ["Sensitive", "Non-sensitive"]
    return labels


def _balanced_label_targets(labels: list[str], n_samples: int) -> dict[str, int]:
    base = n_samples // len(labels)
    remainder = n_samples % len(labels)
    return {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(labels)
    }


def build_fewshot_prompt(
    target_label: str,
    all_examples: list[dict],
    k: int,
    rng: random.Random,
) -> str:
    sens_examples = [ex for ex in all_examples if ex["label"] == "Sensitive"]
    ns_examples = [ex for ex in all_examples if ex["label"] == "Non-sensitive"]
    sens_picked = rng.sample(sens_examples, min(k, len(sens_examples)))
    ns_picked = rng.sample(ns_examples, min(k, len(ns_examples)))

    lines = [TASK_DESC.format(target_label=target_label), ""]
    for sens, ns in zip(sens_picked, ns_picked, strict=False):
        lines.append(f"Non-sensitive: <<<{ns['text']}>>>")
        lines.append(f"Sensitive: <<<{sens['text']}>>>")
    lines.append("")
    lines.append(f"{target_label}: <<<")
    return "\n".join(lines)


def clean_generated_text(text: str | None) -> str | None:
    if not text:
        return None
    if ">>>" in text:
        text = text.split(">>>", 1)[0]
    else:
        text = text.splitlines()[0] if text.splitlines() else text
    text = text.strip()
    for _ in range(3):
        cleaned = MARKER_PREFIX_RE.sub("", text)
        cleaned = LABEL_PREFIX_RE.sub("", cleaned)
        if cleaned == text:
            break
        text = cleaned.strip()
    text = text.strip().strip("\"'<>").strip()
    if PROMPT_ARTIFACT_RE.search(text):
        return None
    if 10 <= len(text) <= 500:
        return text
    return None


async def generate_one(
    client,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    *,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> str | None:
    for _ in range(3):
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=64,
                    stop=[">>>", "\n"],
                    timeout=30.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                text = clean_generated_text(resp.choices[0].message.content)
                if text:
                    return text
            except Exception:
                await asyncio.sleep(0.5)
    return None


async def generate_fewshot_like(
    *,
    seed_pool_dir: Path,
    out_path: Path,
    method: str,
    k: int,
    n_samples: int,
    seed: int,
    model: str,
    vllm_url: str,
    concurrency: int,
    attempt_cap: int = 20000,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> dict:
    seed_examples = load_seed_pool(seed_pool_dir)
    labels = _labels(seed_examples)
    existing = read_jsonl(out_path) if out_path.exists() else []
    remaining = n_samples - len(existing)
    if remaining <= 0:
        return {
            "method": method,
            "seed_pool": str(seed_pool_dir),
            "num_seed_items": len(seed_examples),
            "num_existing": len(existing),
            "num_generated": 0,
            "num_rejected": 0,
        }

    rng = random.Random(seed)
    from openai import AsyncOpenAI
    from tqdm.asyncio import tqdm

    client = AsyncOpenAI(base_url=vllm_url, api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)
    counts: Counter[str] = Counter(item["label"] for item in existing)
    target_counts = _balanced_label_targets(labels, n_samples)
    overfull = {
        label: count
        for label, count in counts.items()
        if count > target_counts.get(label, 0)
    }
    if overfull:
        raise RuntimeError(
            f"{out_path} already contains more items than the balanced target: {overfull}"
        )
    pending_labels = [
        label
        for label in labels
        for _ in range(target_counts[label] - counts[label])
    ]
    rng.shuffle(pending_labels)
    if len(pending_labels) != remaining:
        raise RuntimeError(
            f"{out_path} has inconsistent existing labels: "
            f"existing={dict(counts)}, target={target_counts}"
        )

    generated = 0
    rejected = 0
    buffer: list[dict] = []
    pbar = tqdm(total=remaining, desc=method)

    def flush() -> None:
        nonlocal buffer
        if buffer:
            append_jsonl(buffer, out_path)
            buffer = []

    async def run_one(target_label: str) -> str | None:
        nonlocal generated, rejected
        prompt = build_fewshot_prompt(target_label, seed_examples, k, rng)
        text = await generate_one(
            client,
            model,
            prompt,
            semaphore,
            temperature=temperature,
            top_p=top_p,
        )
        if text:
            buffer.append({"text": text, "label": target_label})
            generated += 1
            pbar.update(1)
            if len(buffer) >= 64:
                flush()
            return None
        else:
            rejected += 1
            return target_label

    attempts = 0
    while pending_labels and attempts < attempt_cap:
        batch_size = min(len(pending_labels), max(1, concurrency * 4))
        batch_labels = pending_labels[:batch_size]
        pending_labels = pending_labels[batch_size:]
        tasks = [asyncio.create_task(run_one(label)) for label in batch_labels]
        attempts += batch_size
        failed_labels = [label for label in await asyncio.gather(*tasks) if label is not None]
        pending_labels.extend(failed_labels)
        rng.shuffle(pending_labels)
        flush()
    flush()
    pbar.close()

    total_written = len(read_jsonl(out_path)) if out_path.exists() else 0
    if total_written < n_samples:
        raise RuntimeError(
            f"{method} generated only {total_written}/{n_samples} items "
            f"after {attempts} attempts"
        )

    return {
        "method": method,
        "seed_pool": str(seed_pool_dir),
        "num_seed_items": len(seed_examples),
        "k_per_label": k,
        "target_samples": n_samples,
        "num_generated": generated,
        "num_rejected": rejected,
        "num_written_total": total_written,
        "attempt_cap": attempt_cap,
        "attempts": attempts,
        "api": "chat.completions.create",
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
    }


def jaccard_3gram(a: str, b: str) -> float:
    def tri(s: str) -> set[str]:
        s = re.sub(r"\s+", " ", s).lower()
        return {s[i : i + 3] for i in range(max(0, len(s) - 2))}

    a_set, b_set = tri(a), tri(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def parse_rewrites(text: str) -> list[str]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except json.JSONDecodeError:
            pass

    rewrites = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*]?\s*\d*[\).\:-]?\s*", "", line).strip()
        line = line.strip("\"'<>")
        if line:
            rewrites.append(line)
    return rewrites


async def generate_rewrite_batch(
    client,
    model: str,
    source_text: str,
    n: int,
    semaphore: asyncio.Semaphore,
    *,
    temperature: float = 1.0,
    top_p: float = 0.9,
    request_timeout: float = 30.0,
) -> list[str]:
    prompt = BATCH_PARAPHRASE_PROMPT.format(
        n=n,
        text=json.dumps(source_text, ensure_ascii=False),
    )
    for _ in range(3):
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max(200, n * 50),
                    timeout=request_timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                text = resp.choices[0].message.content
                if text:
                    rewrites = [
                        rewrite
                        for raw in parse_rewrites(text)
                        if (rewrite := clean_generated_text(raw))
                    ]
                    if rewrites:
                        return rewrites[:n]
            except Exception:
                await asyncio.sleep(0.5)
    return []


async def generate_paraphrase(
    *,
    seed_pool_dir: Path,
    out_path: Path,
    per_seed: int,
    seed: int,
    model: str,
    vllm_url: str,
    concurrency: int,
    target_samples: int | None = None,
    batch_size: int = 10,
    attempt_factor: float = 0.5,
    dup_threshold: float = 0.85,
    temperature: float = 1.0,
    top_p: float = 0.9,
    request_timeout: float = 30.0,
) -> dict:
    seed_items = load_seed_pool(seed_pool_dir)
    from openai import AsyncOpenAI
    from tqdm.asyncio import tqdm

    client = AsyncOpenAI(base_url=vllm_url, api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)
    total = len(seed_items) * per_seed
    pbar = tqdm(total=total, desc="paraphrase")
    output: list[dict] = []
    padding_by_seed: list[dict] = []

    async def collect_for_seed(seed_item: dict, index: int) -> None:
        candidates: list[str] = []
        target_attempts = math.ceil(per_seed * attempt_factor)

        async def collect_batches(n_items: int) -> None:
            n_batches = math.ceil(n_items / batch_size)
            sizes = [batch_size] * n_batches
            sizes[-1] = n_items - batch_size * (n_batches - 1)
            tasks = [
                asyncio.create_task(
                    generate_rewrite_batch(
                        client,
                        model,
                        seed_item["text"],
                        n,
                        semaphore,
                        temperature=temperature,
                        top_p=top_p,
                        request_timeout=request_timeout,
                    )
                )
                for n in sizes
            ]
            for batch in await asyncio.gather(*tasks):
                for text in batch:
                    if jaccard_3gram(text, seed_item["text"]) > dup_threshold:
                        continue
                    if any(
                        jaccard_3gram(text, existing) > dup_threshold for existing in candidates
                    ):
                        continue
                    candidates.append(text)
                    if len(candidates) >= per_seed:
                        return

        await collect_batches(target_attempts)
        if len(candidates) < per_seed:
            await collect_batches(batch_size * 2)

        accepted = len(candidates)
        while len(candidates) < per_seed:
            candidates.append(seed_item["text"])
        padding = per_seed - accepted
        padding_by_seed.append(
            {
                "seed_item_index": index,
                "label": seed_item["label"],
                "accepted": accepted,
                "padding": padding,
            }
        )
        for text in candidates[:per_seed]:
            output.append({"text": text, "label": seed_item["label"]})
            pbar.update(1)

    await asyncio.gather(
        *[asyncio.create_task(collect_for_seed(item, idx)) for idx, item in enumerate(seed_items)]
    )
    pbar.close()
    random.Random(seed).shuffle(output)
    if target_samples is not None:
        output = output[:target_samples]
    write_jsonl(output, out_path)
    padding_total = sum(item["padding"] for item in padding_by_seed)
    return {
        "method": "paraphrase",
        "seed_pool": str(seed_pool_dir),
        "num_seed_items": len(seed_items),
        "per_seed": per_seed,
        "target_samples": target_samples or total,
        "batch_size": batch_size,
        "attempt_factor": attempt_factor,
        "request_timeout": request_timeout,
        "num_written": len(output),
        "padding_total": padding_total,
        "padding_rate": padding_total / max(1, len(output)),
        "padding_by_seed": sorted(padding_by_seed, key=lambda x: x["seed_item_index"]),
        "api": "chat.completions.create",
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
    }
