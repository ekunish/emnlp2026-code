"""Fixed-task Self-Instruct adaptation used for the camera-ready baseline.

The task type is fixed to binary sentence classification.  We therefore adapt
Self-Instruct's growing instruction pool to privacy-decision aspects, retain its
output-first classification step by choosing the label before the sentence,
and keep instances independent rather than introducing the proposed method's
contrastive pairing.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import Counter
from pathlib import Path

from .config import LABELS
from .rubric_guided_contrastive_synthesis import (
    _clean_pair_text,
    _complete_text,
    _max_rouge_l_f1,
    _normalize,
    _parse_json_payload,
    _preflight_vllm_capacity,
    _rouge_l_tokens,
)
from .utils import append_jsonl, load_seed_pool, read_jsonl

METHOD = "self_instruct_faithful_neutral"
PROMPT_ID = "self_instruct_faithful_neutral_v1"
INSTRUCTION_ROUGE_L_THRESHOLD = 0.70
INSTANCE_ROUGE_L_THRESHOLD = 0.95

NEUTRAL_SEED_INSTRUCTIONS: tuple[str, ...] = (
    "Decide whether a sentence discloses private information about the speaker.",
    "Decide whether a sentence would let a reader learn something the speaker "
    "would ordinarily keep private.",
    "Decide whether a sentence is about the speaker or about someone else.",
    "Decide whether a sentence describes a specific person or makes a general statement.",
    "Decide whether a sentence would be safe to publish without the speaker's consent.",
    "Decide whether a sentence carries private information or only mentions a topic "
    "that private information could belong to.",
)

INSTRUCTION_PROMPT = """You are extending a list of instructions for a binary sentence-level privacy classification task.

Each instruction names one aspect of deciding whether an English sentence is Sensitive or Non-sensitive.

Existing instructions:
{instructions}

Write {n_new} new instructions for aspects that the list does not already cover.

Requirements:
- Each instruction must start with "Decide whether a sentence".
- Name a concrete aspect. Do not restate an existing instruction in other words.
- Describe what to look for, not how to phrase a sentence.
- Do not mention datasets, benchmarks, prompts, or label names other than Sensitive and Non-sensitive.

Return only JSON:
{{
  "instructions": ["...", "..."]
}}"""

INSTANCE_PROMPT = """You are preparing synthetic training data for a binary sentence-level privacy classification task.

Use only the labeled demonstrations below to infer the distinction between Sensitive and Non-sensitive.

Sensitive demonstrations:
{sensitive_examples}

Non-sensitive demonstrations:
{nonsensitive_examples}

Instruction for this item:
{instruction}

The correct label for the sentence you write is fixed in advance: {label}

Write one new English sentence whose correct label is {label} and which is relevant to the instruction above.

Requirements:
- Write a natural English sentence compatible with the demonstrations.
- Expand the topics and expressions represented in the demonstrations.
- Do not copy or closely paraphrase a demonstration.
- Do not explain the label decision and do not state the label inside the sentence.
- Do not mention datasets, benchmarks, prompts, or labels inside the sentence.

Return only JSON:
{{
  "text": "..."
}}"""

VERIFIER_PROMPT = """You are verifying one synthetic training example for a binary sentence-level privacy classification task.

Infer the label distinction only from the original labeled examples below.

Sensitive examples:
{sensitive_examples}

Non-sensitive examples:
{nonsensitive_examples}

Candidate, claimed to be {label}:
<<<{text}>>>

Judge whether the candidate really has the claimed label, whether it is a natural standalone English sentence, and whether it copies an original example.

Return only JSON:
{{
  "label_correct": true,
  "natural": true,
  "copied_from_support": false,
  "accept": true,
  "reason": "..."
}}"""


def _render_examples(items: list[dict]) -> str:
    return "\n".join(f"- <<<{str(item['text']).strip()}>>>" for item in items)


def _clean_instruction(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip().strip("\"'").strip()
    if not 20 <= len(text) <= 300:
        return None
    if not text.lower().startswith("decide whether"):
        return None
    return text


def _choose_instruction_demonstrations(
    seed_instructions: list[str],
    generated_instructions: list[str],
    rng: random.Random,
) -> list[str]:
    """Select six human instructions and up to two accepted model instructions."""
    human = rng.sample(seed_instructions, min(6, len(seed_instructions)))
    generated = rng.sample(generated_instructions, min(2, len(generated_instructions)))
    selected = [*human, *generated]
    rng.shuffle(selected)
    return selected


def _instruction_is_novel(candidate: str, pool: list[str]) -> bool:
    candidate_tokens = _rouge_l_tokens(candidate)
    max_score = _max_rouge_l_f1(candidate_tokens, [_rouge_l_tokens(item) for item in pool])
    return max_score < INSTRUCTION_ROUGE_L_THRESHOLD


def _parse_instruction_response(raw: str | None) -> list[str]:
    payload = _parse_json_payload(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("instructions"), list):
        return []
    parsed = []
    for value in payload["instructions"]:
        instruction = _clean_instruction(value)
        if instruction is not None:
            parsed.append(instruction)
    return parsed


def _parse_instance_response(raw: str | None) -> str | None:
    payload = _parse_json_payload(raw)
    if not isinstance(payload, dict):
        return None
    return _clean_pair_text(payload.get("text"))


def _parse_verifier_response(raw: str | None) -> dict | None:
    payload = _parse_json_payload(raw)
    if not isinstance(payload, dict):
        return None
    required = ("label_correct", "natural", "copied_from_support", "accept")
    if not all(isinstance(payload.get(key), bool) for key in required):
        return None
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


async def self_instruct_faithful_neutral(
    *,
    dataset,
    seed_pool_dir: Path,
    out_path: Path,
    train_path: Path,
    n_samples: int,
    seed: int,
    model: str,
    vllm_url: str,
    concurrency: int,
    config: dict,
) -> dict:
    """Generate balanced, independent instances with a growing instruction pool."""
    del dataset, train_path
    if n_samples % 2:
        raise ValueError(f"{METHOD} requires an even n_samples, got {n_samples}")

    generation_cfg = config.get("generation", {})
    method_cfg = config.get(METHOD, {})
    temperature = float(generation_cfg.get("temperature", 1.0))
    top_p = float(generation_cfg.get("top_p", 0.9))
    timeout = float(generation_cfg.get("timeout", 120.0))
    attempt_cap = int(generation_cfg.get("attempt_cap", 20000))
    verifier_temperature = float(method_cfg.get("verifier_temperature", 0.0))
    instructions_per_step = int(method_cfg.get("instructions_per_step", 2))
    instruction_max_tokens = int(method_cfg.get("instruction_max_tokens", 400))
    instance_max_tokens = int(method_cfg.get("instance_max_tokens", 220))
    verifier_max_tokens = int(method_cfg.get("verifier_max_tokens", 200))

    _preflight_vllm_capacity(
        vllm_url=vllm_url,
        model=model,
        generation_cfg=generation_cfg,
    )

    seed_examples = load_seed_pool(seed_pool_dir)
    examples_by_label = {
        label: [item for item in seed_examples if str(item.get("label")) == label]
        for label in LABELS
    }
    if any(len(examples_by_label[label]) < 5 for label in LABELS):
        raise ValueError("the faithful baseline requires at least five seeds per label")
    rendered_examples = {
        label: _render_examples(examples_by_label[label]) for label in LABELS
    }

    existing = read_jsonl(out_path) if out_path.exists() else []
    foreign = {
        str(item.get("_method", ""))
        for item in existing
        if str(item.get("_method", "")) != METHOD
    }
    if foreign:
        raise RuntimeError(f"{out_path} contains rows from other methods: {sorted(foreign)}")

    target_per_label = n_samples // 2
    label_counts = Counter(str(item.get("label")) for item in existing)
    if any(label_counts[label] > target_per_label for label in LABELS):
        raise RuntimeError(f"{out_path} exceeds a balanced label target")

    pool_path = out_path.parent / "instruction_pool.json"
    generated_instructions: list[str] = []
    pool_history: list[dict] = []
    if pool_path.exists():
        saved = json.loads(pool_path.read_text())
        generated_instructions = [str(item) for item in saved.get("generated", [])]
        pool_history = list(saved.get("history", []))
    seed_instructions = list(NEUTRAL_SEED_INSTRUCTIONS)

    seed_keys = {_normalize(str(item["text"])) for item in seed_examples}
    seen = {_normalize(str(item["text"])) for item in existing}
    reference_tokens = [
        _rouge_l_tokens(str(item["text"])) for item in [*seed_examples, *existing]
    ]
    rng = random.Random(seed + 20_260_726)
    stats: Counter[str] = Counter()
    attempts = 0

    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=vllm_url, api_key="dummy", max_retries=0)
    semaphore = asyncio.Semaphore(concurrency)

    async def expand_pool() -> None:
        demonstrations = _choose_instruction_demonstrations(
            seed_instructions,
            generated_instructions,
            rng,
        )
        raw = await _complete_text(
            client=client,
            model=model,
            prompt=INSTRUCTION_PROMPT.format(
                instructions="\n".join(f"- {item}" for item in demonstrations),
                n_new=instructions_per_step,
            ),
            semaphore=semaphore,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_tokens=instruction_max_tokens,
        )
        candidates = _parse_instruction_response(raw)
        if not candidates:
            stats["instruction_parse_or_format_failed"] += 1
        for instruction in candidates:
            pool = [*seed_instructions, *generated_instructions]
            if not _instruction_is_novel(instruction, pool):
                stats["instruction_rouge_l_rejected"] += 1
                continue
            generated_instructions.append(instruction)
            stats["instruction_accepted"] += 1

    async def generate_one(label: str) -> dict:
        pool = [*seed_instructions, *generated_instructions]
        instruction = rng.choice(pool)
        raw = await _complete_text(
            client=client,
            model=model,
            prompt=INSTANCE_PROMPT.format(
                sensitive_examples=rendered_examples["Sensitive"],
                nonsensitive_examples=rendered_examples["Non-sensitive"],
                instruction=instruction,
                label=label,
            ),
            semaphore=semaphore,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_tokens=instance_max_tokens,
        )
        text = _parse_instance_response(raw)
        if text is None:
            return {"status": "generation_parse_or_format_failed"}
        verified_raw = await _complete_text(
            client=client,
            model=model,
            prompt=VERIFIER_PROMPT.format(
                sensitive_examples=rendered_examples["Sensitive"],
                nonsensitive_examples=rendered_examples["Non-sensitive"],
                label=label,
                text=text,
            ),
            semaphore=semaphore,
            temperature=verifier_temperature,
            top_p=top_p,
            timeout=timeout,
            max_tokens=verifier_max_tokens,
        )
        verified = _parse_verifier_response(verified_raw)
        if verified is None:
            return {"status": "verifier_parse_or_format_failed"}
        accepted = (
            verified["label_correct"]
            and verified["natural"]
            and not verified["copied_from_support"]
            and verified["accept"]
        )
        if not accepted:
            return {"status": "verifier_rejected"}
        return {
            "status": "verified",
            "text": text,
            "label": label,
            "instruction": instruction,
            "reason": str(verified.get("reason", "")),
        }

    while len(existing) < n_samples and attempts < attempt_cap:
        await expand_pool()
        remaining = {
            label: target_per_label - label_counts[label] for label in LABELS
        }
        batch_labels: list[str] = []
        batch_size = min(max(1, concurrency * 2), sum(remaining.values()))
        while len(batch_labels) < batch_size:
            for label in LABELS:
                if remaining[label] > 0 and len(batch_labels) < batch_size:
                    batch_labels.append(label)
                    remaining[label] -= 1
        if not batch_labels:
            break

        results = await asyncio.gather(*(generate_one(label) for label in batch_labels))
        attempts += len(results)
        new_rows = []
        for result in results:
            stats[result["status"]] += 1
            if result["status"] != "verified":
                continue
            label = str(result["label"])
            if label_counts[label] >= target_per_label:
                stats["label_quota_full"] += 1
                continue
            text = str(result["text"])
            normalized = _normalize(text)
            if normalized in seed_keys or normalized in seen:
                stats["exact_duplicate_rejected"] += 1
                continue
            candidate_tokens = _rouge_l_tokens(text)
            max_rouge_l = _max_rouge_l_f1(candidate_tokens, reference_tokens)
            if max_rouge_l >= INSTANCE_ROUGE_L_THRESHOLD:
                stats["instance_rouge_l_rejected"] += 1
                continue
            label_counts[label] += 1
            row = {
                "text": text,
                "label": label,
                "_item_id": f"{METHOD}_seed{seed}_{label.lower()}{label_counts[label]:06d}",
                "_method": METHOD,
                "_prompt_id": PROMPT_ID,
                "_instruction": result["instruction"],
                "_verifier_reason": result["reason"],
                "_rouge_l_threshold": INSTANCE_ROUGE_L_THRESHOLD,
                "_rouge_l_max_checked": max_rouge_l,
            }
            new_rows.append(row)
            seen.add(normalized)
            reference_tokens.append(candidate_tokens)
            stats["accepted_items"] += 1
        if new_rows:
            append_jsonl(new_rows, out_path)
            existing.extend(new_rows)
        pool_history.append(
            {
                "attempts": attempts,
                "accepted_items": len(existing),
                "pool_size": len(seed_instructions) + len(generated_instructions),
            }
        )
        _write_json(
            pool_path,
            {
                "seed": seed_instructions,
                "generated": generated_instructions,
                "history": pool_history,
            },
        )

    status = "complete" if len(existing) >= n_samples else "incomplete"
    metadata = {
        "method": METHOD,
        "prompt_id": PROMPT_ID,
        "seed_pool": str(seed_pool_dir),
        "num_seed_items": len(seed_examples),
        "target_samples": n_samples,
        "target_per_label": target_per_label,
        "num_written_total": len(existing),
        "num_written_by_label": dict(label_counts),
        "attempts": attempts,
        "attempt_cap": attempt_cap,
        "generation_status": status,
        "completion_rate": len(existing) / n_samples,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "instruction_pool": {
            "seed_instructions": len(seed_instructions),
            "generated_instructions": len(generated_instructions),
            "demonstrations_per_step": "six seed plus up to two generated",
            "rouge_l_threshold": INSTRUCTION_ROUGE_L_THRESHOLD,
        },
        "instance_generation": {
            "output_first": True,
            "independent_not_paired": True,
            "original_seed_demonstrations_per_label": 5,
            "generated_instances_reused_as_demonstrations": False,
            "normalized_exact_match": True,
            "rouge_l_threshold": INSTANCE_ROUGE_L_THRESHOLD,
        },
        "stats": dict(stats),
    }
    if status != "complete":
        raise RuntimeError(
            f"{METHOD} generated {len(existing)}/{n_samples} items after {attempts} attempts"
        )
    return metadata
