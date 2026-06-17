"""Rubric-guided contrastive synthesis from a few labeled seed examples."""

from __future__ import annotations

import asyncio
import json
import random
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from itertools import combinations, product
from pathlib import Path

from .config import LABELS
from .utils import append_jsonl, load_seed_pool, read_jsonl

RUBRIC_INDUCTION_PROMPT = """You are preparing synthetic training data for a binary privacy text classification task.

Use only the labeled support examples below. Do not use dataset names, benchmark descriptions, external privacy policies, or prior assumptions.

Sensitive examples:
{positive_examples}

Non-sensitive examples:
{negative_examples}

Task:
Induce up to {k} rubrics. Each rubric describes one label-defining difference that is supported by the labeled examples.

A valid rubric must:
- cite at least one Sensitive support example and one Non-sensitive support example;
- describe the Sensitive side by stating what information in the cited example supports the Sensitive label, whose or what information it is, and how the text's speaker or source is related to that information;
- describe the comparable Non-sensitive side using the same three points: what information in the cited example is closest to the Sensitive-side evidence or is absent, whose or what it is about, and how the text's speaker or source relation differs.

Return only JSON in this format:
[
  {{
    "rubric_id": "R1",
    "sensitive_condition": "...",
    "nonsensitive_condition": "...",
    "support_sensitive": ["S1", "S3"],
    "support_nonsensitive": ["N2", "N5"]
  }}
]"""

PAIR_GENERATION_PROMPT = """You are generating synthetic training data for a binary privacy text classification task.

Use only the labeled support examples and the rubric below.

Sensitive examples:
{positive_examples}

Non-sensitive examples:
{negative_examples}

Rubric:
{rubric}

Task:
Generate one matched pair of new texts:
- one text whose correct label is Sensitive;
- one text whose correct label is Non-sensitive.

Requirements:
- Keep the two texts as similar as possible in topic, style, length, and surface cue.
- Change only the minimum relation, role, source, or context needed to move from one label to the other under the rubric.
- Make both texts natural and compatible with the support examples.
- Do not copy any support example.
- Do not mention dataset names, benchmark names, labels, or explanations inside the generated texts.

Return only JSON:
{{
  "sensitive_text": "...",
  "nonsensitive_text": "..."
}}"""

PAIR_VERIFICATION_PROMPT = """You are verifying synthetic data for a binary privacy text classification task.

Use only the labeled support examples and the rubric below.

Sensitive examples:
{positive_examples}

Non-sensitive examples:
{negative_examples}

Rubric:
{rubric}

Generated pair:
Sensitive candidate:
<<<{sensitive_text}>>>

Non-sensitive candidate:
<<<{nonsensitive_text}>>>

Check the pair.

Return only JSON:
{{
  "sensitive_label_correct": true,
  "nonsensitive_label_correct": true,
  "matched_pair": true,
  "copied_from_support": false,
  "accept": true,
  "reason": "short reason"
}}"""

ROUGE_L_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PROMPT_ARTIFACT_RE = re.compile(
    r"(?:<<<|>>>|^\s*(?:Sensitive|Non-sensitive)\s*:|"
    r"Classify one PII-detection text|Respond with exactly one label)",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _rouge_l_tokens(text: str) -> tuple[str, ...]:
    return tuple(ROUGE_L_TOKEN_RE.findall(text.lower()))


def _lcs_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    if not left or not right:
        return 0
    if len(right) > len(left):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        diagonal = 0
        for index, right_token in enumerate(right, start=1):
            above = previous[index]
            value = diagonal + 1 if left_token == right_token else max(above, current[-1])
            current.append(value)
            diagonal = above
        previous = current
    return previous[-1]


def _rouge_l_f1(candidate_tokens: tuple[str, ...], reference_tokens: tuple[str, ...]) -> float:
    if not candidate_tokens or not reference_tokens:
        return 0.0
    lcs = _lcs_length(candidate_tokens, reference_tokens)
    if lcs <= 0:
        return 0.0
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _max_rouge_l_f1(
    candidate_tokens: tuple[str, ...],
    reference_tokens_list: list[tuple[str, ...]],
) -> float:
    if not reference_tokens_list:
        return 0.0
    return max(_rouge_l_f1(candidate_tokens, tokens) for tokens in reference_tokens_list)


def _vllm_metrics_url(vllm_url: str) -> str:
    parsed = urllib.parse.urlsplit(vllm_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    metrics_path = f"{path.rstrip('/')}/metrics" if path else "/metrics"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, metrics_path, "", ""))


def _read_vllm_load_metrics(vllm_url: str, *, model: str, timeout: float) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(  # noqa: S310 - URL is configured vLLM endpoint.
            _vllm_metrics_url(vllm_url),
            timeout=timeout,
        ) as response:
            body = response.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError):
        return None

    metrics = {"running": 0.0, "waiting": 0.0}
    all_metrics = {"running": 0.0, "waiting": 0.0}
    matched_model = False
    for line in body.splitlines():
        if not line.startswith("vllm:num_requests_"):
            continue
        match = re.match(
            r"vllm:num_requests_(running|waiting)\{[^}]*\}\s+([0-9.eE+-]+)",
            line,
        )
        if not match:
            continue
        key = match.group(1)
        value = float(match.group(2))
        all_metrics[key] += value
        if model in line:
            metrics[key] += value
            matched_model = True
    return metrics if matched_model else all_metrics


def _preflight_vllm_capacity(
    *,
    vllm_url: str,
    model: str,
    generation_cfg: dict,
) -> dict[str, float] | None:
    if not bool(generation_cfg.get("preflight_vllm_metrics", True)):
        return None
    metrics = _read_vllm_load_metrics(
        vllm_url,
        model=model,
        timeout=float(generation_cfg.get("preflight_timeout", 5.0)),
    )
    if metrics is None:
        return None
    max_running = float(generation_cfg.get("preflight_max_running", 0))
    max_waiting = float(generation_cfg.get("preflight_max_waiting", 0))
    if metrics["running"] > max_running or metrics["waiting"] > max_waiting:
        raise RuntimeError(
            "vLLM endpoint is already busy or stuck before generation: "
            f"running={metrics['running']}, waiting={metrics['waiting']}, "
            f"allowed_running={max_running}, allowed_waiting={max_waiting}."
        )
    return metrics


def _support_records(seed_pool: list[dict]) -> tuple[list[dict], list[dict]]:
    positive_records: list[dict] = []
    negative_records: list[dict] = []
    label_counters: Counter[str] = Counter()
    for item in seed_pool:
        label = str(item["label"])
        if label not in LABELS:
            continue
        label_counters[label] += 1
        prefix = "S" if label == "Sensitive" else "N"
        example_id = f"{prefix}{label_counters[label]}"
        text = str(item["text"]).replace("<<<", "").replace(">>>", "").strip()
        record = {
            "id": example_id,
            "label": label,
            "text": text,
            "line": f"{example_id}: <<<{text}>>>",
        }
        if label == "Sensitive":
            positive_records.append(record)
        else:
            negative_records.append(record)
    if not positive_records or not negative_records:
        raise ValueError("seed pool must contain both Sensitive and Non-sensitive examples")
    return positive_records, negative_records


def _records_to_blocks(
    positive_records: list[dict],
    negative_records: list[dict],
) -> tuple[str, str, set[str], set[str]]:
    positive_lines = [str(item["line"]) for item in positive_records]
    negative_lines = [str(item["line"]) for item in negative_records]
    positive_ids = {str(item["id"]) for item in positive_records}
    negative_ids = {str(item["id"]) for item in negative_records}
    return "\n".join(positive_lines), "\n".join(negative_lines), positive_ids, negative_ids


def _rubric_contexts(
    *,
    positive_records: list[dict],
    negative_records: list[dict],
    mode: str,
    calls: int,
    rng: random.Random,
) -> list[dict]:
    if calls < 1:
        raise ValueError(f"rubric extraction calls must be positive, got {calls}")

    if mode == "subshot_3x3":
        if len(positive_records) < 3 or len(negative_records) < 3:
            raise ValueError("subshot_3x3 requires at least 3 examples per label")
        contexts = [
            (list(pos_items), list(neg_items))
            for pos_items, neg_items in product(
                combinations(positive_records, 3),
                combinations(negative_records, 3),
            )
        ]
        rng.shuffle(contexts)
        contexts = [contexts[index % len(contexts)] for index in range(calls)]
    elif mode == "adaptive_subshot_3x3":
        subset_size = min(3, len(positive_records), len(negative_records))
        if subset_size < 1:
            raise ValueError("adaptive_subshot_3x3 requires at least one example per label")
        contexts = [
            (list(pos_items), list(neg_items))
            for pos_items, neg_items in product(
                combinations(positive_records, subset_size),
                combinations(negative_records, subset_size),
            )
        ]
        rng.shuffle(contexts)
        contexts = [contexts[index % len(contexts)] for index in range(calls)]
    elif mode == "fullshot":
        contexts = [(positive_records, negative_records) for _ in range(calls)]
    else:
        raise ValueError(f"unknown rubric context mode: {mode}")

    out: list[dict] = []
    for index, (context_positive, context_negative) in enumerate(contexts, start=1):
        positive_examples, negative_examples, positive_ids, negative_ids = _records_to_blocks(
            context_positive,
            context_negative,
        )
        out.append(
            {
                "context_id": f"R{index:03d}",
                "context_index": index,
                "positive_examples": positive_examples,
                "negative_examples": negative_examples,
                "positive_ids": sorted(positive_ids),
                "negative_ids": sorted(negative_ids),
            }
        )
    return out


def _parse_json_payload(raw: str | None):
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
            return payload
        except json.JSONDecodeError:
            continue
    return None


def _json_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _short_prompt_value(value, max_chars: int = 700) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _parse_rubrics(
    raw: str | None,
    *,
    positive_ids: set[str],
    negative_ids: set[str],
    max_rubrics: int,
) -> list[dict]:
    payload = _parse_json_payload(raw)
    if isinstance(payload, dict):
        payload = payload.get("rubrics") or payload.get("cards")
    if not isinstance(payload, list):
        return []

    rubrics: list[dict] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        rubric_id = str(item.get("rubric_id") or item.get("card_id") or f"R{index}").strip()
        if not rubric_id or rubric_id in seen_ids:
            continue
        sensitive_condition = _short_prompt_value(item.get("sensitive_condition", ""))
        nonsensitive_condition = _short_prompt_value(item.get("nonsensitive_condition", ""))
        support_sensitive = [
            str(value).strip()
            for value in item.get("support_sensitive", [])
            if str(value).strip() in positive_ids
        ]
        support_nonsensitive = [
            str(value).strip()
            for value in item.get("support_nonsensitive", [])
            if str(value).strip() in negative_ids
        ]
        if not (
            sensitive_condition
            and nonsensitive_condition
            and support_sensitive
            and support_nonsensitive
        ):
            continue
        seen_ids.add(rubric_id)
        rubrics.append(
            {
                "rubric_id": rubric_id,
                "sensitive_condition": sensitive_condition,
                "nonsensitive_condition": nonsensitive_condition,
                "support_sensitive": support_sensitive,
                "support_nonsensitive": support_nonsensitive,
            }
        )
        if len(rubrics) >= max_rubrics:
            break
    return rubrics


def _rubric_text(rubric: dict) -> str:
    prompt_rubric = {
        "rubric_id": rubric.get("rubric_id", ""),
        "sensitive_condition": rubric.get("sensitive_condition", ""),
        "nonsensitive_condition": rubric.get("nonsensitive_condition", ""),
        "support_sensitive": rubric.get("support_sensitive", []),
        "support_nonsensitive": rubric.get("support_nonsensitive", []),
    }
    return json.dumps(prompt_rubric, ensure_ascii=False, sort_keys=True)


def _rubrics_from_existing_rows(rows: list[dict]) -> list[dict]:
    def as_string_list(value) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    rubrics_by_id: dict[str, dict] = {}
    for item in rows:
        rubric_id = str(item.get("_rubric_id", "")).strip()
        if not rubric_id or rubric_id in rubrics_by_id:
            continue
        sensitive_condition = str(item.get("_sensitive_condition", "")).strip()
        nonsensitive_condition = str(item.get("_nonsensitive_condition", "")).strip()
        support_sensitive = as_string_list(item.get("_support_sensitive", []))
        support_nonsensitive = as_string_list(item.get("_support_nonsensitive", []))
        if not (
            sensitive_condition
            and nonsensitive_condition
            and support_sensitive
            and support_nonsensitive
        ):
            continue
        rubrics_by_id[rubric_id] = {
            "rubric_id": rubric_id,
            "sensitive_condition": sensitive_condition,
            "nonsensitive_condition": nonsensitive_condition,
            "support_sensitive": support_sensitive,
            "support_nonsensitive": support_nonsensitive,
            "rubric_context_mode": item.get("_rubric_context_mode", ""),
            "rubric_context_id": item.get("_rubric_context_id", ""),
            "rubric_context_index": item.get("_rubric_context_index", 0),
            "rubric_context_sensitive_ids": as_string_list(
                item.get("_rubric_context_sensitive_ids", [])
            ),
            "rubric_context_nonsensitive_ids": as_string_list(
                item.get("_rubric_context_nonsensitive_ids", [])
            ),
        }
    return sorted(
        rubrics_by_id.values(),
        key=lambda rubric: (
            int(rubric.get("rubric_context_index", 0) or 0),
            str(rubric.get("rubric_id", "")),
        ),
    )


def _prompt_artifact_reject_reason(text: str) -> str | None:
    if PROMPT_ARTIFACT_RE.search(text):
        return "candidate contains prompt delimiters, label prefixes, or prompt wording"
    return None


def _clean_pair_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    text = text.strip().strip("\"'").strip()
    for _ in range(2):
        if text.startswith("<<<") and text.endswith(">>>"):
            text = text[3:-3].strip().strip("\"'").strip()
    if not 10 <= len(text) <= 500:
        return None
    if _prompt_artifact_reject_reason(text):
        return None
    return text


def _parse_generated_pair(raw: str | None) -> tuple[str, str] | None:
    payload = _parse_json_payload(raw)
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    sensitive_text = _clean_pair_text(payload.get("sensitive_text"))
    nonsensitive_text = _clean_pair_text(payload.get("nonsensitive_text"))
    if not sensitive_text or not nonsensitive_text:
        return None
    return sensitive_text, nonsensitive_text


def _parse_verification(raw: str | None) -> dict | None:
    payload = _parse_json_payload(raw)
    if not isinstance(payload, dict):
        return None
    required = [
        "sensitive_label_correct",
        "nonsensitive_label_correct",
        "matched_pair",
        "copied_from_support",
        "accept",
    ]
    parsed: dict[str, bool | str] = {}
    for key in required:
        value = _json_bool(payload.get(key))
        if value is None:
            return None
        parsed[key] = value
    parsed["reason"] = _short_prompt_value(payload.get("reason", ""))
    return parsed


def _looks_incomplete(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith(("...", "...")):
        return True
    if stripped.count('"') % 2 == 1:
        return True
    if stripped[-1] in {",", ";", ":", "-", "(", "["}:
        return True
    if stripped[-1] not in {".", "?", "!", '"', "'", ")", "]"}:
        return True
    opens = sum(stripped.count(ch) for ch in "([")
    closes = sum(stripped.count(ch) for ch in ")]")
    if opens > closes:
        return True
    words = re.findall(r"[A-Za-z']+", stripped.lower())
    if not words:
        return True
    return words[-1] in {
        "a",
        "an",
        "and",
        "at",
        "because",
        "but",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }


async def _complete_text(
    *,
    client,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    timeout: float,
    max_tokens: int,
    top_k: int | None = None,
    presence_penalty: float | None = None,
) -> str | None:
    def drain_exception(task: asyncio.Task) -> None:
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            return

    async with semaphore:
        try:
            extra_body: dict[str, object] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
            if top_k is not None:
                extra_body["top_k"] = int(top_k)
            request_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "extra_body": extra_body,
            }
            if presence_penalty is not None:
                request_kwargs["presence_penalty"] = float(presence_penalty)
            request = asyncio.create_task(client.chat.completions.create(**request_kwargs))
            request.add_done_callback(drain_exception)
            done, _ = await asyncio.wait({request}, timeout=max(timeout, 1.0))
            if request not in done:
                request.cancel()
                return None
            response = request.result()
        except Exception:
            return None
    content = response.choices[0].message.content
    return content.strip() if content else None


async def rubric_guided_contrastive_synthesis(
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
    """Generate matched Sensitive/Non-sensitive training pairs from induced rubrics."""

    method_name = "rubric_guided_contrastive_synthesis"
    prompt_id = "rubric_guided_contrastive_synthesis_v1"
    if n_samples % 2 != 0:
        raise ValueError(f"{method_name} requires an even n_samples, got {n_samples}")

    existing = read_jsonl(out_path) if out_path.exists() else []
    if existing:
        foreign_methods = {
            str(item.get("_method", ""))
            for item in existing
            if str(item.get("_method", "")) != method_name
        }
        if foreign_methods:
            raise RuntimeError(
                f"{out_path} contains rows from other methods {sorted(foreign_methods)}; "
                "use a fresh output root or --overwrite"
            )
        incompatible_rows = [
            index
            for index, item in enumerate(existing)
            if item.get("_prompt_id") != prompt_id
            or not item.get("_pair_id")
            or item.get("_pair_role") not in LABELS
            or not item.get("_rubric_id")
        ]
        if incompatible_rows:
            raise RuntimeError(
                f"{out_path} contains existing rows with incompatible metadata; "
                "use a fresh output root or --overwrite"
            )
        if len(existing) % 2 != 0:
            raise RuntimeError(f"{out_path} contains an odd number of rows")
    if len(existing) >= n_samples:
        return {
            "method": method_name,
            "prompt_id": prompt_id,
            "target_samples": n_samples,
            "num_written_total": len(existing),
            "generation_status": "complete",
            "completion_rate": 1.0,
        }

    generation_cfg = config.get("generation", {})
    method_cfg = config.get("rubric_guided_contrastive_synthesis", {})

    configured_attempt_cap = int(generation_cfg.get("attempt_cap", 20000))
    endpoint_failure_cap = int(
        generation_cfg.get("endpoint_failure_cap", max(16, concurrency * 4))
    )
    temperature = float(generation_cfg.get("temperature", 1.0))
    top_p = float(generation_cfg.get("top_p", 0.9))
    top_k_raw = generation_cfg.get("top_k")
    top_k = None if top_k_raw in {None, "", "off", "none", "null", 0} else int(top_k_raw)
    presence_penalty = float(generation_cfg.get("presence_penalty", 0.0) or 0.0)
    judge_temperature = float(method_cfg.get("verifier_temperature", 0.0))
    judge_top_p = float(method_cfg.get("verifier_top_p", top_p))
    timeout = float(generation_cfg.get("timeout", 30.0))

    rubric_extraction_calls = int(method_cfg.get("rubric_extraction_calls", 25))
    rubric_extraction_k = int(method_cfg.get("rubric_extraction_k", 1))
    rubric_extraction_temperature = float(method_cfg.get("rubric_extraction_temperature", 1.0))
    rubric_context_mode = str(method_cfg.get("rubric_context_mode", "subshot_3x3"))
    attempts_per_target_pair = max(1, int(method_cfg.get("pair_attempts_per_target_pair", 5)))
    min_completion_rate = float(method_cfg.get("min_completion_rate", 0.95))
    rouge_l_threshold = float(method_cfg.get("rouge_l_threshold", 0.95))
    support_scope = str(method_cfg.get("support_scope", "all_seed"))
    rubric_max_tokens = int(method_cfg.get("rubric_max_tokens", 900))
    pair_max_tokens = int(method_cfg.get("pair_max_tokens", 420))
    verifier_max_tokens = int(method_cfg.get("verifier_max_tokens", 260))
    if support_scope not in {"all_seed", "rubric_context"}:
        raise ValueError(f"unknown support scope: {support_scope}")

    preflight_metrics = _preflight_vllm_capacity(
        vllm_url=vllm_url,
        model=model,
        generation_cfg=generation_cfg,
    )

    seed_pool = load_seed_pool(seed_pool_dir)
    positive_records, negative_records = _support_records(seed_pool)
    positive_examples, negative_examples, _, _ = _records_to_blocks(
        positive_records,
        negative_records,
    )
    records_by_id = {
        str(record["id"]): record for record in [*positive_records, *negative_records]
    }
    seed_keys = {_normalize(str(item["text"])) for item in seed_pool}
    seen = {_normalize(str(item["text"])) for item in existing}
    seed_rouge_tokens = [_rouge_l_tokens(str(item["text"])) for item in seed_pool]
    accepted_rouge_tokens = [
        _rouge_l_tokens(str(item["text"]))
        for item in existing
        if str(item.get("text", "")).strip()
    ]

    label_counts = Counter(str(item.get("label")) for item in existing)
    target_label_count = n_samples // 2
    overfull_labels = {
        label: count
        for label, count in label_counts.items()
        if count > target_label_count
    }
    if overfull_labels:
        raise RuntimeError(
            f"{out_path} already contains more items than the balanced target: "
            f"{overfull_labels}"
        )

    from openai import AsyncOpenAI
    from tqdm.asyncio import tqdm

    client = AsyncOpenAI(base_url=vllm_url, api_key="dummy", max_retries=0)
    semaphore = asyncio.Semaphore(concurrency)

    rubric_contexts = _rubric_contexts(
        positive_records=positive_records,
        negative_records=negative_records,
        mode=rubric_context_mode,
        calls=rubric_extraction_calls,
        rng=random.Random(seed + 1701),
    )
    rubric_extraction_stats = Counter()
    existing_rubrics = _rubrics_from_existing_rows(existing)

    async def extract_rubrics_for_context(context: dict) -> list[dict]:
        prompt = RUBRIC_INDUCTION_PROMPT.format(
            positive_examples=context["positive_examples"],
            negative_examples=context["negative_examples"],
            k=rubric_extraction_k,
        )
        raw = await _complete_text(
            client=client,
            model=model,
            prompt=prompt,
            semaphore=semaphore,
            temperature=rubric_extraction_temperature,
            top_p=judge_top_p,
            timeout=timeout,
            max_tokens=rubric_max_tokens,
        )
        parsed = _parse_rubrics(
            raw,
            positive_ids=set(context["positive_ids"]),
            negative_ids=set(context["negative_ids"]),
            max_rubrics=rubric_extraction_k,
        )
        if parsed:
            rubric_extraction_stats["valid_contexts"] += 1
        else:
            rubric_extraction_stats["empty_or_invalid_contexts"] += 1
        rubrics_for_context: list[dict] = []
        for local_index, rubric in enumerate(parsed, start=1):
            local_rubric_id = str(rubric["rubric_id"])
            rubric["rubric_id"] = f"{context['context_id']}_{local_rubric_id}"
            rubric["rubric_context_mode"] = rubric_context_mode
            rubric["rubric_context_id"] = context["context_id"]
            rubric["rubric_context_index"] = context["context_index"]
            rubric["rubric_context_local_rubric_id"] = local_rubric_id
            rubric["rubric_context_local_rubric_index"] = local_index
            rubric["rubric_context_sensitive_ids"] = list(context["positive_ids"])
            rubric["rubric_context_nonsensitive_ids"] = list(context["negative_ids"])
            rubrics_for_context.append(rubric)
        rubric_extraction_stats["valid_rubrics"] += len(rubrics_for_context)
        return rubrics_for_context

    if len(existing_rubrics) >= len(rubric_contexts):
        rubrics = existing_rubrics
        rubric_extraction_stats["reused_existing_rubrics"] = len(existing_rubrics)
        rubric_extraction_stats["valid_contexts"] = len(existing_rubrics)
        rubric_extraction_stats["valid_rubrics"] = len(existing_rubrics)
    else:
        extracted_batches = await asyncio.gather(
            *(extract_rubrics_for_context(context) for context in rubric_contexts)
        )
        extracted = [rubric for batch in extracted_batches for rubric in batch]
        rubrics_by_id = {str(rubric["rubric_id"]): rubric for rubric in extracted}
        if existing_rubrics:
            rubric_extraction_stats["reused_existing_rubrics"] = len(existing_rubrics)
            rubrics_by_id.update({str(rubric["rubric_id"]): rubric for rubric in existing_rubrics})
        rubrics = sorted(
            rubrics_by_id.values(),
            key=lambda rubric: (
                int(rubric.get("rubric_context_index", 0) or 0),
                str(rubric.get("rubric_id", "")),
            ),
        )
    if not rubrics:
        raise RuntimeError(f"{method_name} could not extract any valid rubrics")

    def support_blocks_for_rubric(rubric: dict) -> tuple[str, str]:
        if support_scope == "all_seed":
            return positive_examples, negative_examples
        sensitive_ids = [
            str(value).strip()
            for value in rubric.get("rubric_context_sensitive_ids", [])
            if str(value).strip()
        ]
        nonsensitive_ids = [
            str(value).strip()
            for value in rubric.get("rubric_context_nonsensitive_ids", [])
            if str(value).strip()
        ]
        missing = [
            example_id
            for example_id in [*sensitive_ids, *nonsensitive_ids]
            if example_id not in records_by_id
        ]
        if missing:
            raise RuntimeError(f"{method_name} found unknown support ids {missing}")
        context_positive = [records_by_id[example_id] for example_id in sensitive_ids]
        context_negative = [records_by_id[example_id] for example_id in nonsensitive_ids]
        context_positive_examples, context_negative_examples, _, _ = _records_to_blocks(
            context_positive,
            context_negative,
        )
        return context_positive_examples, context_negative_examples

    total_pairs = n_samples // 2
    rng = random.Random(seed)
    existing_pair_ids = {str(item.get("_pair_id")) for item in existing if item.get("_pair_id")}
    existing_pair_count = len(existing_pair_ids)
    base_pairs = total_pairs // len(rubrics)
    extra_pairs = total_pairs % len(rubrics)
    target_rubric_counts = Counter(
        {
            rubric["rubric_id"]: base_pairs + (1 if index < extra_pairs else 0)
            for index, rubric in enumerate(rubrics)
        }
    )
    existing_rubric_counts: Counter[str] = Counter(
        str(item.get("_rubric_id"))
        for item in existing
        if item.get("_pair_role") == "Sensitive"
    )

    rubric_attempt_budgets: Counter[str] = Counter()
    pending: list[dict] = []
    for rubric in rubrics:
        rubric_id = str(rubric["rubric_id"])
        remaining = target_rubric_counts[rubric_id] - existing_rubric_counts[rubric_id]
        if remaining <= 0:
            continue
        rubric_attempt_budgets[rubric_id] = remaining * attempts_per_target_pair
        pending.append({"rubric": rubric, "retries": 0})
    attempt_cap = min(configured_attempt_cap, sum(rubric_attempt_budgets.values()))
    rng.shuffle(pending)

    accepted: list[dict] = []
    stats = Counter()
    rubric_attempts: Counter[str] = Counter()
    rubric_accepts: Counter[str] = Counter()
    rubric_rejects: Counter[str] = Counter()
    retry_counts: Counter[int] = Counter()
    next_pair_index = existing_pair_count
    max_rouge_l_to_seed_observed = 0.0
    max_rouge_l_to_generated_observed = 0.0
    pbar = tqdm(total=n_samples - len(existing), desc=method_name)

    def flush() -> None:
        nonlocal accepted
        if accepted:
            append_jsonl(accepted, out_path)
            accepted = []

    async def verify_pair(
        *,
        rubric: dict,
        sensitive_text: str,
        nonsensitive_text: str,
    ) -> dict | None:
        rubric_positive_examples, rubric_negative_examples = support_blocks_for_rubric(rubric)
        prompt = PAIR_VERIFICATION_PROMPT.format(
            positive_examples=rubric_positive_examples,
            negative_examples=rubric_negative_examples,
            rubric=_rubric_text(rubric),
            sensitive_text=sensitive_text,
            nonsensitive_text=nonsensitive_text,
        )
        raw = await _complete_text(
            client=client,
            model=model,
            prompt=prompt,
            semaphore=semaphore,
            temperature=judge_temperature,
            top_p=judge_top_p,
            timeout=timeout,
            max_tokens=verifier_max_tokens,
        )
        return _parse_verification(raw)

    async def run_one(task: dict) -> dict | None:
        nonlocal max_rouge_l_to_generated_observed
        nonlocal max_rouge_l_to_seed_observed
        nonlocal next_pair_index
        rubric = task["rubric"]
        rubric_id = str(rubric["rubric_id"])
        task["retries"] = int(task.get("retries", 0)) + 1
        stats["attempts"] += 1
        stats[f"attempt_rubric_{rubric_id}"] += 1
        rubric_attempts[rubric_id] += 1

        rubric_positive_examples, rubric_negative_examples = support_blocks_for_rubric(rubric)
        prompt = PAIR_GENERATION_PROMPT.format(
            positive_examples=rubric_positive_examples,
            negative_examples=rubric_negative_examples,
            rubric=_rubric_text(rubric),
        )
        raw_pair = await _complete_text(
            client=client,
            model=model,
            prompt=prompt,
            semaphore=semaphore,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            timeout=timeout,
            max_tokens=pair_max_tokens,
        )
        pair = _parse_generated_pair(raw_pair)
        if pair is None:
            stats["pair_parse_failed"] += 1
            rubric_rejects[rubric_id] += 1
            return task
        sensitive_text, nonsensitive_text = pair

        if _looks_incomplete(sensitive_text) or _looks_incomplete(nonsensitive_text):
            stats["incomplete_rejected"] += 1
            rubric_rejects[rubric_id] += 1
            return task

        sensitive_key = _normalize(sensitive_text)
        nonsensitive_key = _normalize(nonsensitive_text)
        if sensitive_key == nonsensitive_key:
            stats["pair_duplicate_rejected"] += 1
            rubric_rejects[rubric_id] += 1
            return task
        if sensitive_key in seed_keys or nonsensitive_key in seed_keys:
            stats["seed_copy_rejected"] += 1
            rubric_rejects[rubric_id] += 1
            return task
        if sensitive_key in seen or nonsensitive_key in seen:
            stats["duplicate_rejected"] += 1
            rubric_rejects[rubric_id] += 1
            return task

        sensitive_rouge_tokens = _rouge_l_tokens(sensitive_text)
        nonsensitive_rouge_tokens = _rouge_l_tokens(nonsensitive_text)
        sensitive_seed_rouge = _max_rouge_l_f1(sensitive_rouge_tokens, seed_rouge_tokens)
        nonsensitive_seed_rouge = _max_rouge_l_f1(nonsensitive_rouge_tokens, seed_rouge_tokens)
        max_rouge_l_to_seed_observed = max(
            max_rouge_l_to_seed_observed,
            sensitive_seed_rouge,
            nonsensitive_seed_rouge,
        )
        if sensitive_seed_rouge >= rouge_l_threshold or nonsensitive_seed_rouge >= rouge_l_threshold:
            stats["rouge_l_seed_copy_rejected"] += 1
            rubric_rejects[rubric_id] += 1
            return task

        sensitive_generated_rouge = _max_rouge_l_f1(
            sensitive_rouge_tokens,
            accepted_rouge_tokens,
        )
        nonsensitive_generated_rouge = _max_rouge_l_f1(
            nonsensitive_rouge_tokens,
            accepted_rouge_tokens,
        )
        max_rouge_l_to_generated_observed = max(
            max_rouge_l_to_generated_observed,
            sensitive_generated_rouge,
            nonsensitive_generated_rouge,
        )
        if (
            sensitive_generated_rouge >= rouge_l_threshold
            or nonsensitive_generated_rouge >= rouge_l_threshold
        ):
            stats["rouge_l_generated_duplicate_rejected"] += 1
            rubric_rejects[rubric_id] += 1
            return task

        verification = await verify_pair(
            rubric=rubric,
            sensitive_text=sensitive_text,
            nonsensitive_text=nonsensitive_text,
        )
        if verification is None:
            stats["verifier_parse_failed"] += 1
            rubric_rejects[rubric_id] += 1
            return task
        if not bool(verification["sensitive_label_correct"]):
            stats["verifier_sensitive_label_rejected"] += 1
        if not bool(verification["nonsensitive_label_correct"]):
            stats["verifier_nonsensitive_label_rejected"] += 1
        if not bool(verification["matched_pair"]):
            stats["verifier_matched_pair_rejected"] += 1
        if bool(verification["copied_from_support"]):
            stats["verifier_copied_from_support_rejected"] += 1
        if not bool(verification["accept"]):
            stats["verifier_accept_false"] += 1
        if not (
            verification["sensitive_label_correct"]
            and verification["nonsensitive_label_correct"]
            and verification["matched_pair"]
            and not verification["copied_from_support"]
            and verification["accept"]
        ):
            stats["verifier_rejected"] += 1
            rubric_rejects[rubric_id] += 1
            return task

        next_pair_index += 1
        pair_id = f"{method_name}_seed{seed}_pair{next_pair_index:06d}"
        retry_count = int(task.get("retries", 0))
        base_meta = {
            "_method": method_name,
            "_prompt_id": prompt_id,
            "_pair_id": pair_id,
            "_rubric_id": rubric_id,
            "_sensitive_condition": rubric["sensitive_condition"],
            "_nonsensitive_condition": rubric["nonsensitive_condition"],
            "_support_sensitive": rubric["support_sensitive"],
            "_support_nonsensitive": rubric["support_nonsensitive"],
            "_rubric_context_mode": rubric.get("rubric_context_mode", rubric_context_mode),
            "_rubric_context_id": rubric.get("rubric_context_id", ""),
            "_rubric_context_index": rubric.get("rubric_context_index", 0),
            "_rubric_context_sensitive_ids": rubric.get("rubric_context_sensitive_ids", []),
            "_rubric_context_nonsensitive_ids": rubric.get(
                "rubric_context_nonsensitive_ids",
                [],
            ),
            "_verifier_reason": verification.get("reason", ""),
            "_retry_count": retry_count,
        }
        accepted.extend(
            [
                {
                    "text": sensitive_text,
                    "label": "Sensitive",
                    "_pair_role": "Sensitive",
                    **base_meta,
                },
                {
                    "text": nonsensitive_text,
                    "label": "Non-sensitive",
                    "_pair_role": "Non-sensitive",
                    **base_meta,
                },
            ]
        )
        seen.add(sensitive_key)
        seen.add(nonsensitive_key)
        accepted_rouge_tokens.append(sensitive_rouge_tokens)
        accepted_rouge_tokens.append(nonsensitive_rouge_tokens)
        stats["accepted_pairs"] += 1
        stats["accepted_items"] += 2
        rubric_accepts[rubric_id] += 1
        retry_counts[retry_count] += 1
        pbar.update(2)
        if len(accepted) >= 64:
            flush()
        return {"rubric": rubric, "retries": 0}

    attempts = 0
    consecutive_endpoint_failures = 0
    while pending and attempts < attempt_cap:
        batch_size = min(len(pending), max(1, concurrency), attempt_cap - attempts)
        batch = pending[:batch_size]
        pending = pending[batch_size:]
        attempts += batch_size
        accepted_pairs_before_batch = stats["accepted_pairs"]
        endpoint_failures_before_batch = (
            stats["pair_parse_failed"] + stats["verifier_parse_failed"]
        )
        completed = [
            task
            for task in await asyncio.gather(*(run_one(task) for task in batch))
            if task
        ]
        accepted_pairs_in_batch = stats["accepted_pairs"] - accepted_pairs_before_batch
        endpoint_failures_in_batch = (
            stats["pair_parse_failed"]
            + stats["verifier_parse_failed"]
            - endpoint_failures_before_batch
        )
        if accepted_pairs_in_batch:
            consecutive_endpoint_failures = 0
        elif endpoint_failures_in_batch:
            consecutive_endpoint_failures += endpoint_failures_in_batch
        for task in completed:
            rubric_id = str(task["rubric"]["rubric_id"])
            total_rubric_accepts = existing_rubric_counts[rubric_id] + rubric_accepts[rubric_id]
            target_reached = total_rubric_accepts >= target_rubric_counts[rubric_id]
            budget_remaining = rubric_attempts[rubric_id] < rubric_attempt_budgets[rubric_id]
            if not target_reached and budget_remaining:
                pending.append(task)
        rng.shuffle(pending)
        flush()
        endpoint_failures = stats["pair_parse_failed"] + stats["verifier_parse_failed"]
        if (
            stats["accepted_pairs"] == 0
            and endpoint_failures >= endpoint_failure_cap
        ) or consecutive_endpoint_failures >= endpoint_failure_cap:
            pbar.close()
            raise RuntimeError(
                f"{method_name} hit {consecutive_endpoint_failures} consecutive "
                f"endpoint-like failures ({endpoint_failures} total); check the LLM endpoint."
            )

    flush()
    pbar.close()
    if not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.touch()

    total_items = read_jsonl(out_path)
    total_written = len(total_items)
    generation_status = "complete" if total_written >= n_samples else "incomplete_rubric_budget"
    completion_rate = total_written / n_samples if n_samples else 1.0
    min_required = int(n_samples * min_completion_rate)

    label_distribution = Counter(str(item.get("label")) for item in total_items)
    pair_ids = {str(item.get("_pair_id")) for item in total_items if item.get("_pair_id")}
    rubric_distribution = Counter(
        str(item.get("_rubric_id"))
        for item in total_items
        if item.get("_pair_role") == "Sensitive"
    )
    target_rubric_ids = [
        rubric_id for rubric_id, target in target_rubric_counts.items() if target > 0
    ]
    completed_rubric_ids = [
        rubric_id
        for rubric_id in target_rubric_ids
        if rubric_distribution.get(rubric_id, 0) >= target_rubric_counts[rubric_id]
    ]
    budget_exhausted_rubric_ids = [
        rubric_id
        for rubric_id in target_rubric_ids
        if rubric_distribution.get(rubric_id, 0) < target_rubric_counts[rubric_id]
        and rubric_attempts.get(rubric_id, 0) >= rubric_attempt_budgets.get(rubric_id, 0) > 0
    ]
    rubric_completion_rate = (
        len(completed_rubric_ids) / len(target_rubric_ids) if target_rubric_ids else 1.0
    )

    return {
        "method": method_name,
        "prompt_id": prompt_id,
        "seed_pool": str(seed_pool_dir),
        "num_seed_items": len(seed_pool),
        "target_samples": n_samples,
        "target_pairs": total_pairs,
        "num_existing_start": len(existing),
        "num_written_total": total_written,
        "num_pairs_total": len(pair_ids),
        "num_generated_this_run": total_written - len(existing),
        "attempt_cap": attempt_cap,
        "attempt_cap_configured": configured_attempt_cap,
        "pair_attempts_per_target_pair": attempts_per_target_pair,
        "rubric_budget_total": sum(rubric_attempt_budgets.values()),
        "endpoint_failure_cap": endpoint_failure_cap,
        "consecutive_endpoint_failures_final": consecutive_endpoint_failures,
        "preflight_vllm_metrics": preflight_metrics,
        "attempts": attempts,
        "generation_status": generation_status,
        "completion_rate": completion_rate,
        "min_completion_rate": min_completion_rate,
        "min_required_items": min_required,
        "acceptance_policy": "exact_plus_rouge_l_plus_llm_verifier",
        "rouge_l_threshold": rouge_l_threshold,
        "support_scope": support_scope,
        "max_rouge_l_to_seed_observed": max_rouge_l_to_seed_observed,
        "max_rouge_l_to_generated_observed": max_rouge_l_to_generated_observed,
        "allow_incomplete_training_data": (
            generation_status == "incomplete_rubric_budget"
            and completion_rate >= min_completion_rate
        ),
        "rubric_completion_rate": rubric_completion_rate,
        "target_rubric_count": len(target_rubric_ids),
        "completed_rubric_count": len(completed_rubric_ids),
        "budget_exhausted_rubric_count": len(budget_exhausted_rubric_ids),
        "budget_exhausted_rubric_ids": budget_exhausted_rubric_ids,
        "pair_acceptance_rate": float(stats["accepted_pairs"]) / attempts if attempts else 0.0,
        "stats": dict(stats),
        "label_distribution": dict(label_distribution),
        "rubrics": rubrics,
        "rubric_context_mode": rubric_context_mode,
        "rubric_extraction_calls_requested": len(rubric_contexts),
        "rubric_extraction_calls_valid": int(rubric_extraction_stats["valid_contexts"]),
        "rubric_extraction_rubrics_valid": int(rubric_extraction_stats["valid_rubrics"]),
        "rubric_extraction_stats": dict(rubric_extraction_stats),
        "rubric_extraction_k": rubric_extraction_k,
        "rubric_extraction_temperature": rubric_extraction_temperature,
        "target_rubric_pair_distribution": dict(target_rubric_counts),
        "target_rubric_attempt_budget_distribution": dict(rubric_attempt_budgets),
        "rubric_attempts_this_run": dict(rubric_attempts),
        "accepted_rubric_pair_distribution": dict(rubric_distribution),
        "rubric_accepts_this_run": dict(rubric_accepts),
        "rubric_rejects_this_run": dict(rubric_rejects),
        "accepted_pair_retry_distribution": dict(retry_counts),
        "model": model,
        "vllm_url": vllm_url,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "generation_source": "fresh_vllm_generation",
        "fresh_vllm_generation": True,
        "uses_dataset_name_in_prompt": False,
        "prompt_policy": (
            "Rubric-Guided Contrastive Synthesis. The LLM first induces rubrics "
            "from labeled seed examples, then generates matched Sensitive/Non-sensitive "
            "pairs that differ by the rubric's label-defining evidence."
        ),
        "hard_reject_policy": (
            "Rejected attempts are not used as data. Rejections include JSON parse "
            "failure, prompt artifacts, incomplete text, exact seed copies, exact "
            "generated duplicates, ROUGE-L near copies, verifier parse failure, verifier "
            "label disagreement, copied-from-support detection, or failure to form a "
            "matched pair."
        ),
        "json_max_tokens": {
            "rubric": rubric_max_tokens,
            "pair": pair_max_tokens,
            "verifier": verifier_max_tokens,
        },
    }
