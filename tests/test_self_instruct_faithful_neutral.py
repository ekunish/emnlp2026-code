from pii_bench.self_instruct_faithful_neutral import (
    NEUTRAL_SEED_INSTRUCTIONS,
    _choose_instruction_demonstrations,
    _clean_instruction,
    _instruction_is_novel,
    _parse_instance_response,
)


def test_neutral_seed_pool_has_six_dataset_independent_instructions():
    assert len(NEUTRAL_SEED_INSTRUCTIONS) == 6
    joined = " ".join(NEUTRAL_SEED_INSTRUCTIONS).lower()
    assert "spedac" not in joined
    assert "spy" not in joined


def test_instruction_demonstrations_reuse_accepted_instructions():
    import random

    generated = ["Decide whether a sentence contains an accepted aspect."]
    selected = _choose_instruction_demonstrations(
        list(NEUTRAL_SEED_INSTRUCTIONS), generated, random.Random(42)
    )
    assert len(selected) == 7
    assert generated[0] in selected


def test_instruction_filter_uses_strict_rouge_threshold():
    instruction = NEUTRAL_SEED_INSTRUCTIONS[0]
    assert not _instruction_is_novel(instruction, list(NEUTRAL_SEED_INSTRUCTIONS))
    novel = "Decide whether a sentence presents a precise date tied to a private event."
    assert _instruction_is_novel(novel, list(NEUTRAL_SEED_INSTRUCTIONS))


def test_instruction_and_instance_parsers():
    assert _clean_instruction("too short") is None
    assert _clean_instruction("Decide whether a sentence contains a private disclosure.")
    assert _parse_instance_response('{"text": "I changed my home address yesterday."}')
    assert _parse_instance_response('{"other": "missing"}') is None
