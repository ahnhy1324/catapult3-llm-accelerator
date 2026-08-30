from __future__ import annotations

from common import generation_health_flags


def test_partial_prompt_prefix_is_flagged_as_echo():
    flags = generation_health_flags(
        "Write a correct Python function named is_prime that returns whether an integer is prime.",
        "Write a correct Python function named is_prime that returns whether an integer is",
        list(range(16)),
    )
    assert "PROMPT_ECHO" in flags


def test_two_identical_generated_halves_are_flagged():
    flags = generation_health_flags(
        "What is the capital of France?",
        "Paris is the capital. Paris is the capital.",
        [1, 2, 3, 4, 1, 2, 3, 4],
    )
    assert "REPEATED_SEQUENCE_X2" in flags


def test_normal_short_answer_is_not_flagged():
    assert generation_health_flags(
        "What is the capital of France?",
        "Paris is the capital of France.",
        [1, 2, 3, 4, 5, 6],
    ) == []


def test_unicode_replacement_character_is_flagged():
    flags = generation_health_flags("질문", "잘못된 출력 �", [1, 2, 3])
    assert "UNICODE_REPLACEMENT_CHARACTER" in flags
