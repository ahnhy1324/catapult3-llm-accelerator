from __future__ import annotations

import math

import torch

from fixed_byte_eval import make_window_plan, normalize_text_rows, score_fixed_bytes, utf8_prefix


class _Tokenizer:
    bos_token_id = 7
    eos_token_id = 8

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return {"input_ids": torch.tensor([[ord(char) % 11 for char in text]], dtype=torch.long)}


class _Model:
    def __call__(self, *, input_ids, use_cache, **_kwargs):
        vocab = 16
        logits = torch.zeros((*input_ids.shape, vocab), dtype=torch.float32)
        target = (input_ids + 1) % vocab
        logits.scatter_(-1, target.unsqueeze(-1), 2.0)
        return type("Output", (), {"logits": logits})()


def test_window_plan_scores_every_target_once_with_overlap():
    windows = make_window_plan(4097, 512)
    targets = [position for window in windows for position in range(window.target_start, window.target_end)]
    assert targets == list(range(1, 4098))
    assert all(window.input_end - window.input_start <= 512 for window in windows)
    assert all(window.input_start < window.target_start for window in windows)


def test_context_512_and_2048_use_same_target_span():
    small = make_window_plan(5000, 512)
    large = make_window_plan(5000, 2048)
    assert sum(window.target_count for window in small) == 5000
    assert sum(window.target_count for window in large) == 5000
    assert small[-1].target_end == large[-1].target_end == 5001


def test_utf8_normalization_and_prefix_do_not_split_codepoint():
    normalized = normalize_text_rows(["\ufeffalpha\r\n", "한글\rbeta"])
    assert normalized.decode("utf-8") == "alpha\n\n한글\nbeta"
    prefix = utf8_prefix("a한b".encode("utf-8"), 3)
    assert prefix == b"a"


def test_scorer_reports_bpb_and_scores_all_content_tokens():
    corpus = b"abcdefghi"
    metrics, cache = score_fixed_bytes(
        _Model(), _Tokenizer(), corpus, torch.device("cpu"), context=5, stride=2
    )
    assert cache == []
    assert metrics["scored_tokens"] == len(corpus)
    assert metrics["scored_utf8_bytes"] == len(corpus)
    assert metrics["window_count"] > 1
    assert math.isfinite(metrics["bits_per_byte"])
    assert metrics["bytes_per_token"] == 1.0
