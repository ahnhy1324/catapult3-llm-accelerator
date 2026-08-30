#!/usr/bin/env python3
"""Tokenizer-fair fixed-UTF-8-byte evaluation and sliding-window scoring.

Both model backends receive the exact same normalized UTF-8 byte span.  Each
tokenizer may produce a different number of tokens; cross-model comparison is
therefore based on bits per source byte (BPB), while PPL ratios remain
within-model degradation metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ScoreWindow:
    input_start: int
    input_end: int
    target_start: int
    target_end: int

    @property
    def target_count(self) -> int:
        return self.target_end - self.target_start


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text_rows(rows: Iterable[str]) -> bytes:
    """Canonicalize BOM/newlines and join source rows deterministically."""
    normalized: list[str] = []
    first = True
    for value in rows:
        text = str(value).replace("\r\n", "\n").replace("\r", "\n")
        if first:
            text = text.removeprefix("\ufeff")
            first = False
        normalized.append(text)
    return "\n".join(normalized).encode("utf-8")


def utf8_prefix(source: bytes, maximum_bytes: int) -> bytes:
    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    if len(source) <= maximum_bytes:
        source.decode("utf-8", errors="strict")
        return source
    end = maximum_bytes
    while end > 0:
        try:
            return source[:end].decode("utf-8", errors="strict").encode("utf-8")
        except UnicodeDecodeError:
            end -= 1
    raise ValueError("could not find a valid UTF-8 prefix")


def prepare_parquet_corpus(source_parquet: Path, output: Path, *, maximum_bytes: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(source_parquet, columns=["text"])
    normalized = normalize_text_rows(value or "" for value in table.column("text").to_pylist())
    corpus = utf8_prefix(normalized, maximum_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(corpus)
    return {
        "source_file": source_parquet.name,
        "source_sha256": sha256_bytes(source_parquet.read_bytes()),
        "normalization": "UTF8_STRICT; REMOVE_ONE_LEADING_BOM; CRLF_AND_CR_TO_LF; JOIN_ROWS_WITH_LF",
        "normalized_full_bytes": len(normalized),
        "selected_utf8_bytes": len(corpus),
        "selected_sha256": sha256_bytes(corpus),
        "selection": f"first <= {maximum_bytes} bytes without splitting a UTF-8 codepoint",
        "evidence_tag": "MEASURED_MODEL_FILE",
    }


def make_window_plan(content_tokens: int, context: int, *, stride: int | None = None) -> list[ScoreWindow]:
    """Plan overlapping windows that score each content token exactly once.

    Token position zero is a BOS/EOS warmup prefix and is never scored.
    Content token positions are therefore ``[1, content_tokens + 1)``.
    """
    if content_tokens <= 0:
        raise ValueError("content_tokens must be positive")
    if context < 2:
        raise ValueError("context must be at least two tokens")
    stride = stride or context // 2
    if stride <= 0 or stride >= context:
        raise ValueError("stride must be in [1, context)")
    windows: list[ScoreWindow] = []
    target_start = 1
    stream_end = content_tokens + 1
    while target_start < stream_end:
        target_end = min(target_start + stride, stream_end)
        input_end = target_end
        input_start = max(0, input_end - context)
        if input_start >= target_start:
            raise AssertionError("window has no prefix token for its first target")
        windows.append(ScoreWindow(input_start, input_end, target_start, target_end))
        target_start = target_end
    if sum(window.target_count for window in windows) != content_tokens:
        raise AssertionError("window plan did not score every content token once")
    for left, right in zip(windows, windows[1:]):
        if left.target_end != right.target_start:
            raise AssertionError("duplicate or missing target across window boundary")
    return windows


def prefix_token_id(tokenizer: Any) -> tuple[int, str]:
    if tokenizer.bos_token_id is not None:
        return int(tokenizer.bos_token_id), "BOS"
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id), "EOS_AS_WARMUP_WHEN_BOS_UNAVAILABLE"
    raise ValueError("tokenizer has neither BOS nor EOS for first-byte-span token scoring")


def tokenize_fixed_bytes(tokenizer: Any, corpus: bytes) -> tuple[torch.Tensor, dict[str, Any]]:
    text = corpus.decode("utf-8", errors="strict")
    content = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].long()
    if content.numel() == 0:
        raise ValueError("fixed-byte corpus tokenized to zero tokens")
    prefix_id, prefix_kind = prefix_token_id(tokenizer)
    stream = torch.cat([torch.tensor([prefix_id], dtype=torch.long), content])
    return stream, {
        "source_utf8_bytes": len(corpus),
        "source_sha256": sha256_bytes(corpus),
        "content_tokens": int(content.numel()),
        "warmup_tokens": 1,
        "warmup_kind": prefix_kind,
        "bos_scored": False,
        "eos_appended": False,
        "bytes_per_token": len(corpus) / int(content.numel()),
        "tokenizer_expansion_ratio_tokens_per_byte": int(content.numel()) / len(corpus),
    }


def score_fixed_bytes(
    model: Any,
    tokenizer: Any,
    corpus: bytes,
    device: torch.device,
    *,
    context: int,
    stride: int | None = None,
    cache_factory: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stream, tokenization = tokenize_fixed_bytes(tokenizer, corpus)
    plan = make_window_plan(tokenization["content_tokens"], context, stride=stride)
    total_nll = 0.0
    total_targets = 0
    cache_stats: list[dict[str, Any]] = []
    with torch.inference_mode():
        for window in plan:
            input_ids = stream[window.input_start : window.input_end].unsqueeze(0).to(device)
            local_target_start = window.target_start - window.input_start
            labels = input_ids.clone()
            labels[:, :local_target_start] = -100
            kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": cache_factory is not None}
            cache = cache_factory() if cache_factory else None
            if cache is not None:
                kwargs["past_key_values"] = cache
            output = model(**kwargs)
            shift_logits = output.logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            count = int((shift_labels != -100).sum().item())
            if count != window.target_count:
                raise RuntimeError(f"window scored {count} targets, expected {window.target_count}")
            total_nll += float(nll.item())
            total_targets += count
            if cache is not None and hasattr(cache, "stats_dict"):
                cache_stats.append(cache.stats_dict())
    if total_targets != tokenization["content_tokens"]:
        raise RuntimeError(f"scored {total_targets} targets, expected {tokenization['content_tokens']}")
    mean_nll = total_nll / total_targets
    return {
        "total_nll": total_nll,
        "mean_token_nll": mean_nll,
        "model_internal_perplexity": math.exp(min(mean_nll, 80.0)),
        "bits_per_byte": total_nll / (len(corpus) * math.log(2.0)),
        "scored_utf8_bytes": len(corpus),
        "scored_tokens": total_targets,
        "bytes_per_token": tokenization["bytes_per_token"],
        "tokenizer_expansion_ratio_tokens_per_byte": tokenization[
            "tokenizer_expansion_ratio_tokens_per_byte"
        ],
        "context_tokens": context,
        "stride_tokens": stride or context // 2,
        "window_count": len(plan),
        "window_policy": "OVERLAP_SLIDING_WINDOW_TARGETS_SCORED_EXACTLY_ONCE",
        "context_semantics": "each window resets backend state but retains up to context-stride overlapping prefix tokens",
        "bos_eos_policy": tokenization["warmup_kind"] + "; ONE_UNSCORED_PREFIX; NO_EOS_APPEND",
        "corpus_sha256": tokenization["source_sha256"],
        "evidence_tag": "MEASURED_CPU",
    }, cache_stats


def add_baseline_ratios(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result = dict(candidate)
    result["perplexity_ratio_vs_baseline"] = (
        candidate["model_internal_perplexity"] / baseline["model_internal_perplexity"]
    )
    result["bpb_delta_vs_baseline"] = candidate["bits_per_byte"] - baseline["bits_per_byte"]
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-corpus")
    prepare.add_argument("--source-parquet", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--maximum-bytes", type=int, default=24_576)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--content-tokens", type=int, required=True)
    plan.add_argument("--context", type=int, required=True)
    plan.add_argument("--stride", type=int)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.command == "prepare-corpus":
        value = prepare_parquet_corpus(args.source_parquet, args.output, maximum_bytes=args.maximum_bytes)
    else:
        value = {
            "windows": [asdict(window) for window in make_window_plan(args.content_tokens, args.context, stride=args.stride)]
        }
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
