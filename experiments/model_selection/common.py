"""Shared CPU evaluation helpers for model-selection adapters."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from quantizers import QuantizedDynamicCache, symmetric_fake_quantize


PROMPTS = [
    "Explain in three concise sentences why the sky appears blue.",
    "Write a correct Python function that returns whether an integer is prime.",
    "Alice has 12 apples and gives 5 away. Explain how many remain.",
    "대한민국의 수도는 어디인지 한 문장으로 답해 주세요.",
    "Continue with a short technical paragraph: An old FPGA card can still be useful when",
]


def configure_runtime(seed: int, threads: int) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_logits(logits: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in logits:
        array = tensor.detach().float().contiguous().numpy().astype("<f4", copy=False)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def environment_manifest(seed: int, threads: int) -> dict[str, Any]:
    import safetensors
    import transformers

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "numpy": np.__version__,
        "seed": seed,
        "torch_threads": threads,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "byte_order": sys.byteorder,
    }


def read_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(PROMPTS)
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) < 5:
        raise ValueError("at least five non-empty prompts are required")
    return prompts


def encode_prompt(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": "You are a concise and helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        encoded = tokenizer(prompt, return_tensors="pt")
    if isinstance(encoded, torch.Tensor):
        encoded = {"input_ids": encoded}
    return {key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)}


def _cache_kwargs(factory: Callable[[], QuantizedDynamicCache] | None) -> tuple[dict[str, Any], QuantizedDynamicCache | None]:
    if factory is None:
        return {}, None
    cache = factory()
    return {"past_key_values": cache, "use_cache": True}, cache


def greedy_generate_reference(
    model: Any,
    encoded: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    eos_token_id: int | list[int] | None,
    cache: Any | None = None,
) -> torch.Tensor:
    """Backend-neutral greedy decode without relying on ``GenerationMixin``.

    The 2024 BitNet remote model predates the Transformers 4.50 generation
    inheritance change.  Direct forward calls keep its frozen model code usable
    and exercise the same cache boundary as newer models.
    """
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    past_key_values = cache
    generated: list[torch.Tensor] = []
    eos_values = set()
    if isinstance(eos_token_id, int):
        eos_values.add(eos_token_id)
    elif eos_token_id is not None:
        eos_values.update(int(value) for value in eos_token_id)
    with torch.inference_mode():
        for step in range(max_new_tokens):
            current_ids = input_ids if step == 0 else generated[-1]
            kwargs: dict[str, Any] = {
                "input_ids": current_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            output = model(**kwargs)
            next_token = output.logits[:, -1:].argmax(dim=-1)
            generated.append(next_token)
            past_key_values = output.past_key_values
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=-1,
            )
            if int(next_token[0, 0].item()) in eos_values:
                break
    if not generated:
        return torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat(generated, dim=-1)


def capture_prompts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    *,
    max_new_tokens: int,
    cache_factory: Callable[[], QuantizedDynamicCache] | None = None,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    logits_list: list[torch.Tensor] = []
    cache_stats: list[dict[str, Any]] = []
    for prompt in prompts:
        encoded = encode_prompt(tokenizer, prompt, device)
        kwargs, cache = _cache_kwargs(cache_factory)
        with torch.inference_mode():
            output = model(**encoded, use_cache=cache is not None, **{k: v for k, v in kwargs.items() if k != "use_cache"})
        logits = output.logits[0, -1].detach().float().cpu()
        logits_list.append(logits)
        if cache is not None:
            cache_stats.append(cache.stats_dict())

        input_length = int(encoded["input_ids"].shape[-1])
        generation_cache = cache_factory() if cache_factory else None
        generated = greedy_generate_reference(
            model,
            encoded,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            cache=generation_cache,
        )[0].detach().cpu()
        if generation_cache is not None:
            cache_stats.append(generation_cache.stats_dict())
        ids = [int(value) for value in generated.tolist()]
        top = torch.topk(logits, min(10, logits.numel()))
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        rows.append(
            {
                "prompt": prompt,
                "input_tokens": input_length,
                "generated_token_ids": ids,
                "generated_text": generated_text,
                "health_flags": generation_health_flags(prompt, generated_text, ids),
                "top10_token_ids": [int(value) for value in top.indices.tolist()],
                "top10_logits": [float(value) for value in top.values.tolist()],
            }
        )
    return rows, logits_list, cache_stats


def generation_health_flags(prompt: str, generated_text: str, token_ids: list[int]) -> list[str]:
    flags: list[str] = []
    stripped = generated_text.strip()
    if not stripped:
        flags.append("EMPTY_OUTPUT")
    normalized_prompt = " ".join(prompt.lower().split())
    normalized_output = " ".join(stripped.lower().split())
    if normalized_prompt and (normalized_output.startswith(normalized_prompt) or normalized_prompt in normalized_output):
        flags.append("PROMPT_ECHO")
    else:
        prompt_words = normalized_prompt.split()
        output_words = normalized_output.split()
        common_prefix_words = 0
        for prompt_word, output_word in zip(prompt_words, output_words):
            if prompt_word != output_word:
                break
            common_prefix_words += 1
        if common_prefix_words >= min(6, len(prompt_words)):
            flags.append("PROMPT_ECHO")
    if len(token_ids) >= 8 and len(token_ids) % 2 == 0:
        midpoint = len(token_ids) // 2
        if token_ids[:midpoint] == token_ids[midpoint:]:
            flags.append("REPEATED_SEQUENCE_X2")
    if len(token_ids) >= 12:
        grams = [tuple(token_ids[index : index + 4]) for index in range(len(token_ids) - 3)]
        if grams and max(grams.count(gram) for gram in set(grams)) >= 3:
            flags.append("REPETITION_4GRAM_X3")
    if stripped and sum(character.isprintable() for character in stripped) / len(stripped) < 0.9:
        flags.append("NONPRINTABLE_OUTPUT")
    if "\ufffd" in stripped:
        flags.append("UNICODE_REPLACEMENT_CHARACTER")
    return flags


def compare_prompt_sets(
    baseline_rows: list[dict[str, Any]],
    baseline_logits: list[torch.Tensor],
    variant_rows: list[dict[str, Any]],
    variant_logits: list[torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    comparisons: list[dict[str, Any]] = []
    for base_row, base_logits, row, logits in zip(
        baseline_rows, baseline_logits, variant_rows, variant_logits, strict=True
    ):
        delta = logits.float() - base_logits.float()
        log_base = F.log_softmax(base_logits.float(), dim=-1)
        log_variant = F.log_softmax(logits.float(), dim=-1)
        base_ids = base_row["generated_token_ids"]
        ids = row["generated_token_ids"]
        common = 0
        for left, right in zip(base_ids, ids):
            if left != right:
                break
            common += 1
        denominator = max(len(base_ids), len(ids), 1)
        first_divergence = common if (common < len(base_ids) or common < len(ids)) else None
        positional = sum(int(left == right) for left, right in zip(base_ids, ids)) / denominator
        top1 = int(base_logits.argmax().item() == logits.argmax().item())
        base_top5 = set(torch.topk(base_logits, 5).indices.tolist())
        top5 = set(torch.topk(logits, 5).indices.tolist())
        base_top10 = set(torch.topk(base_logits, 10).indices.tolist())
        top10 = set(torch.topk(logits, 10).indices.tolist())
        comparisons.append(
            {
                "prompt": base_row["prompt"],
                "baseline_generated_text": base_row["generated_text"],
                "variant_generated_text": row["generated_text"],
                "baseline_generated_token_ids": base_ids,
                "variant_generated_token_ids": ids,
                "baseline_health_flags": base_row.get("health_flags", []),
                "variant_health_flags": row.get("health_flags", []),
                "last_token_logit_cosine": float(F.cosine_similarity(base_logits, logits, dim=0).item()),
                "last_token_logit_rmse": float(delta.square().mean().sqrt().item()),
                "last_token_logit_max_error": float(delta.abs().amax().item()),
                "kl_baseline_to_variant": float((log_base.exp() * (log_base - log_variant)).sum().item()),
                "top1_match": bool(top1),
                "top5_overlap": len(base_top5 & top5) / 5,
                "top10_overlap": len(base_top10 & top10) / 10,
                "greedy_common_prefix_tokens": common,
                "first_divergence_position": first_divergence,
                "positional_token_agreement": positional,
                "exact_generation_agreement": base_ids == ids,
            }
        )
    n = max(len(comparisons), 1)
    aggregate = {
        "prompt_count": len(comparisons),
        "mean_last_token_logit_cosine": sum(x["last_token_logit_cosine"] for x in comparisons) / n,
        "mean_last_token_logit_rmse": sum(x["last_token_logit_rmse"] for x in comparisons) / n,
        "max_last_token_logit_error": max((x["last_token_logit_max_error"] for x in comparisons), default=0.0),
        "mean_kl_baseline_to_variant": sum(x["kl_baseline_to_variant"] for x in comparisons) / n,
        "top1_match_rate": sum(x["top1_match"] for x in comparisons) / n,
        "mean_top5_overlap": sum(x["top5_overlap"] for x in comparisons) / n,
        "mean_top10_overlap": sum(x["top10_overlap"] for x in comparisons) / n,
        "mean_common_prefix_tokens": sum(x["greedy_common_prefix_tokens"] for x in comparisons) / n,
        "mean_positional_token_agreement": sum(x["positional_token_agreement"] for x in comparisons) / n,
        "exact_generation_agreement_rate": sum(x["exact_generation_agreement"] for x in comparisons) / n,
    }
    return comparisons, aggregate


def tokenize_evaluation_text(tokenizer: Any, path: Path, predicted_tokens: int) -> torch.Tensor:
    text = path.read_text(encoding="utf-8")
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if ids.numel() < predicted_tokens + 1:
        repeats = (predicted_tokens + 1 + ids.numel() - 1) // max(ids.numel(), 1)
        ids = ids.repeat(repeats)
    return ids[: predicted_tokens + 1].contiguous()


def perplexity(
    model: Any,
    ids: torch.Tensor,
    device: torch.device,
    *,
    sequence_length: int,
    cache_factory: Callable[[], QuantizedDynamicCache] | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    total_nll = 0.0
    total_targets = 0
    start = 0
    cache_stats: list[dict[str, Any]] = []
    with torch.inference_mode():
        while start < ids.numel() - 1:
            end = min(start + sequence_length, ids.numel())
            chunk = ids[start:end].unsqueeze(0).to(device)
            cache = cache_factory() if cache_factory else None
            kwargs: dict[str, Any] = {"input_ids": chunk, "labels": chunk, "use_cache": cache is not None}
            if cache is not None:
                kwargs["past_key_values"] = cache
            output = model(**kwargs)
            count = int(chunk.shape[-1] - 1)
            total_nll += float(output.loss.detach().float().item()) * count
            total_targets += count
            if cache is not None:
                cache_stats.append(cache.stats_dict())
            if end == ids.numel():
                break
            start = end - 1
    mean_nll = total_nll / max(total_targets, 1)
    return {
        "predicted_tokens": total_targets,
        "mean_nll": mean_nll,
        "perplexity": float(np.exp(min(mean_nll, 80.0))),
    }, cache_stats


def aggregate_cache_stats(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    values = sum(int(row["value_count"]) for row in rows)
    clipped = sum(int(row["clipped_count"]) for row in rows)
    weighted_endpoints = sum(float(row["endpoint_rate"]) * int(row["value_count"]) for row in rows)
    result = {
        "bits": rows[0]["bits"],
        "k_bits": rows[0].get("k_bits", rows[0]["bits"]),
        "v_bits": rows[0].get("v_bits", rows[0]["bits"]),
        "code_mapping": rows[0]["code_mapping"],
        "scale_granularity": rows[0]["scale_granularity"],
        "scale_method": rows[0].get("scale_method", "AMAX"),
        "clip_percentile": rows[0].get("clip_percentile"),
        "rounding": rows[0]["rounding"],
        "clipping": rows[0]["clipping"],
        "update_count": sum(int(row["update_count"]) for row in rows),
        "value_count": values,
        "clipped_count": clipped,
        "saturation_rate": clipped / max(values, 1),
        "endpoint_rate": weighted_endpoints / max(values, 1),
        "input_abs_max": max(float(row["input_abs_max"]) for row in rows),
    }
    for kind in ("k", "v"):
        if all(kind in row for row in rows):
            kind_values = sum(int(row[kind]["value_count"]) for row in rows)
            kind_clipped = sum(int(row[kind]["clipped_count"]) for row in rows)
            result[kind] = {
                "value_count": kind_values,
                "clipped_count": kind_clipped,
                "saturation_rate": kind_clipped / max(kind_values, 1),
                "endpoint_rate": sum(
                    float(row[kind]["endpoint_rate"]) * int(row[kind]["value_count"])
                    for row in rows
                )
                / max(kind_values, 1),
            }
    return result


def scan_model_health(
    model: Any,
    missing: list[str],
    unexpected: list[str],
    *,
    mismatched: list[Any] | None = None,
    all_zero_whitelist: set[str] | None = None,
) -> dict[str, Any]:
    all_zero: list[str] = []
    abnormal_scales: list[str] = []
    nonfinite: list[str] = []
    parameter_count = 0
    parameter_names: list[str] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_names.append(name)
        value = parameter.detach()
        parameter_count += value.numel()
        if not torch.isfinite(value).all():
            nonfinite.append(name)
        if value.numel() and torch.count_nonzero(value).item() == 0:
            all_zero.append(name)
        if "scale" in name.lower() and (not torch.isfinite(value).all() or (value <= 0).any()):
            abnormal_scales.append(name)
    whitelist = all_zero_whitelist or set()
    unapproved_all_zero = sorted(set(all_zero) - whitelist)
    duplicate_parameter_names = sorted({name for name in parameter_names if parameter_names.count(name) > 1})
    return {
        "finite": not nonfinite,
        "nonfinite_tensors": nonfinite,
        "all_zero_tensors": all_zero,
        "all_zero_whitelist": sorted(whitelist),
        "unapproved_all_zero_tensors": unapproved_all_zero,
        "missing_tensors": list(missing),
        "unexpected_tensors": list(unexpected),
        "shape_mismatches": list(mismatched or []),
        "duplicate_parameter_names": duplicate_parameter_names,
        "abnormal_scales": abnormal_scales,
        "parameter_count": parameter_count,
    }


@dataclass
class _BoundaryAggregate:
    calls: int = 0
    percentile_calls: int = 0
    values: int = 0
    clipped: int = 0
    endpoints: int = 0
    maximum: float = 0.0
    p50_sum: float = 0.0
    p90_sum: float = 0.0
    p99_sum: float = 0.0
    p999_sum: float = 0.0


class LinearBoundaryQuantizer:
    """Context manager that records or fake-quantizes FPGA binary-linear inputs."""

    PROJECTION_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def __init__(self, model: Any, *, bits: int | None, group_size: int | None) -> None:
        self.model = model
        self.bits = bits
        self.group_size = group_size
        self.handles: list[Any] = []
        self.by_kind = {"attention": _BoundaryAggregate(), "mlp": _BoundaryAggregate()}

    def _hook(self, name: str):
        kind = "attention" if "self_attn" in name else "mlp"

        def apply(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return None
            tensor = inputs[0]
            collect_percentiles = self.by_kind[kind].percentile_calls < 32
            if self.bits is None:
                _, _, _, stats = symmetric_fake_quantize(
                    tensor, 16, group_size=self.group_size, collect_percentiles=collect_percentiles
                )
                replacement = tensor
            else:
                replacement, _, _, stats = symmetric_fake_quantize(
                    tensor,
                    self.bits,
                    group_size=self.group_size,
                    collect_percentiles=collect_percentiles,
                )
            aggregate = self.by_kind[kind]
            aggregate.calls += 1
            aggregate.values += stats.value_count
            aggregate.clipped += stats.clipped_count
            aggregate.endpoints += stats.endpoint_count
            aggregate.maximum = max(aggregate.maximum, stats.input_abs_max)
            if collect_percentiles:
                aggregate.percentile_calls += 1
                aggregate.p50_sum += stats.input_abs_p50
                aggregate.p90_sum += stats.input_abs_p90
                aggregate.p99_sum += stats.input_abs_p99
                aggregate.p999_sum += stats.input_abs_p999
            if self.bits is None:
                return None
            return (replacement, *inputs[1:])

        return apply

    def __enter__(self) -> "LinearBoundaryQuantizer":
        for name, module in self.model.named_modules():
            if name.endswith(self.PROJECTION_SUFFIXES) and isinstance(module, torch.nn.Linear):
                self.handles.append(module.register_forward_pre_hook(self._hook(name)))
        if not self.handles:
            raise RuntimeError("no binary-linear projection boundaries were found")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def stats_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "bits": self.bits,
            "group_size": self.group_size,
            "granularity": "profile_only" if self.bits is None else (
                "per_token" if self.group_size is None else f"per_{self.group_size}_group"
            ),
            "rounding": "RNE_TIES_EVEN" if self.bits is not None else None,
            "code_mapping": "RESERVED_MIN_SYMMETRIC" if self.bits is not None else None,
            "by_projection_kind": {},
        }
        for kind, aggregate in self.by_kind.items():
            percentile_calls = max(aggregate.percentile_calls, 1)
            result["by_projection_kind"][kind] = {
                "calls": aggregate.calls,
                "percentile_sampled_calls": aggregate.percentile_calls,
                "value_count": aggregate.values,
                "saturation_rate": aggregate.clipped / max(aggregate.values, 1),
                "endpoint_rate": aggregate.endpoints / max(aggregate.values, 1),
                "abs_max": aggregate.maximum,
                "mean_sampled_call_abs_p50": aggregate.p50_sum / percentile_calls,
                "mean_sampled_call_abs_p90": aggregate.p90_sum / percentile_calls,
                "mean_sampled_call_abs_p99": aggregate.p99_sum / percentile_calls,
                "mean_sampled_call_abs_p999": aggregate.p999_sum / percentile_calls,
            }
        return result


class Timer:
    def __init__(self) -> None:
        self.started = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.elapsed = time.perf_counter() - self.started
