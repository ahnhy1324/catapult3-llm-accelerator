#!/usr/bin/env python3
"""Run the unpacked-reference Bonsai 1.7B CPU model-selection experiment.

The official Q1_0 GGUF remains the deployment baseline.  This adapter uses
PrismML's official unpacked checkpoint to expose binary-linear activation
boundaries and compare activation, accumulator, and scale candidates without
rewriting the original model files.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from common import (
    LinearBoundaryQuantizer,
    Timer,
    aggregate_cache_stats,
    capture_prompts,
    compare_prompt_sets,
    configure_runtime,
    encode_prompt,
    environment_manifest,
    peak_rss_bytes,
    perplexity,
    read_prompts,
    scan_model_health,
    sha256_file,
    sha256_logits,
    tokenize_evaluation_text,
)
from fixed_byte_eval import add_baseline_ratios, score_fixed_bytes
from manifest_verify import inspect_safetensors_header, verify_artifact
from quantizers import QuantizedDynamicCache, binary_group_linear_reference


MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
MODEL_REVISION = "a7f720bd688d7563714f3118edd97b83d06f0615"
MODEL_FILE = "model.safetensors"
MODEL_BYTES = 3_440_091_640
MODEL_SHA256 = "cf9a24cbd02e6e257bcfd3177475aaca7f8bd1a63a745441f30d3e40f4313a6b"
TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_BYTES = 11_422_654
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
GGUF_ID = "prism-ml/Bonsai-1.7B-gguf"
GGUF_REVISION = "210a9e99f79cb184909d49595906526eb2b3dd9a"
GGUF_BYTES = 248_302_272
GGUF_SHA256 = "3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, default=Path(__file__).resolve().parent / "prompts.txt")
    parser.add_argument("--ppl-text", type=Path, default=Path(__file__).resolve().parent / "ppl_smoke.txt")
    parser.add_argument("--run-mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--predicted-tokens", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    root = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=root / "artifact_manifest_v2.json")
    parser.add_argument("--fixed-byte-corpus", type=Path)
    parser.add_argument("--fixed-byte-contexts", type=int, nargs="+", default=(512, 2048))
    parser.add_argument(
        "--fixed-byte-variants",
        default="A8_g128_plus_KV8,A10_g128_plus_KV8,A8_g128_plus_K4_V6,A8_g128_plus_K4_V5",
        help="comma-separated variant names, or 'all'",
    )
    return parser.parse_args()


def _evaluate_variant(
    *,
    name: str,
    components: list[str],
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    baseline_rows: list[dict[str, Any]],
    baseline_logits: list[torch.Tensor],
    baseline_ppl: dict[str, Any],
    ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    sequence_length: int,
    activation_bits: int | None = None,
    activation_group_size: int | None = None,
    kv_k_bits: int | None = None,
    kv_v_bits: int | None = None,
    kv_clip_percentile: float | None = None,
    fixed_corpus: bytes | None = None,
    fixed_contexts: tuple[int, ...] = (),
    run_fixed: bool = False,
) -> dict[str, Any]:
    cache_factory = (
        lambda: QuantizedDynamicCache(
            kv_k_bits,
            v_bits=kv_v_bits,
            clip_percentile=kv_clip_percentile,
            v_clip_percentile=kv_clip_percentile,
        )
    ) if kv_k_bits else None
    quantizer = (
        LinearBoundaryQuantizer(model, bits=activation_bits, group_size=activation_group_size)
        if activation_bits
        else None
    )
    with Timer() as timer:
        if quantizer:
            quantizer.__enter__()
        try:
            rows, logits, prompt_cache = capture_prompts(
                model,
                tokenizer,
                prompts,
                device,
                max_new_tokens=max_new_tokens,
                cache_factory=cache_factory,
            )
            ppl, ppl_cache = perplexity(
                model,
                ids,
                device,
                sequence_length=sequence_length,
                cache_factory=cache_factory,
            )
            fixed_byte: dict[str, Any] = {}
            if fixed_corpus is not None and run_fixed:
                for context in fixed_contexts:
                    metrics, fixed_cache = score_fixed_bytes(
                        model,
                        tokenizer,
                        fixed_corpus,
                        device,
                        context=context,
                        cache_factory=cache_factory,
                    )
                    fixed_byte[str(context)] = metrics
                    ppl_cache.extend(fixed_cache)
        finally:
            if quantizer:
                quantizer.__exit__(None, None, None)
    per_prompt, aggregate = compare_prompt_sets(baseline_rows, baseline_logits, rows, logits)
    aggregate["perplexity"] = ppl["perplexity"]
    aggregate["mean_nll"] = ppl["mean_nll"]
    aggregate["perplexity_ratio"] = ppl["perplexity"] / baseline_ppl["perplexity"]
    kv_stats = aggregate_cache_stats(prompt_cache + ppl_cache)
    if kv_k_bits is not None and (kv_stats is None or kv_stats["update_count"] == 0):
        raise RuntimeError(f"K{kv_k_bits}/V{kv_v_bits or kv_k_bits} cache adapter did not observe any updates")
    return {
        "name": name,
        "components": components,
        "status": "PASS",
        "logits_sha256": sha256_logits(logits),
        "per_prompt": per_prompt,
        "aggregate": aggregate,
        "quantizer": {
            "activation": quantizer.stats_dict() if quantizer else None,
            "kv": kv_stats,
        },
        "fixed_byte": fixed_byte,
        "wall_seconds": timer.elapsed,
    }


def _capture_accumulator_inputs(model: Any, tokenizer: Any, prompt: str) -> dict[str, torch.Tensor]:
    layers = len(model.model.layers)
    selected_layers = sorted({0, layers // 2, layers - 1})
    selected_names = []
    for layer in selected_layers:
        selected_names.extend(
            [
                f"model.layers.{layer}.self_attn.q_proj",
                f"model.layers.{layer}.mlp.gate_proj",
            ]
        )
    modules = dict(model.named_modules())
    captures: dict[str, torch.Tensor] = {}
    handles = []
    for name in selected_names:
        module = modules.get(name)
        if module is None:
            raise RuntimeError(f"required accumulator sample module is missing: {name}")

        def hook(_module: Any, inputs: tuple[Any, ...], *, sample_name: str = name) -> None:
            if sample_name not in captures:
                value = inputs[0].detach().reshape(-1, inputs[0].shape[-1])[:4].cpu().to(torch.bfloat16)
                captures[sample_name] = value

        handles.append(module.register_forward_pre_hook(hook))
    try:
        encoded = encode_prompt(tokenizer, prompt, torch.device("cpu"))
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != set(selected_names):
        raise RuntimeError(f"incomplete accumulator capture: {sorted(captures)}")
    return captures


def _numeric_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    out = candidate.float()
    delta = out - ref
    denominator = float(ref.square().sum().item())
    return {
        "relative_rmse": math.sqrt(float(delta.square().sum().item()) / max(denominator, 1e-30)),
        "rmse": float(delta.square().mean().sqrt().item()),
        "max_abs_error": float(delta.abs().amax().item()),
        "cosine": float(F.cosine_similarity(ref.reshape(-1), out.reshape(-1), dim=0).item()),
    }


def _accumulator_matrix(model: Any, captures: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    results: list[dict[str, Any]] = []
    activation_modes = [
        (12, 128, "A12_G128"),
        (10, 128, "A10_G128"),
        (8, None, "A8_PER_TOKEN"),
        (8, 128, "A8_G128"),
    ]
    for name, activation in captures.items():
        module = modules[name]
        weight = module.weight.detach().cpu()[:64].to(torch.bfloat16)
        reference = F.linear(activation.float(), weight.float()).to(torch.bfloat16)
        projection_kind = "attention" if "self_attn" in name else "mlp"
        for activation_bits, activation_group, activation_name in activation_modes:
            for accumulator_bits in (32, 24, 20):
                for scale_mode, fixed_bits, group_round_bits in (
                    ("FP_REFERENCE_SCALE", None, None),
                    ("FIXED_UQ4_20_BEFORE_MULTIPLY", 20, None),
                    ("FP_SCALE_THEN_Q12_GROUP_OUTPUT", None, 12),
                ):
                    output, stats = binary_group_linear_reference(
                        activation,
                        weight,
                        activation_bits=activation_bits,
                        activation_group_size=activation_group,
                        weight_group_size=128,
                        accumulator_bits=accumulator_bits,
                        fixed_scale_fraction_bits=fixed_bits,
                        group_output_fraction_bits=group_round_bits,
                    )
                    results.append(
                        {
                            "module": name,
                            "projection_kind": projection_kind,
                            "activation": activation_name,
                            "accumulator": f"INT{accumulator_bits}",
                            "scale_mode": scale_mode,
                            "sample_tokens": int(activation.shape[0]),
                            "sample_output_rows": int(weight.shape[0]),
                            "metrics_vs_unpacked_bf16_linear": _numeric_metrics(reference, output),
                            "reference_stats": stats.to_dict(),
                            "evidence_scope": "CPU_MEASURED_REPRESENTATIVE_BOUNDARY",
                        }
                    )
    return results


def main() -> None:
    args = arguments()
    predicted_tokens = args.predicted_tokens or (256 if args.run_mode == "smoke" else 4096)
    sequence_length = args.sequence_length or (128 if args.run_mode == "smoke" else 512)
    max_new_tokens = args.max_new_tokens or (16 if args.run_mode == "smoke" else 32)
    if predicted_tokens < 256:
        raise SystemExit("predicted token count must be at least 256")
    configure_runtime(args.seed, args.threads)
    started = time.perf_counter()
    checkpoint_dir = args.checkpoint_dir.resolve()
    required_roles = {
        "weight", "config", "tokenizer_json", "tokenizer_config", "special_tokens_map",
        "added_tokens", "chat_template", "merges", "vocab", "generation_config",
    }
    verified_files = verify_artifact(
        args.manifest.resolve(), "bonsai-1.7b-unpacked", checkpoint_dir, required_roles=required_roles
    )
    verified_corpus = []
    if args.fixed_byte_corpus:
        verified_corpus = verify_artifact(
            args.manifest.resolve(),
            "fixed-byte-corpus-selected",
            args.fixed_byte_corpus.resolve().parent,
            required_roles={"corpus_selected"},
        )
    tensor_file_health = inspect_safetensors_header(checkpoint_dir / MODEL_FILE)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    model = model.eval()
    device = torch.device("cpu")
    prompts = read_prompts(args.prompts_file)
    ids = tokenize_evaluation_text(tokenizer, args.ppl_text, predicted_tokens)
    fixed_corpus = args.fixed_byte_corpus.read_bytes() if args.fixed_byte_corpus else None
    fixed_contexts = tuple(dict.fromkeys(args.fixed_byte_contexts))
    selected_fixed = {name for name in args.fixed_byte_variants.split(",") if name}
    fixed_all = args.fixed_byte_variants == "all"
    health = scan_model_health(
        model,
        loading_info.get("missing_keys", []),
        loading_info.get("unexpected_keys", []),
        mismatched=loading_info.get("mismatched_keys", []),
    )
    health["tensor_file"] = tensor_file_health

    with LinearBoundaryQuantizer(model, bits=None, group_size=None) as profiler:
        baseline_rows, baseline_logits, _ = capture_prompts(
            model,
            tokenizer,
            prompts,
            device,
            max_new_tokens=max_new_tokens,
        )
    baseline_ppl, _ = perplexity(model, ids, device, sequence_length=sequence_length)
    baseline_fixed: dict[str, Any] = {}
    if fixed_corpus is not None:
        for context in fixed_contexts:
            baseline_fixed[str(context)], _ = score_fixed_bytes(
                model, tokenizer, fixed_corpus, device, context=context
            )
    baseline = {
        "logits_sha256": sha256_logits(baseline_logits),
        "prompts": baseline_rows,
        "perplexity": baseline_ppl,
        "activation_profile": profiler.stats_dict(),
        "baseline_scope": "UNPACKED_FP16_SEMANTIC_REFERENCE_NOT_NATIVE_Q1_KERNEL",
        "fixed_byte": baseline_fixed,
    }

    variants: list[dict[str, Any]] = []
    for bits, group_size, name in (
        (12, 128, "A12_per_128_group"),
        (10, 128, "A10_per_128_group"),
        (8, None, "A8_per_token"),
        (8, 128, "A8_per_128_group"),
    ):
        variants.append(
            _evaluate_variant(
                name=name,
                components=[name.upper()],
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                baseline_rows=baseline_rows,
                baseline_logits=baseline_logits,
                baseline_ppl=baseline_ppl,
                ids=ids,
                device=device,
                max_new_tokens=max_new_tokens,
                sequence_length=sequence_length,
                activation_bits=bits,
                activation_group_size=group_size,
                fixed_corpus=fixed_corpus,
                fixed_contexts=fixed_contexts,
                run_fixed=fixed_all or name in selected_fixed,
            )
        )
    for name, k_bits, v_bits, clip in (
        ("KV8_only", 8, 8, None),
        ("K4_V8_only", 4, 8, None),
        ("K4_V6_only", 4, 6, None),
        ("K4_V5_only", 4, 5, None),
        ("KV4_only", 4, 4, None),
        ("KV3_only", 3, 3, None),
        ("KV4_percentile_0p999", 4, 4, 0.999),
    ):
        variants.append(
            _evaluate_variant(
                name=name,
                components=[f"K{k_bits}", f"V{v_bits}"],
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                baseline_rows=baseline_rows,
                baseline_logits=baseline_logits,
                baseline_ppl=baseline_ppl,
                ids=ids,
                device=device,
                max_new_tokens=max_new_tokens,
                sequence_length=sequence_length,
                kv_k_bits=k_bits,
                kv_v_bits=v_bits,
                kv_clip_percentile=clip,
                fixed_corpus=fixed_corpus,
                fixed_contexts=fixed_contexts,
                run_fixed=fixed_all or name in selected_fixed,
            )
        )

    for name, activation_bits, k_bits, v_bits in (
        ("A8_g128_plus_KV8", 8, 8, 8),
        ("A10_g128_plus_KV8", 10, 8, 8),
        ("A8_g128_plus_K4_V6", 8, 4, 6),
        ("A8_g128_plus_K4_V5", 8, 4, 5),
    ):
        variants.append(
            _evaluate_variant(
                name=name,
                components=[f"A{activation_bits}_G128", f"K{k_bits}", f"V{v_bits}"],
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                baseline_rows=baseline_rows,
                baseline_logits=baseline_logits,
                baseline_ppl=baseline_ppl,
                ids=ids,
                device=device,
                max_new_tokens=max_new_tokens,
                sequence_length=sequence_length,
                activation_bits=activation_bits,
                activation_group_size=128,
                kv_k_bits=k_bits,
                kv_v_bits=v_bits,
                fixed_corpus=fixed_corpus,
                fixed_contexts=fixed_contexts,
                run_fixed=fixed_all or name in selected_fixed,
            )
        )

    captures = _capture_accumulator_inputs(model, tokenizer, prompts[0])
    accumulator_matrix = _accumulator_matrix(model, captures)
    blockers: list[dict[str, Any]] = [
        {
            "code": "NATIVE_GGUF_BASELINE_SEPARATE",
            "detail": "This result uses the official unpacked semantic reference. The official PrismML Q1_0 runtime is recorded as a separate native result and must be considered jointly.",
        }
    ]
    if (
        not health["finite"]
        or health["missing_tensors"]
        or health["unexpected_tensors"]
        or health["shape_mismatches"]
        or health["unapproved_all_zero_tensors"]
    ):
        blockers.append({"code": "MODEL_HEALTH_FAILURE", "detail": health})
    result = {
        "schema_version": "catapult3-model-selection-result-v2",
        "run_id": f"bonsai-1.7b-reference-{args.run_mode}-seed{args.seed}",
        "run_mode": args.run_mode,
        "status": "PARTIAL" if blockers else "PASS",
        "evidence_scope": ["CPU_MEASURED", "MODEL_FILE_CALCULATED"],
        "evidence_tags": ["MEASURED_CPU", "MEASURED_MODEL_FILE", "CALCULATED_FROM_CONFIG"],
        "environment": environment_manifest(args.seed, args.threads),
        "model": {
            "candidate": "B",
            "model_id": MODEL_ID,
            "architecture": "Qwen3ForCausalLM",
            "parameter_class": "1.7B",
            "deployment_model_id": GGUF_ID,
        },
        "checkpoint": {
            "revision": MODEL_REVISION,
            "license": "Apache-2.0",
            "manifest": str(args.manifest),
            "files": verified_files
            + [
                {
                    "name": "Bonsai-1.7B-Q1_0.gguf",
                    "byte_size": GGUF_BYTES,
                    "sha256": GGUF_SHA256,
                    "verification": "PINNED_MANIFEST_EXPECTATION; verified separately by native runner",
                    "revision": GGUF_REVISION,
                }
            ],
        },
        "backend": {
            "name": "Transformers unpacked Bonsai semantic reference",
            "revision": MODEL_REVISION,
            "execution_path": "CPU_BFLOAT16_WITH_BINARY_LINEAR_BOUNDARY_HOOKS",
            "official_native_runtime": "PrismML-Eng/llama.cpp",
        },
        "health": health,
        "baseline": baseline,
        "variants": variants,
        "accumulator_scale_matrix": accumulator_matrix,
        "performance": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
            "predicted_tokens_per_variant": predicted_tokens,
            "prompt_count": len(prompts),
            "max_new_tokens_per_prompt": max_new_tokens,
        },
        "artifacts": [
            {"kind": "checkpoint", "path": row["name"], **row}
            for row in verified_files
        ] + [{"kind": "fixed_byte_corpus", **row} for row in verified_corpus],
        "blockers": blockers,
    }
    for variant in result["variants"]:
        for context, metrics in variant.get("fixed_byte", {}).items():
            variant["fixed_byte"][context] = add_baseline_ratios(metrics, baseline_fixed[context])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "variants": len(variants),
                "accumulator_rows": len(accumulator_matrix),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
