#!/usr/bin/env python3
"""Run the reproducible BitNet 0.7B CPU model-selection experiment."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
NVFP4_ROOT = HERE.parent / "nvfp4_bitnet"
if str(NVFP4_ROOT) not in sys.path:
    sys.path.insert(0, str(NVFP4_ROOT))

from nvfp4_fakequant import fake_quantize_nvfp4  # noqa: E402

from common import (  # noqa: E402
    LinearBoundaryQuantizer,
    Timer,
    aggregate_cache_stats,
    capture_prompts,
    compare_prompt_sets,
    configure_runtime,
    environment_manifest,
    peak_rss_bytes,
    perplexity,
    read_prompts,
    scan_model_health,
    sha256_file,
    sha256_logits,
    tokenize_evaluation_text,
)
from fixed_byte_eval import add_baseline_ratios, score_fixed_bytes  # noqa: E402
from manifest_verify import inspect_safetensors_header, verify_artifact  # noqa: E402
from quantizers import QuantizedDynamicCache  # noqa: E402


MODEL_ID = "1bitLLM/bitnet_b1_58-large"
MODEL_REVISION = "85d047191dcb224f0e04f20d26110caaf8dc1a47"
MODEL_FILE = "model.safetensors"
MODEL_BYTES = 2_915_408_840
MODEL_SHA256 = "100062646f1f85771ebe297c5e476642d171c2e0e916b2ed8d19dfbe201b4b52"


def _load_checkpoint_classes(checkpoint_dir: Path):
    """Load the repository's frozen custom code as an isolated local package.

    The 2024 checkpoint predates ``auto_map`` metadata, so current
    Transformers cannot discover ``BitnetTokenizer`` from a local snapshot.
    Loading the exact revision files explicitly preserves their implementation
    without editing the checkpoint or falling back to a different tokenizer.
    """
    package_name = "catapult_bitnet_0_7b_frozen"
    package = types.ModuleType(package_name)
    package.__path__ = [str(checkpoint_dir)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    loaded = {}
    for module_name in ("configuration_bitnet", "utils_quant", "modeling_bitnet", "tokenization_bitnet"):
        full_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(full_name, checkpoint_dir / f"{module_name}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load frozen module: {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module
    frozen_modeling = loaded["modeling_bitnet"]
    original_dynamic_cache = frozen_modeling.DynamicCache

    class AdapterCompatibleDynamicCache(original_dynamic_cache):
        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            if isinstance(past_key_values, QuantizedDynamicCache):
                return past_key_values
            return original_dynamic_cache.from_legacy_cache(past_key_values)

    frozen_modeling.DynamicCache = AdapterCompatibleDynamicCache
    return (
        loaded["configuration_bitnet"].BitnetConfig,
        loaded["modeling_bitnet"].BitnetForCausalLM,
        loaded["tokenization_bitnet"].BitnetTokenizer,
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, default=HERE / "prompts.txt")
    parser.add_argument("--ppl-text", type=Path, default=HERE / "ppl_smoke.txt")
    parser.add_argument("--run-mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--predicted-tokens", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--manifest", type=Path, default=HERE / "artifact_manifest_v2.json")
    parser.add_argument("--fixed-byte-corpus", type=Path)
    parser.add_argument("--fixed-byte-contexts", type=int, nargs="+", default=(512, 2048))
    parser.add_argument(
        "--fixed-byte-variants",
        default="row16_head_plus_KV4,block16x16_head_plus_KV4",
        help="comma-separated variant names, or 'all'",
    )
    return parser.parse_args()


def _copy_weight(destination: torch.Tensor, source: torch.Tensor, chunk_rows: int = 256) -> None:
    if destination.shape != source.shape:
        raise ValueError(f"weight shape mismatch: {destination.shape} != {source.shape}")
    with torch.inference_mode():
        for start in range(0, destination.shape[0], chunk_rows):
            end = min(start + chunk_rows, destination.shape[0])
            destination[start:end].copy_(source[start:end].to(destination.dtype))


def _install_tied_weight(model: Any, source: torch.Tensor) -> dict[str, Any]:
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    if input_module is None or not hasattr(input_module, "weight"):
        raise RuntimeError("input embedding is missing")
    physically_tied = bool(
        output_module is not None
        and hasattr(output_module, "weight")
        and output_module.weight.data_ptr() == input_module.weight.data_ptr()
    )
    _copy_weight(input_module.weight, source)
    if output_module is not None and hasattr(output_module, "weight") and not physically_tied:
        _copy_weight(output_module.weight, source)
    return {"physically_tied": physically_tied, "shape": list(source.shape)}


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
    kv_bits: int | None,
    fixed_corpus: bytes | None = None,
    fixed_contexts: tuple[int, ...] = (),
    run_fixed: bool = False,
) -> dict[str, Any]:
    cache_factory = (lambda: QuantizedDynamicCache(kv_bits)) if kv_bits else None
    with Timer() as timer:
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
    per_prompt, aggregate = compare_prompt_sets(baseline_rows, baseline_logits, rows, logits)
    aggregate["perplexity"] = ppl["perplexity"]
    aggregate["mean_nll"] = ppl["mean_nll"]
    aggregate["perplexity_ratio"] = ppl["perplexity"] / baseline_ppl["perplexity"]
    kv_stats = aggregate_cache_stats(prompt_cache + ppl_cache)
    if kv_bits is not None and (kv_stats is None or kv_stats["update_count"] == 0):
        raise RuntimeError(f"KV{kv_bits} cache adapter did not observe any updates")
    return {
        "name": name,
        "components": components,
        "status": "PASS",
        "logits_sha256": sha256_logits(logits),
        "per_prompt": per_prompt,
        "aggregate": aggregate,
        "quantizer": {
            "kv": kv_stats,
            "body_activation_contract": "UNCHANGED_NATIVE_BITNET_INPUT_INT8",
        },
        "fixed_byte": fixed_byte,
        "wall_seconds": timer.elapsed,
    }


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
    model_path = checkpoint_dir / MODEL_FILE
    required_roles = {
        "weight", "config", "remote_configuration", "remote_modeling", "remote_quantization",
        "remote_tokenizer", "tokenizer_json", "tokenizer_model", "tokenizer_config",
        "special_tokens_map", "added_tokens", "generation_config",
    }
    verified_files = verify_artifact(
        args.manifest.resolve(), "bitnet-0.7b", checkpoint_dir, required_roles=required_roles
    )
    verified_corpus = []
    if args.fixed_byte_corpus:
        verified_corpus = verify_artifact(
            args.manifest.resolve(),
            "fixed-byte-corpus-selected",
            args.fixed_byte_corpus.resolve().parent,
            required_roles={"corpus_selected"},
        )
    tensor_file_health = inspect_safetensors_header(model_path)
    weight_record = next(record for record in verified_files if record["role"] == "weight")
    actual_size = weight_record["byte_size"]
    actual_hash = weight_record["sha256"]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    BitnetConfig, BitnetForCausalLM, BitnetTokenizer = _load_checkpoint_classes(checkpoint_dir)
    tokenizer = BitnetTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    config = BitnetConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    loaded = BitnetForCausalLM.from_pretrained(
        checkpoint_dir,
        config=config,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    model, loading_info = loaded
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
        "fixed_byte": baseline_fixed,
    }

    variants: list[dict[str, Any]] = []
    for bits in (8, 4, 3):
        variants.append(
            _evaluate_variant(
                name=f"KV{bits}_only",
                components=[f"KV{bits}"],
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
                kv_bits=bits,
                fixed_corpus=fixed_corpus,
                fixed_contexts=fixed_contexts,
                run_fixed=fixed_all or f"KV{bits}_only" in selected_fixed,
            )
        )

    embedding = model.get_input_embeddings()
    if embedding is None:
        raise RuntimeError("input embedding is missing")
    original = embedding.weight.detach().cpu().clone()
    quantized, row16_stats = fake_quantize_nvfp4(
        original,
        layout="row16",
        chunk_rows=256,
        work_device="cpu",
        output_device="cpu",
        output_dtype=original.dtype,
    )
    tie_info = _install_tied_weight(model, quantized)
    del quantized
    gc.collect()
    for name, bits in (("row16_head_only", None), ("row16_head_plus_KV4", 4), ("row16_head_plus_KV3", 3)):
        components = ["ROW16_HEAD"] + ([] if bits is None else [f"KV{bits}"])
        variant = _evaluate_variant(
            name=name,
            components=components,
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
            kv_bits=bits,
            fixed_corpus=fixed_corpus,
            fixed_contexts=fixed_contexts,
            run_fixed=fixed_all or name in selected_fixed,
        )
        variant["quantizer"]["row16_head"] = row16_stats.to_dict()
        variant["quantizer"]["tie_info"] = tie_info
        variants.append(variant)
    _install_tied_weight(model, original)
    block_quantized, block16_stats = fake_quantize_nvfp4(
        original,
        layout="block16x16",
        chunk_rows=256,
        work_device="cpu",
        output_device="cpu",
        output_dtype=original.dtype,
    )
    block_tie_info = _install_tied_weight(model, block_quantized)
    del block_quantized
    gc.collect()
    block_variant = _evaluate_variant(
        name="block16x16_head_only",
        components=["BLOCK16X16_HEAD"],
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
        kv_bits=None,
        fixed_corpus=fixed_corpus,
        fixed_contexts=fixed_contexts,
        run_fixed=fixed_all or "block16x16_head_only" in selected_fixed,
    )
    block_variant["quantizer"]["block16x16_head"] = block16_stats.to_dict()
    block_variant["quantizer"]["tie_info"] = block_tie_info
    variants.append(block_variant)
    block_kv4_variant = _evaluate_variant(
        name="block16x16_head_plus_KV4",
        components=["BLOCK16X16_HEAD", "KV4"],
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
        kv_bits=4,
        fixed_corpus=fixed_corpus,
        fixed_contexts=fixed_contexts,
        run_fixed=fixed_all or "block16x16_head_plus_KV4" in selected_fixed,
    )
    block_kv4_variant["quantizer"]["block16x16_head"] = block16_stats.to_dict()
    block_kv4_variant["quantizer"]["tie_info"] = block_tie_info
    variants.append(block_kv4_variant)
    _install_tied_weight(model, original)
    del original
    gc.collect()

    blockers: list[dict[str, Any]] = []
    if (
        not health["finite"]
        or health["missing_tensors"]
        or health["unexpected_tensors"]
        or health["shape_mismatches"]
        or health["unapproved_all_zero_tensors"]
    ):
        blockers.append({"code": "MODEL_HEALTH_FAILURE", "detail": health})
    status = "PASS" if not blockers else "PARTIAL"
    result = {
        "schema_version": "catapult3-model-selection-result-v2",
        "run_id": f"bitnet-0.7b-{args.run_mode}-seed{args.seed}",
        "run_mode": args.run_mode,
        "status": status,
        "evidence_scope": ["CPU_MEASURED", "MODEL_FILE_CALCULATED"],
        "evidence_tags": ["MEASURED_CPU", "MEASURED_MODEL_FILE", "CALCULATED_FROM_CONFIG"],
        "environment": environment_manifest(args.seed, args.threads),
        "model": {
            "candidate": "A",
            "model_id": MODEL_ID,
            "architecture": "BitnetForCausalLM",
            "parameter_class": "0.7B",
            "provenance": "PUBLIC_1bitLLM_REPRODUCTION_NOT_MICROSOFT_OFFICIAL",
        },
        "checkpoint": {
            "revision": MODEL_REVISION,
            "license": "MIT",
            "manifest": str(args.manifest),
            "files": verified_files,
        },
        "backend": {
            "name": "Transformers trust_remote_code BitNet",
            "revision": MODEL_REVISION,
            "execution_path": "CPU_BFLOAT16_REFERENCE_WITH_NATIVE_BITNET_BODY_CONTRACT",
        },
        "health": health,
        "baseline": baseline,
        "variants": variants,
        "performance": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
            "predicted_tokens_per_variant": predicted_tokens,
            "prompt_count": len(prompts),
            "max_new_tokens_per_prompt": max_new_tokens,
        },
        "artifacts": [
            {"kind": "checkpoint", "path": MODEL_FILE, "sha256": actual_hash, "byte_size": actual_size},
            *({"kind": "fixed_byte_corpus", **row} for row in verified_corpus),
        ],
        "blockers": blockers,
    }
    for variant in result["variants"]:
        for context, metrics in variant.get("fixed_byte", {}).items():
            variant["fixed_byte"][context] = add_baseline_ratios(metrics, baseline_fixed[context])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(args.output), "variants": len(variants)}, indent=2))


if __name__ == "__main__":
    main()
