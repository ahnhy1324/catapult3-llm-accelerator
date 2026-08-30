#!/usr/bin/env python3
"""Bounded CPU/geometry smoke for the supported Falcon3 1B ternary instruct model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common import (
    capture_prompts,
    configure_runtime,
    environment_manifest,
    peak_rss_bytes,
    perplexity,
    read_prompts,
    scan_model_health,
    sha256_logits,
    tokenize_evaluation_text,
)
from manifest_verify import inspect_safetensors_header, verify_artifact


MODEL_ID = "tiiuae/Falcon3-1B-Instruct-1.58bit"
MODEL_REVISION = "72fd3f95fcd82639c902304919629edda8c6f2b4"
TRANSFORMERS_REVISION = "096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4"


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=root / "artifact_manifest_v2.json")
    parser.add_argument("--prompts-file", type=Path, default=root / "prompts.txt")
    parser.add_argument("--ppl-text", type=Path, default=root / "ppl_smoke.txt")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    started = time.perf_counter()
    configure_runtime(args.seed, args.threads)
    checkpoint = args.checkpoint_dir.resolve()
    verified = verify_artifact(
        args.manifest.resolve(),
        "falcon3-1b-instruct-1.58bit",
        checkpoint,
        required_roles={
            "weight", "config", "generation_config", "special_tokens_map",
            "tokenizer_json", "tokenizer_config",
        },
    )
    tensor_file_health = inspect_safetensors_header(checkpoint / "model.safetensors")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    # The Transformers BitNet integration decorates weight unpacking with
    # torch.compile.  On a CPU-only Windows host that otherwise requires an
    # external MSVC compiler just to execute the reference.  This bounded
    # health run intentionally uses the exact eager function instead.
    original_compile = torch.compile

    def eager_compile(function=None, *compile_args, **compile_kwargs):
        del compile_args, compile_kwargs
        if function is None:
            return lambda selected: selected
        return function

    torch.compile = eager_compile
    try:
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
    finally:
        torch.compile = original_compile
    model = model.eval()
    device = torch.device("cpu")
    prompts = read_prompts(args.prompts_file)
    prompt_rows, logits, _ = capture_prompts(
        model, tokenizer, prompts, device, max_new_tokens=args.max_new_tokens
    )
    ids = tokenize_evaluation_text(tokenizer, args.ppl_text, 256)
    ppl, _ = perplexity(model, ids, device, sequence_length=128)
    health = scan_model_health(
        model,
        loading_info.get("missing_keys", []),
        loading_info.get("unexpected_keys", []),
        mismatched=loading_info.get("mismatched_keys", []),
    )
    health_valid = health["finite"] and not any(
        health[key]
        for key in (
            "unapproved_all_zero_tensors", "missing_tensors", "unexpected_tensors",
            "shape_mismatches", "duplicate_parameter_names", "abnormal_scales",
        )
    )
    geometry = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "tied_word_embeddings": config.tie_word_embeddings,
        "body_major_linear_weight_elements": 1_132_462_080,
        "lm_head_weight_elements": 268_435_456,
        "e2e_linear_weight_elements_per_token": 1_400_897_536,
        "ideal_lanes_for_100_tps_at_225MHz": 622.6211271111112,
        "evidence_tag": "CALCULATED_FROM_CONFIG",
    }
    result = {
        "schema_version": "catapult3-model-selection-result-v2",
        "run_id": "falcon3-1b-instruct-1.58bit-limited-smoke",
        "run_mode": "smoke",
        "status": "PASS" if health_valid else "FAIL",
        "evidence_tags": ["MEASURED_CPU", "MEASURED_MODEL_FILE", "CALCULATED_FROM_CONFIG", "BLOCKED"],
        "environment": environment_manifest(args.seed, args.threads),
        "model": {
            "model_id": MODEL_ID,
            "architecture": "LlamaForCausalLM with BitNet linear modules",
            "parameter_class": "1B-class native 1.58-bit instruct",
            "geometry": geometry,
            "license": "TII Falcon License 2.0",
        },
        "checkpoint": {
            "revision": MODEL_REVISION,
            "manifest": str(args.manifest),
            "files": verified,
            "safetensors_header_health": tensor_file_health,
        },
        "backend": {
            "name": "Transformers/PyTorch CPU",
            "revision": TRANSFORMERS_REVISION,
            "execution_path": "AutoModelForCausalLM native BitNet modules with eager CPU weight unpack",
        },
        "health": health,
        "baseline": {
            "logits_sha256": sha256_logits(logits),
            "prompts": prompt_rows,
            "perplexity": ppl,
        },
        "variants": [],
        "performance": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
            "prompt_count": len(prompts),
            "max_new_tokens_per_prompt": args.max_new_tokens,
        },
        "artifacts": [{"path": str(args.output), "kind": "compact_result_json"}],
        "blockers": [{
            "code": "LIMITED_CANDIDATE_NO_FIXED_BYTE_MATRIX",
            "impact": "The official untied FP16 LM head makes the fully-on-card memory roof a no-go, so the bounded investigation stops after generation health and smoke PPL.",
            "evidence_tag": "BLOCKED",
        }],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "logits_sha256": result["baseline"]["logits_sha256"],
        "wall_seconds": result["performance"]["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
