#!/usr/bin/env python3
"""Compare stock BitNet against an NVFP4-fake-quantized tied embedding/LM head.

This is a numerical quality demo, not a native NVFP4 speed benchmark. Quantized
weights are dequantized back to the model dtype before inference so the script
can run on CPUs, AMD GPUs, and pre-Blackwell NVIDIA GPUs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from nvfp4_fakequant import Layout, fake_quantize_nvfp4

DEFAULT_PROMPTS = [
    "Explain in three concise sentences why the sky appears blue.",
    "Write a correct Python function that returns whether an integer is prime.",
    "Alice has 12 apples and gives 5 away. Explain how many remain.",
    "대한민국의 수도는 어디인지 한 문장으로 답해 주세요.",
    "Continue this sentence with a short technical paragraph: An old FPGA card can still be useful when",
]


@dataclass
class PromptSnapshot:
    prompt: str
    input_tokens: int
    generated_token_ids: list[int]
    generated_text: str
    last_logits: torch.Tensor


@dataclass
class PromptComparison:
    prompt: str
    input_tokens: int
    baseline_text: str
    quantized_text: str
    baseline_generated_tokens: int
    quantized_generated_tokens: int
    common_prefix_tokens: int
    positional_token_agreement: float
    exact_generation_match: bool
    last_logit_cosine: float
    last_logit_rmse: float
    last_logit_max_abs_error: float
    kl_baseline_to_quantized: float
    top1_match: bool
    top5_overlap: float
    top10_overlap: float


class JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.dtype):
            return str(obj).removeprefix("torch.")
        return super().default(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure BitNet quality loss when its tied embedding/LM head is fake-quantized to NVFP4."
    )
    parser.add_argument(
        "--model-id",
        default="microsoft/bitnet-b1.58-2B-4T",
        help="Packed deployment checkpoint by default; use the -bf16 checkpoint for the master-weight path.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="MI50/ROCm users should normally choose float16.",
    )
    parser.add_argument(
        "--quant-work-device",
        default="cpu",
        help="Device used for fake quantization. CPU is safest for ROCm/pre-Blackwell systems.",
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("row16", "block16x16"),
        default=["row16", "block16x16"],
    )
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--ppl", action="store_true", help="Also evaluate perplexity on WikiText-2.")
    parser.add_argument("--ppl-tokens", type=int, default=4096)
    parser.add_argument("--ppl-seq-len", type=int, default=512)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("results/nvfp4_bitnet"))
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if device.type == "cpu":
        return torch.float32
    if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported"):
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def encode_chat_prompt(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
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


def model_forward_last_logits(model: Any, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        outputs = model(**encoded, use_cache=False)
    return outputs.logits[0, -1].detach().float().cpu()


def generate_snapshot(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    skip_generation: bool,
) -> PromptSnapshot:
    encoded = encode_chat_prompt(tokenizer, prompt, device)
    last_logits = model_forward_last_logits(model, encoded)
    input_len = int(encoded["input_ids"].shape[-1])

    if skip_generation:
        generated_ids: list[int] = []
        generated_text = ""
    else:
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generation_kwargs)
        generated = output_ids[0, input_len:].detach().cpu()
        generated_ids = [int(value) for value in generated.tolist()]
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)

    return PromptSnapshot(
        prompt=prompt,
        input_tokens=input_len,
        generated_token_ids=generated_ids,
        generated_text=generated_text,
        last_logits=last_logits,
    )


def capture_prompts(
    model: Any,
    tokenizer: Any,
    prompts: Iterable[str],
    device: torch.device,
    max_new_tokens: int,
    skip_generation: bool,
) -> list[PromptSnapshot]:
    snapshots = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"  prompt {index}: {prompt[:72]}")
        snapshots.append(
            generate_snapshot(model, tokenizer, prompt, device, max_new_tokens, skip_generation)
        )
    return snapshots


def topk_overlap(lhs: torch.Tensor, rhs: torch.Tensor, k: int) -> float:
    lhs_ids = set(torch.topk(lhs, k=k).indices.tolist())
    rhs_ids = set(torch.topk(rhs, k=k).indices.tolist())
    return len(lhs_ids & rhs_ids) / k


def compare_prompt(base: PromptSnapshot, quant: PromptSnapshot) -> PromptComparison:
    lhs = base.last_logits.float()
    rhs = quant.last_logits.float()
    diff = rhs - lhs
    log_p = F.log_softmax(lhs, dim=-1)
    log_q = F.log_softmax(rhs, dim=-1)
    p = log_p.exp()
    kl = float((p * (log_p - log_q)).sum().item())

    minimum = min(len(base.generated_token_ids), len(quant.generated_token_ids))
    common_prefix = 0
    for index in range(minimum):
        if base.generated_token_ids[index] != quant.generated_token_ids[index]:
            break
        common_prefix += 1
    position_matches = sum(
        int(left == right)
        for left, right in zip(base.generated_token_ids, quant.generated_token_ids, strict=False)
    )
    denominator = max(len(base.generated_token_ids), len(quant.generated_token_ids), 1)

    return PromptComparison(
        prompt=base.prompt,
        input_tokens=base.input_tokens,
        baseline_text=base.generated_text,
        quantized_text=quant.generated_text,
        baseline_generated_tokens=len(base.generated_token_ids),
        quantized_generated_tokens=len(quant.generated_token_ids),
        common_prefix_tokens=common_prefix,
        positional_token_agreement=position_matches / denominator,
        exact_generation_match=base.generated_token_ids == quant.generated_token_ids,
        last_logit_cosine=float(F.cosine_similarity(lhs, rhs, dim=0).item()),
        last_logit_rmse=float(diff.square().mean().sqrt().item()),
        last_logit_max_abs_error=float(diff.abs().max().item()),
        kl_baseline_to_quantized=kl,
        top1_match=bool(lhs.argmax().item() == rhs.argmax().item()),
        top5_overlap=topk_overlap(lhs, rhs, 5),
        top10_overlap=topk_overlap(lhs, rhs, 10),
    )


def load_ppl_tokens(
    tokenizer: Any,
    dataset_name: str,
    dataset_config: str,
    split: str,
    max_tokens: int,
) -> torch.Tensor:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional 'datasets' dependency to use --ppl") from exc
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(str(row["text"]) for row in dataset if row.get("text"))
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if encoded.numel() < 2:
        raise RuntimeError("The selected dataset produced fewer than two tokens")
    return encoded[: max_tokens + 1].contiguous()


def evaluate_perplexity(
    model: Any,
    token_ids: torch.Tensor,
    device: torch.device,
    seq_len: int,
) -> dict[str, float | int]:
    if seq_len < 2:
        raise ValueError("ppl_seq_len must be at least 2")
    total_nll = 0.0
    total_targets = 0
    start = 0
    with torch.inference_mode():
        while start < token_ids.numel() - 1:
            end = min(start + seq_len, token_ids.numel())
            chunk = token_ids[start:end].unsqueeze(0).to(device)
            outputs = model(input_ids=chunk, labels=chunk, use_cache=False)
            targets = int(chunk.shape[-1] - 1)
            total_nll += float(outputs.loss.float().item()) * targets
            total_targets += targets
            if end == token_ids.numel():
                break
            start = end - 1
    mean_nll = total_nll / max(total_targets, 1)
    return {
        "tokens": int(token_ids.numel()),
        "predicted_tokens": total_targets,
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 80.0)),
    }


def copy_matrix_chunked(destination: torch.Tensor, source_cpu: torch.Tensor, chunk_rows: int) -> None:
    if tuple(destination.shape) != tuple(source_cpu.shape):
        raise ValueError(f"Shape mismatch: destination={tuple(destination.shape)}, source={tuple(source_cpu.shape)}")
    with torch.inference_mode():
        for start in range(0, destination.shape[0], chunk_rows):
            end = min(destination.shape[0], start + chunk_rows)
            destination[start:end].copy_(
                source_cpu[start:end].to(device=destination.device, dtype=destination.dtype),
                non_blocking=False,
            )


def install_tied_weight(model: Any, weight_cpu: torch.Tensor, chunk_rows: int) -> dict[str, Any]:
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    if input_module is None or not hasattr(input_module, "weight"):
        raise RuntimeError("The model does not expose an input embedding weight")

    copy_matrix_chunked(input_module.weight, weight_cpu, chunk_rows)
    tied = False
    if output_module is not None and hasattr(output_module, "weight"):
        tied = output_module.weight.data_ptr() == input_module.weight.data_ptr()
        if not tied:
            copy_matrix_chunked(output_module.weight, weight_cpu, chunk_rows)
    return {
        "input_shape": list(input_module.weight.shape),
        "output_module_present": output_module is not None,
        "physically_tied": tied,
    }


def aggregate_prompt_metrics(rows: list[PromptComparison]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "prompt_count": float(len(rows)),
        "top1_match_rate": sum(row.top1_match for row in rows) / len(rows),
        "mean_top5_overlap": sum(row.top5_overlap for row in rows) / len(rows),
        "mean_top10_overlap": sum(row.top10_overlap for row in rows) / len(rows),
        "mean_last_logit_cosine": sum(row.last_logit_cosine for row in rows) / len(rows),
        "mean_kl_baseline_to_quantized": sum(row.kl_baseline_to_quantized for row in rows) / len(rows),
        "mean_positional_generation_agreement": sum(row.positional_token_agreement for row in rows) / len(rows),
        "exact_generation_match_rate": sum(row.exact_generation_match for row in rows) / len(rows),
    }


def markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# BitNet tied embedding/LM-head NVFP4 fake-quant report",
        "",
        f"- Model: `{results['model_id']}`",
        f"- Device/dtype: `{results['device']}` / `{results['dtype']}`",
        f"- Matrix shape: `{results['matrix_shape'][0]} x {results['matrix_shape'][1]}`",
        "- Important: this is a numerical quality experiment, not a native NVFP4 speed benchmark.",
        "",
        "## Summary",
        "",
        "| Variant | bits/weight | packed MiB | PPL | PPL ratio | top-1 match | logit cosine | KL(base||q) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_ppl = results.get("baseline", {}).get("ppl", {}).get("perplexity")
    for name, variant in results["variants"].items():
        stats = variant["quantization"]
        aggregate = variant["aggregate_prompt_metrics"]
        ppl = variant.get("ppl", {}).get("perplexity")
        ratio = (ppl / baseline_ppl) if ppl is not None and baseline_ppl else None
        lines.append(
            "| {name} | {bits:.5f} | {mib:.2f} | {ppl} | {ratio} | {top1:.1%} | {cos:.6f} | {kl:.6g} |".format(
                name=name,
                bits=stats["effective_bits_per_weight"],
                mib=stats["packed_bytes"] / (1024**2),
                ppl="—" if ppl is None else f"{ppl:.4f}",
                ratio="—" if ratio is None else f"{ratio:.4f}x",
                top1=aggregate.get("top1_match_rate", 0.0),
                cos=aggregate.get("mean_last_logit_cosine", float("nan")),
                kl=aggregate.get("mean_kl_baseline_to_quantized", float("nan")),
            )
        )

    for name, variant in results["variants"].items():
        lines.extend(["", f"## {name}", ""])
        for row in variant["prompt_comparisons"]:
            lines.extend(
                [
                    f"### {row['prompt']}",
                    "",
                    f"- top-1 match: `{row['top1_match']}`",
                    f"- top-5 overlap: `{row['top5_overlap']:.1%}`",
                    f"- logit cosine: `{row['last_logit_cosine']:.6f}`",
                    f"- KL(base||q): `{row['kl_baseline_to_quantized']:.6g}`",
                    f"- common generated prefix: `{row['common_prefix_tokens']}` tokens",
                    "",
                    "**Baseline**",
                    "",
                    row["baseline_text"] or "_(generation skipped)_",
                    "",
                    "**Quantized**",
                    "",
                    row["quantized_text"] or "_(generation skipped)_",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    prompts = load_prompts(args.prompts_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

    print(f"Loading {args.model_id} on {device} as {dtype} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    embedding = model.get_input_embeddings()
    if embedding is None or not hasattr(embedding, "weight"):
        raise RuntimeError("Could not locate the model input embedding")
    original_weight = embedding.weight.detach().cpu().clone()
    print(f"Tied matrix: {tuple(original_weight.shape)}, {original_weight.numel() / 1e6:.1f}M values")

    ppl_tokens = None
    if args.ppl:
        print("Loading perplexity corpus ...")
        ppl_tokens = load_ppl_tokens(
            tokenizer,
            args.dataset,
            args.dataset_config,
            args.dataset_split,
            args.ppl_tokens,
        )

    print("Capturing baseline prompts ...")
    baseline_prompts = capture_prompts(
        model, tokenizer, prompts, device, args.max_new_tokens, args.skip_generation
    )
    baseline_ppl = None
    if ppl_tokens is not None:
        print("Evaluating baseline perplexity ...")
        baseline_ppl = evaluate_perplexity(model, ppl_tokens, device, args.ppl_seq_len)
        print(f"  baseline PPL: {baseline_ppl['perplexity']:.4f}")

    results: dict[str, Any] = {
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "quant_work_device": args.quant_work_device,
        "matrix_shape": list(original_weight.shape),
        "seed": args.seed,
        "baseline": {"ppl": baseline_ppl, "prompt_count": len(baseline_prompts)},
        "variants": {},
    }

    for layout_name in args.layouts:
        layout: Layout = layout_name  # type: ignore[assignment]
        print(f"Fake-quantizing tied matrix as {layout} ...")
        started = time.perf_counter()
        quantized_weight, quant_stats = fake_quantize_nvfp4(
            original_weight,
            layout=layout,
            chunk_rows=args.chunk_rows,
            work_device=args.quant_work_device,
            output_device="cpu",
            output_dtype=original_weight.dtype,
        )
        quant_seconds = time.perf_counter() - started
        tie_info = install_tied_weight(model, quantized_weight, args.chunk_rows)
        del quantized_weight
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"Capturing {layout} prompts ...")
        quant_prompts = capture_prompts(
            model, tokenizer, prompts, device, args.max_new_tokens, args.skip_generation
        )
        comparisons = [
            compare_prompt(base, quant)
            for base, quant in zip(baseline_prompts, quant_prompts, strict=True)
        ]

        quant_ppl = None
        if ppl_tokens is not None:
            print(f"Evaluating {layout} perplexity ...")
            quant_ppl = evaluate_perplexity(model, ppl_tokens, device, args.ppl_seq_len)
            print(f"  {layout} PPL: {quant_ppl['perplexity']:.4f}")

        results["variants"][layout] = {
            "quantization_seconds": quant_seconds,
            "quantization": quant_stats.to_dict(),
            "tie_info": tie_info,
            "ppl": quant_ppl,
            "aggregate_prompt_metrics": aggregate_prompt_metrics(comparisons),
            "prompt_comparisons": [asdict(row) for row in comparisons],
        }

        install_tied_weight(model, original_weight, args.chunk_rows)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json_path = args.output_dir / "results.json"
    report_path = args.output_dir / "report.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, cls=JsonEncoder), encoding="utf-8")
    report_path.write_text(markdown_report(results), encoding="utf-8")

    print("\nResult summary")
    if baseline_ppl is not None:
        print(f"  baseline PPL: {baseline_ppl['perplexity']:.4f}")
    for name, variant in results["variants"].items():
        aggregate = variant["aggregate_prompt_metrics"]
        ppl = variant.get("ppl")
        ppl_text = "n/a" if not ppl else f"{ppl['perplexity']:.4f}"
        print(
            f"  {name:12s} PPL={ppl_text}, top1={aggregate.get('top1_match_rate', 0):.1%}, "
            f"cos={aggregate.get('mean_last_logit_cosine', float('nan')):.6f}, "
            f"KL={aggregate.get('mean_kl_baseline_to_quantized', float('nan')):.6g}"
        )
    print(f"Saved {json_path} and {report_path}")


if __name__ == "__main__":
    main()
