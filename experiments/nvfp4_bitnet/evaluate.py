#!/usr/bin/env python3
"""BitNet tied embedding/LM-head NVFP4 quality experiment.

Fake quantization is dequantized back to the model dtype, so this measures
quality rather than native NVFP4 speed and works on CPU/ROCm/pre-Blackwell GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nvfp4_fakequant import fake_quantize_nvfp4

PROMPTS = [
    "Explain in three concise sentences why the sky appears blue.",
    "Write a correct Python function that returns whether an integer is prime.",
    "Alice has 12 apples and gives 5 away. Explain how many remain.",
    "대한민국의 수도는 어디인지 한 문장으로 답해 주세요.",
    "Continue with a short technical paragraph: An old FPGA card can still be useful when",
]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="microsoft/bitnet-b1.58-2B-4T")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--quant-work-device", default="cpu")
    p.add_argument("--layouts", nargs="+", choices=["row16", "block16x16"], default=["row16", "block16x16"])
    p.add_argument("--chunk-rows", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--prompts-file", type=Path)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--ppl", action="store_true")
    p.add_argument("--ppl-tokens", type=int, default=4096)
    p.add_argument("--ppl-seq-len", type=int, default=512)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", default="test")
    p.add_argument("--output-dir", type=Path, default=Path("results/nvfp4_bitnet"))
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    explicit = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name in explicit:
        return explicit[name]
    if device.type == "cpu":
        return torch.float32
    if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported"):
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def read_prompts(path: Path | None) -> list[str]:
    if path is None:
        return PROMPTS
    values = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not values:
        raise ValueError(f"No prompts in {path}")
    return values


def encode_prompt(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": "You are a concise and helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
    except Exception:
        encoded = tokenizer(prompt, return_tensors="pt")
    if isinstance(encoded, torch.Tensor):
        encoded = {"input_ids": encoded}
    return {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}


def snapshot(model: Any, tokenizer: Any, prompt: str, device: torch.device,
             max_new_tokens: int, skip_generation: bool) -> dict[str, Any]:
    encoded = encode_prompt(tokenizer, prompt, device)
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[0, -1].detach().float().cpu()
    input_len = int(encoded["input_ids"].shape[-1])
    token_ids: list[int] = []
    text = ""
    if not skip_generation:
        kwargs: dict[str, Any] = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            kwargs["pad_token_id"] = tokenizer.eos_token_id
        with torch.inference_mode():
            output = model.generate(**encoded, **kwargs)[0, input_len:].detach().cpu()
        token_ids = [int(x) for x in output.tolist()]
        text = tokenizer.decode(output, skip_special_tokens=True)
    return {"prompt": prompt, "input_tokens": input_len, "ids": token_ids, "text": text, "logits": logits}


def capture(model: Any, tokenizer: Any, prompts: list[str], device: torch.device,
            max_new_tokens: int, skip_generation: bool) -> list[dict[str, Any]]:
    rows = []
    for i, prompt in enumerate(prompts, 1):
        print(f"  prompt {i}: {prompt[:72]}")
        rows.append(snapshot(model, tokenizer, prompt, device, max_new_tokens, skip_generation))
    return rows


def overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    return len(set(torch.topk(a, k).indices.tolist()) & set(torch.topk(b, k).indices.tolist())) / k


def compare(base: dict[str, Any], quant: dict[str, Any]) -> dict[str, Any]:
    a, b = base["logits"].float(), quant["logits"].float()
    d = b - a
    log_a, log_b = F.log_softmax(a, -1), F.log_softmax(b, -1)
    kl = float((log_a.exp() * (log_a - log_b)).sum().item())
    common = 0
    for left, right in zip(base["ids"], quant["ids"]):
        if left != right:
            break
        common += 1
    positional = sum(int(x == y) for x, y in zip(base["ids"], quant["ids"]))
    denom = max(len(base["ids"]), len(quant["ids"]), 1)
    return {
        "prompt": base["prompt"],
        "input_tokens": base["input_tokens"],
        "baseline_text": base["text"],
        "quantized_text": quant["text"],
        "common_prefix_tokens": common,
        "positional_token_agreement": positional / denom,
        "exact_generation_match": base["ids"] == quant["ids"],
        "last_logit_cosine": float(F.cosine_similarity(a, b, dim=0).item()),
        "last_logit_rmse": float(d.square().mean().sqrt().item()),
        "last_logit_max_abs_error": float(d.abs().max().item()),
        "kl_baseline_to_quantized": kl,
        "top1_match": bool(a.argmax().item() == b.argmax().item()),
        "top5_overlap": overlap(a, b, 5),
        "top10_overlap": overlap(a, b, 10),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(rows), 1)
    return {
        "prompt_count": len(rows),
        "top1_match_rate": sum(r["top1_match"] for r in rows) / n,
        "mean_top5_overlap": sum(r["top5_overlap"] for r in rows) / n,
        "mean_top10_overlap": sum(r["top10_overlap"] for r in rows) / n,
        "mean_last_logit_cosine": sum(r["last_logit_cosine"] for r in rows) / n,
        "mean_kl_baseline_to_quantized": sum(r["kl_baseline_to_quantized"] for r in rows) / n,
        "mean_positional_generation_agreement": sum(r["positional_token_agreement"] for r in rows) / n,
        "exact_generation_match_rate": sum(r["exact_generation_match"] for r in rows) / n,
    }


def corpus_tokens(tokenizer: Any, args: argparse.Namespace) -> torch.Tensor:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to use --ppl") from exc
    ds = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    text = "\n\n".join(str(row["text"]) for row in ds if row.get("text"))
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    return ids[:args.ppl_tokens + 1].contiguous()


def perplexity(model: Any, ids: torch.Tensor, device: torch.device, seq_len: int) -> dict[str, float | int]:
    total_nll, total_targets, start = 0.0, 0, 0
    with torch.inference_mode():
        while start < ids.numel() - 1:
            end = min(start + seq_len, ids.numel())
            chunk = ids[start:end].unsqueeze(0).to(device)
            out = model(input_ids=chunk, labels=chunk, use_cache=False)
            count = int(chunk.shape[-1] - 1)
            total_nll += float(out.loss.float().item()) * count
            total_targets += count
            if end == ids.numel():
                break
            start = end - 1
    mean = total_nll / max(total_targets, 1)
    return {"tokens": int(ids.numel()), "predicted_tokens": total_targets,
            "mean_nll": mean, "perplexity": math.exp(min(mean, 80.0))}


def copy_weight(destination: torch.Tensor, source: torch.Tensor, rows: int) -> None:
    if destination.shape != source.shape:
        raise ValueError(f"Weight shape mismatch: {destination.shape} vs {source.shape}")
    with torch.inference_mode():
        for start in range(0, destination.shape[0], rows):
            end = min(start + rows, destination.shape[0])
            destination[start:end].copy_(source[start:end].to(destination.device, destination.dtype))


def install_weight(model: Any, source: torch.Tensor, rows: int) -> dict[str, Any]:
    inp, out = model.get_input_embeddings(), model.get_output_embeddings()
    if inp is None or not hasattr(inp, "weight"):
        raise RuntimeError("Model input embedding was not found")
    copy_weight(inp.weight, source, rows)
    physically_tied = False
    if out is not None and hasattr(out, "weight"):
        physically_tied = out.weight.data_ptr() == inp.weight.data_ptr()
        if not physically_tied:
            copy_weight(out.weight, source, rows)
    return {"output_module_present": out is not None, "physically_tied": physically_tied}


def write_report(result: dict[str, Any], path: Path) -> None:
    base_ppl = (result["baseline"].get("ppl") or {}).get("perplexity")
    lines = [
        "# BitNet tied embedding/LM-head NVFP4 fake-quant report", "",
        f"- Model: `{result['model_id']}`",
        f"- Device/dtype: `{result['device']}` / `{result['dtype']}`",
        "- Numerical quality experiment only; not a native NVFP4 speed benchmark.", "",
        "| Variant | bits/weight | packed MiB | PPL | PPL ratio | top-1 | cosine | KL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, variant in result["variants"].items():
        q, a = variant["quantization"], variant["aggregate"]
        p = (variant.get("ppl") or {}).get("perplexity")
        ratio = p / base_ppl if p is not None and base_ppl else None
        lines.append(
            f"| {name} | {q['effective_bits_per_weight']:.5f} | {q['packed_bytes']/2**20:.2f} | "
            f"{'—' if p is None else f'{p:.4f}'} | {'—' if ratio is None else f'{ratio:.4f}x'} | "
            f"{a['top1_match_rate']:.1%} | {a['mean_last_logit_cosine']:.6f} | "
            f"{a['mean_kl_baseline_to_quantized']:.6g} |"
        )
    for name, variant in result["variants"].items():
        lines += ["", f"## {name}"]
        for row in variant["prompts"]:
            lines += [
                "", f"### {row['prompt']}",
                f"Top-1: `{row['top1_match']}`, top-5 overlap: `{row['top5_overlap']:.1%}`, "
                f"cosine: `{row['last_logit_cosine']:.6f}`, KL: `{row['kl_baseline_to_quantized']:.6g}`, "
                f"common prefix: `{row['common_prefix_tokens']}` tokens", "",
                "**Baseline**", "", row["baseline_text"] or "_(generation skipped)_", "",
                "**Quantized**", "", row["quantized_text"] or "_(generation skipped)_",
            ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    prompts = read_prompts(args.prompts_file)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

    print(f"Loading {args.model_id} on {device} as {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    embedding = model.get_input_embeddings()
    if embedding is None:
        raise RuntimeError("Model input embedding was not found")
    original = embedding.weight.detach().cpu().clone()
    print(f"Tied matrix: {tuple(original.shape)}, {original.numel()/1e6:.1f}M weights")

    ids = corpus_tokens(tokenizer, args) if args.ppl else None
    print("Baseline prompts")
    base = capture(model, tokenizer, prompts, device, args.max_new_tokens, args.skip_generation)
    base_ppl = perplexity(model, ids, device, args.ppl_seq_len) if ids is not None else None
    result: dict[str, Any] = {
        "model_id": args.model_id, "device": str(device),
        "dtype": str(dtype).removeprefix("torch."), "matrix_shape": list(original.shape),
        "baseline": {"ppl": base_ppl}, "variants": {},
    }

    for layout in args.layouts:
        print(f"Quantizing {layout}")
        started = time.perf_counter()
        qweight, stats = fake_quantize_nvfp4(
            original, layout=layout, chunk_rows=args.chunk_rows,
            work_device=args.quant_work_device, output_device="cpu", output_dtype=original.dtype,
        )
        tie = install_weight(model, qweight, args.chunk_rows)
        del qweight
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{layout} prompts")
        quant = capture(model, tokenizer, prompts, device, args.max_new_tokens, args.skip_generation)
        rows = [compare(a, b) for a, b in zip(base, quant)]
        q_ppl = perplexity(model, ids, device, args.ppl_seq_len) if ids is not None else None
        result["variants"][layout] = {
            "quantization_seconds": time.perf_counter() - started,
            "quantization": stats.to_dict(), "tie_info": tie,
            "ppl": q_ppl, "aggregate": aggregate(rows), "prompts": rows,
        }
        install_weight(model, original, args.chunk_rows)

    json_path, report_path = args.output_dir / "results.json", args.output_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(result, report_path)
    print(f"Saved {json_path} and {report_path}")


if __name__ == "__main__":
    main()
