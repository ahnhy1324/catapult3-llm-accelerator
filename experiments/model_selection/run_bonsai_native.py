#!/usr/bin/env python3
"""Run the official PrismML Q1_0 GGUF with its pinned CPU runtime.

Large checkpoint, runtime, and temporary full-logit files stay outside Git.
Only their byte hashes and the compact common-schema result are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil


MODEL_ID = "prism-ml/Bonsai-1.7B-gguf"
MODEL_REVISION = "210a9e99f79cb184909d49595906526eb2b3dd9a"
MODEL_BYTES = 248_302_272
MODEL_SHA256 = "3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3"
RUNTIME_TAG = "prism-b10660-e311ed3"
RUNTIME_COMMIT = "e311ed38fe7ab8fb577a5435b049d48b7d040923"
RUNTIME_ARCHIVE = "llama-prism-b10660-e311ed3-bin-win-cpu-x64.zip"
RUNTIME_ARCHIVE_BYTES = 18_731_604
RUNTIME_ARCHIVE_SHA256 = "c87e4ae315d17b8ef9695001db7ad0f9eb8ab275c33d11c02395c64d844fe764"


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, default=root / "prompts.txt")
    parser.add_argument("--ppl-text", type=Path, default=root / "ppl_smoke.txt")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--ppl-chunks", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_measured(command: list[str]) -> tuple[subprocess.CompletedProcess[str], int, float]:
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    measured_process = psutil.Process(process.pid)
    peak = 0
    while process.poll() is None:
        try:
            peak = max(peak, measured_process.memory_info().rss)
            for child in measured_process.children(recursive=True):
                peak = max(peak, child.memory_info().rss)
        except (psutil.Error, OSError):
            pass
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"native command failed ({completed.returncode}):\n{completed.stderr[-4000:]}")
    return completed, peak, elapsed


def require_bytes(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"byte-size mismatch for {path}: {path.stat().st_size} != {expected_bytes}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")


def parse_rate(stderr: str, label: str) -> float | None:
    matches = re.findall(rf"{re.escape(label)} time.*?([0-9.]+) tokens per second", stderr)
    return float(matches[-1]) if matches else None


def main() -> None:
    args = arguments()
    if args.context % 2:
        raise ValueError("llama.cpp perplexity context must be even")
    prompts = [line.strip() for line in args.prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) < 5:
        raise ValueError("at least five fixed prompts are required")

    completion = args.runtime_dir / "llama-completion.exe"
    perplexity_exe = args.runtime_dir / "llama-perplexity.exe"
    for executable in (completion, perplexity_exe):
        if not executable.is_file():
            raise FileNotFoundError(executable)
    require_bytes(args.model, MODEL_BYTES, MODEL_SHA256)
    if args.runtime_archive is not None:
        require_bytes(args.runtime_archive, RUNTIME_ARCHIVE_BYTES, RUNTIME_ARCHIVE_SHA256)

    version, _, _ = run_measured([str(completion), "--version"])
    version_text = (version.stdout + version.stderr).strip()
    if "commit e311ed38f" not in version_text:
        raise RuntimeError(f"unexpected PrismML runtime version: {version_text!r}")

    peak_rss = 0
    wall_seconds = 0.0
    prompt_rows: list[dict[str, Any]] = []
    generation_rates: list[float] = []
    for prompt in prompts:
        command = [
            str(completion), "-m", str(args.model), "-sys", "You are a concise and helpful assistant.",
            "-p", prompt, "-n", str(args.max_new_tokens), "--temp", "0", "--seed", str(args.seed),
            "-t", str(args.threads), "-tb", str(args.threads), "-c", "512", "--no-display-prompt",
            "--simple-io", "--no-warmup", "-st",
        ]
        completed, peak, elapsed = run_measured(command)
        peak_rss = max(peak_rss, peak)
        wall_seconds += elapsed
        rate = parse_rate(completed.stderr, "eval")
        if rate is not None:
            generation_rates.append(rate)
        generated = completed.stdout.strip()
        prompt_rows.append(
            {
                "prompt": prompt,
                "generated_text": generated,
                "generated_output_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
                "top10": None,
                "top10_unavailable_reason": "official completion CLI does not expose full-vocabulary top-k",
            }
        )

    # llama.cpp scores the latter half of each context window. Repeat the
    # frozen text so four 128-token chunks provide 4 * 64 = 256 targets.
    source_text = args.ppl_text.read_text(encoding="utf-8")
    predicted_tokens = args.ppl_chunks * (args.context // 2)
    with tempfile.TemporaryDirectory(prefix="catapult3-bonsai-native-") as temp_name:
        temp = Path(temp_name)
        repeated_text = temp / "ppl_repeated.txt"
        logits_file = temp / "all_logits.bin"
        repeats = max(2, math.ceil((args.ppl_chunks * args.context * 4) / max(len(source_text), 1)))
        repeated_text.write_text((source_text + "\n") * repeats, encoding="utf-8")
        command = [
            str(perplexity_exe), "-m", str(args.model), "-f", str(repeated_text), "--chunks", str(args.ppl_chunks),
            "-c", str(args.context), "-b", str(args.context), "-ub", str(args.context), "--no-warmup",
            "-t", str(args.threads), "-tb", str(args.threads), "--save-all-logits", str(logits_file),
        ]
        ppl_run, peak, elapsed = run_measured(command)
        peak_rss = max(peak_rss, peak)
        wall_seconds += elapsed
        match = re.search(r"Final estimate: PPL = ([0-9.eE+-]+) \+/- ([0-9.eE+-]+)", ppl_run.stderr)
        if not match:
            raise RuntimeError(f"could not parse native perplexity output:\n{ppl_run.stderr[-4000:]}")
        ppl = float(match.group(1))
        ppl_uncertainty = float(match.group(2))
        logits_hash = sha256_file(logits_file)
        logits_bytes = logits_file.stat().st_size

    result = {
        "schema_version": "catapult3-model-selection-result-v1",
        "run_id": "bonsai-1.7b-q1_0-native-smoke-seed-20260830",
        "run_mode": "smoke",
        "status": "PASS",
        "evidence_scope": ["CPU_MEASURED", "MODEL_FILE_CALCULATED"],
        "environment": {
            "platform": "Windows x86_64",
            "seed": args.seed,
            "threads": args.threads,
            "deterministic_greedy": True,
        },
        "model": {
            "candidate": "Bonsai 1.7B true-1bit native GGUF",
            "model_id": MODEL_ID,
            "architecture": "Qwen3ForCausalLM / W1 Q1_0 g128",
            "parameter_class": "1.7B",
            "packed_model_size_bytes": MODEL_BYTES,
        },
        "checkpoint": {
            "revision": MODEL_REVISION,
            "files": [{"name": args.model.name, "byte_size": MODEL_BYTES, "sha256": MODEL_SHA256, "verification": "LOCAL_BYTES"}],
        },
        "backend": {
            "name": "PrismML llama.cpp official Windows CPU release",
            "revision": RUNTIME_COMMIT,
            "tag": RUNTIME_TAG,
            "execution_path": "llama-completion + llama-perplexity Q1_0 native",
            "version_output": version_text,
        },
        "health": {
            "finite": math.isfinite(ppl),
            "all_zero_tensors": [],
            "missing_tensors": [],
            "unexpected_tensors": [],
            "abnormal_scales": [],
            "inspection_scope": "strict GGUF loader success plus finite native evaluation; tensor-value scan is in unpacked-reference result",
        },
        "baseline": {
            "logits_sha256": logits_hash,
            "logits_hash_scope": "llama-perplexity --save-all-logits binary",
            "logits_file_bytes": logits_bytes,
            "generation_output_sha256": sha256_json(prompt_rows),
            "prompts": prompt_rows,
            "perplexity": {
                "predicted_tokens": predicted_tokens,
                "mean_nll": math.log(ppl),
                "perplexity": ppl,
                "reported_uncertainty": ppl_uncertainty,
                "context": args.context,
                "chunks": args.ppl_chunks,
                "evidence_label": "SMOKE_ONLY_REPEATED_FIXED_TEXT_NOT_PUBLICATION_GRADE",
            },
        },
        "variants": [],
        "performance": {
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": peak_rss,
            "mean_native_generation_tokens_per_second": sum(generation_rates) / max(len(generation_rates), 1),
            "generation_rate_samples": generation_rates,
        },
        "artifacts": [
            {
                "name": RUNTIME_ARCHIVE,
                "byte_size": RUNTIME_ARCHIVE_BYTES,
                "sha256": RUNTIME_ARCHIVE_SHA256,
                "verification": "LOCAL_BYTES" if args.runtime_archive is not None else "UPSTREAM_RELEASE_METADATA",
            }
        ],
        "blockers": [
            {
                "code": "NATIVE_PROMPT_TOPK_NOT_EXPOSED",
                "impact": "native full-logit hash is retained, while prompt top-k is supplied by the unpacked-reference adapter",
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "ppl": ppl, "logits_sha256": logits_hash}, indent=2))


if __name__ == "__main__":
    main()
