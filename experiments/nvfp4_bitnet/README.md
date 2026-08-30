# BitNet + NVFP4 tied embedding/LM-head quality demo

This experiment answers one narrow question:

> What happens to BitNet b1.58 2B-4T quality when the shared input embedding / output LM-head matrix is stored in an NVFP4-like format?

The BitNet transformer body is left unchanged. Only the tied `128256 x 2560` embedding/head matrix is fake-quantized and then dequantized back to the model dtype.

This is a **quality experiment**, not a native NVFP4 speed benchmark. It therefore runs on CPU, AMD ROCm GPUs such as MI50, older NVIDIA GPUs, and Apple Silicon. The fake-quant path does not make inference faster.

## Variants

- `row16`: one FP8 E4M3 scale per 16 consecutive weights in each vocabulary row, plus one tensor-wide FP32 scale. Effective storage is approximately **4.5 bits/weight**. This is the current FPGA-friendly candidate.
- `block16x16`: one FP8 E4M3 scale per 16x16 weight block, plus one tensor-wide FP32 scale. Effective storage is approximately **4.03125 bits/weight** and follows the default 2-D weight-scaling idea documented by NVIDIA Transformer Engine.

Both modes use FP4 E2M1 values with magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.

## What it measures

- Side-by-side deterministic generation for several prompts
- Last-token logit cosine similarity and RMSE
- `KL(baseline || quantized)`
- Top-1 equality and top-5/top-10 overlap
- Generated-token common-prefix and positional agreement
- Optional WikiText-2 perplexity
- Tied-matrix reconstruction error, saturation rate, packed size, and effective bits/weight

Exact generated text is a noisy indicator: tiny logit changes can make greedy decoding diverge early. Perplexity, KL divergence, and top-k agreement are the primary quality signals.

## Installation

Use a recent PyTorch build that exposes `torch.float8_e4m3fn`. On MI50, install the appropriate ROCm PyTorch wheel first.

```bash
cd experiments/nvfp4_bitnet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick prompt comparison

MI50 / ROCm:

```bash
python evaluate.py \
  --device cuda \
  --dtype float16 \
  --quant-work-device cpu \
  --layouts row16 block16x16 \
  --max-new-tokens 48
```

NVIDIA GPU with BF16 support:

```bash
python evaluate.py --device cuda --dtype bfloat16 --quant-work-device cpu
```

CPU-only smoke run:

```bash
python evaluate.py \
  --device cpu \
  --dtype float32 \
  --quant-work-device cpu \
  --layouts row16 \
  --skip-generation
```

## Perplexity run

```bash
python evaluate.py \
  --device cuda \
  --dtype float16 \
  --quant-work-device cpu \
  --layouts row16 block16x16 \
  --ppl \
  --ppl-tokens 8192 \
  --ppl-seq-len 512
```

Results are written to:

- `results/nvfp4_bitnet/results.json`
- `results/nvfp4_bitnet/report.md`

## Checkpoint choice

The default is the packed deployment checkpoint:

```text
microsoft/bitnet-b1.58-2B-4T
```

To test the BF16 master-weight execution path instead:

```bash
python evaluate.py --model-id microsoft/bitnet-b1.58-2B-4T-bf16 ...
```

The packed checkpoint is the more relevant system baseline. The BF16 checkpoint can be useful as a cross-check, but standard Transformers execution is not representative of optimized `bitnet.cpp` runtime speed.

## Suggested decision rule

These are engineering heuristics, not universal quality guarantees:

- PPL ratio `<= 1.02x`, top-1 match `>= 90%`, and very small KL: strong candidate
- PPL ratio `1.02–1.05x`: probably usable, but run task benchmarks
- PPL ratio `> 1.10x` or frequent top-1 flips: head precision is too aggressive without calibration/fine-tuning
- If `block16x16` loses too much quality but `row16` is healthy, spend the extra scale bandwidth and keep the FPGA `row16` format

## Why this experiment is intentionally unusual

Production NVFP4 recipes commonly preserve embeddings and `lm_head` in BF16, and Hugging Face's NVFP4 examples explicitly support excluding `lm_head`. This demo deliberately quantizes the sensitive tied matrix because it is a real bandwidth bottleneck for the Catapult3 design.

References:

- NVIDIA Transformer Engine NVFP4 format: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html
- Hugging Face NVFP4 integration and limitations: https://huggingface.co/docs/transformers/main/en/quantization/nvfp4
- Microsoft BitNet model: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
- Microsoft BitNet inference framework: https://github.com/microsoft/BitNet
