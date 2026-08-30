# Catapult3 CPU model selection v1

This directory contains a reproducible CPU-only comparison for choosing the
first Catapult3 Rev E end-to-end model target. It does not implement RTL or
claim board measurements.

The experiment separates three evidence classes in every result:

- `CPU_MEASURED`: fixed-checkpoint fake-quantization or official native-runtime
  output measured on the host.
- `MODEL_FILE_CALCULATED`: tensor geometry, hashes, and packed sizes derived
  from pinned model files.
- `CATAPULT_ASSUMPTION_ESTIMATE`: DDR and compute roofs based on explicitly
  listed bandwidth, utilization, clock, and lane assumptions.

## Candidates and backends

| Candidate | Weight contract | Evaluation backend |
|---|---|---|
| `1bitLLM/bitnet_b1_58-large` | ternary body, public 0.7B reproduction | frozen checkpoint Python code + PyTorch CPU |
| `prism-ml/Bonsai-1.7B-gguf` | binary `Q1_0` g128 | official PrismML llama.cpp release |
| `prism-ml/Bonsai-1.7B-unpacked` | unpacked reference for boundary experiments | Transformers/PyTorch CPU |
| Bonsai 4B/8B | binary `Q1_0` g128 geometry | metadata and roof calculation only |

The BitNet checkpoint is a public reproduction, not a Microsoft official
checkpoint. Bonsai is binary `{-1,+1}` plus group scales; it is not ternary.

## Numerical contracts

- Activation and KV codes reserve the most-negative two's-complement value:
  A/KV3 `[-3,3]`, A/KV4 `[-7,7]`, A/KV8 `[-127,127]`.
- Scale is `amax / qmax`; zero groups use scale 1 and code 0.
- Rounding is round-to-nearest, ties-to-even, followed by explicit clipping.
- Activation scale is per token or per 128-element group. KV scale is per
  token and head across head dimension, after RoPE for K.
- Cached K/V prefixes are stored quantized once. A regression test proves a
  later append does not requantize the prefix.
- Bonsai binary-linear reference uses W1 g128, per-group integer dot products,
  INT32/INT24/INT20 saturation candidates, and FP, unsigned Q4.20, or Q12
  group-output scale paths.
- Bankai row XOR is tested only at the biasless binary integer-accumulator
  boundary. Flipping every sign in an output row must negate that accumulator
  bit-exactly; attention and MLP shapes are tested separately.

## Files

- `run_bitnet.py`: baseline, KV8/4/3, row16 head, row16+KV4/3, and retained
  block16x16 head comparison.
- `run_bonsai.py`: BF16 reference, A12/A10/A8/A8-g128, KV8/4/3, and the
  accumulator/scale matrix.
- `run_bonsai_native.py`: official Q1_0 GGUF generation and PPL, including a
  hash of the temporary full-logits binary.
- `hardware_roof.py`: contexts 128/512/2048/4096, KV8/4/3, sustained
  29/31/33/35 GB/s, 80/90/95% utilization, and 200/210/225/240 MHz.
- `result_schema.json`: strict common result envelope.
- `artifact_manifest.json`: pinned revisions, licenses, sizes, and SHA-256.
- `environment.lock.txt`: exact packages used for the checked-in measurements.
- `results/summary.md`: compact human-readable result.

## Reproduction

Create one environment and install the requirements:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r experiments/model_selection/requirements.txt
```

Download checkpoints sequentially into directories outside Git. `hf download`
can pin the exact snapshots:

```powershell
hf download 1bitLLM/bitnet_b1_58-large --revision 85d047191dcb224f0e04f20d26110caaf8dc1a47 --local-dir C:\models\bitnet-0.7b
hf download prism-ml/Bonsai-1.7B-unpacked --revision a7f720bd688d7563714f3118edd97b83d06f0615 --local-dir C:\models\bonsai-1.7b-unpacked
hf download prism-ml/Bonsai-1.7B-gguf Bonsai-1.7B-Q1_0.gguf --revision 210a9e99f79cb184909d49595906526eb2b3dd9a --local-dir C:\models\bonsai-1.7b-gguf
```

Verify every downloaded byte against `artifact_manifest.json`, then run each
model separately:

```powershell
python experiments/model_selection/run_bitnet.py --checkpoint-dir C:\models\bitnet-0.7b --output experiments/model_selection/results/bitnet_0_7b_smoke.json
python experiments/model_selection/run_bonsai.py --checkpoint-dir C:\models\bonsai-1.7b-unpacked --output experiments/model_selection/results/bonsai_1_7b_reference_smoke.json
python experiments/model_selection/run_bonsai_native.py --runtime-dir C:\tools\prism-b10660-e311ed3 --runtime-archive C:\downloads\llama-prism-b10660-e311ed3-bin-win-cpu-x64.zip --model C:\models\bonsai-1.7b-gguf\Bonsai-1.7B-Q1_0.gguf --output experiments/model_selection/results/bonsai_1_7b_native_smoke.json
python experiments/model_selection/hardware_roof.py --output experiments/model_selection/results/hardware_roof.json
python -m pytest experiments/model_selection/tests -q
```

Use `--run-mode full` for 4096 predicted tokens in the PyTorch adapters. Smoke
defaults are 256 predicted tokens, five prompts, seed 20260830, and eight CPU
threads. The models are loaded sequentially; measured peak RSS was about
5.14 GB for BitNet and 7.25 GB for the unpacked Bonsai reference, so the smoke
path is suitable for a 16 GB host. Full mode is intentionally not represented
by the checked-in smoke results.

## Interpretation boundary

The fixed text is deliberately small and repeated when necessary to reach the
token count. PPL values are comparable only within the same tokenizer/backend.
They can catch catastrophic regressions but are not publication-grade quality
evidence. See `docs/experiments/model-selection-cpu-v1.md` for the decision and
the single experiment still required.
