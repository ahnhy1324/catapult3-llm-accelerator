# Catapult3 model selection

This directory contains the reproducible CPU and projected-FPGA evidence used
to choose the first Catapult3 Rev E end-to-end checkpoint. The original v1
results remain under `experiments/model_selection/results/`; v2 results are
written to `results/model_selection_v2/` and do not overwrite v1.

## Evidence labels

- `MEASURED_CPU`: output produced by an executed CPU model, numeric reference,
  unit test, or functional simulator.
- `MEASURED_MODEL_FILE`: hashes, headers, and geometry read from pinned files.
- `CALCULATED_FROM_CONFIG`: deterministic counts derived from pinned configs.
- `PROJECTED_FPGA`: memory/compute roofs or post-fit Quartus estimates, not
  board measurements.
- `ASSUMPTION`: a selected bandwidth, utilization, lane, clock, or codec
  contract.
- `BLOCKED`: an explicitly unavailable comparison; no placeholder success.

## Pinned candidates

| Candidate | Role | Backend |
|---|---|---|
| `1bitLLM/bitnet_b1_58-large` | public 0.7B ternary reproduction | frozen checkpoint code + PyTorch |
| `prism-ml/Bonsai-1.7B-gguf` | official binary Q1 g128 candidate | pinned PrismML llama.cpp release |
| `prism-ml/Bonsai-1.7B-unpacked` | boundary/quantization reference | Transformers + PyTorch |
| `tiiuae/Falcon3-1B-Instruct-1.58bit` | limited small instruct investigation | config/file projection; execution status in v2 report |
| Bonsai 4B/8B | long-term geometry only | config and roof projection |

BitNet 0.7B is a reproduction, not a Microsoft official checkpoint. Bonsai
weights are binary `{-1,+1}` with group scales, not ternary.

## V2 contracts

- `manifest_verify.py` hashes every file used by a run and fails closed before
  model loading. Safetensors tensor names, shapes, dtypes, and ranges are also
  checked; GGUF magic/version is checked before the strict native loader.
- `fixed_byte_eval.py` scores the same pinned UTF-8 byte sequence for both
  tokenizers. Its overlapping 512/2048-token windows retain prefix context but
  score every target once. BPB is the cross-tokenizer metric; PPL ratios are
  within-model only.
- KV is quantized at cache write, after RoPE for K. K/V bit widths and scales
  are independently configurable, codes reserve the signed minimum, and
  rounding is ties-to-even followed by clipping/saturation.
- `hardware_roof.py` includes the full LM head, keeps projection, attention,
  scale, vector, top-k, and memory roofs separate, and labels host-head offload
  separately from the fully-on-card headline mode.
- `rtl/microbench/` compares binary g128, direct ternary, and TL5 at II=1 using
  NumPy golden vectors and a reproducible exact-part Quartus flow.
- Bankai row patches are valid only at the wide symmetric projection sum before
  residual addition, narrowing, or asymmetric saturation. See the architecture
  note and tests.

## Reproduction

Create an environment and install the exact v2 dependencies:

```powershell
python -m venv C:\msv2env
C:\msv2env\Scripts\python.exe -m pip install -r experiments\model_selection\requirements-v2.txt
```

Place the pinned snapshots outside Git and verify them before use:

```powershell
python experiments\model_selection\manifest_verify.py --manifest experiments\model_selection\artifact_manifest_v2.json --artifact-key bitnet-0.7b --root C:\models\bitnet-0.7b --safetensors model.safetensors
python experiments\model_selection\manifest_verify.py --manifest experiments\model_selection\artifact_manifest_v2.json --artifact-key bonsai-1.7b-unpacked --root C:\models\bonsai-1.7b-unpacked --safetensors model.safetensors
```

The checked-in orchestration script supports `test`, `roof`, `summary`,
`rtl-iverilog`, `rtl-quartus`, `bitnet`, `bonsai`, and `native-bonsai`
targets:

```bash
MODEL_SELECTION_ARTIFACT_ROOT=/models PYTHON_BIN=python \
  scripts/run_model_selection_v2.sh test
```

Large checkpoints, caches, native binaries, full-logit captures, and Quartus
build databases must remain outside Git. The report at
`docs/experiments/model-selection-cpu-v2.md` records measured commands,
blockers, the selected bring-up checkpoint, and the separate headline target.

On Windows, launch the limited Falcon run with `PYTHONUTF8=1`; PyTorch's
Inductor package contains UTF-8 templates that the legacy CP949 default cannot
read. The runner deliberately keeps the CPU weight-unpack reference eager, so
an external MSVC compiler is not required for this bounded health smoke.
