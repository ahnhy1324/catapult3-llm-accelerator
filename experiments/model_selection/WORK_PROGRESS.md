# Model-selection experiment history

## CPU v1

The v1 experiment established reproducible CPU smoke baselines for the public
`1bitLLM/bitnet_b1_58-large` reproduction, the official unpacked Bonsai 1.7B
reference, and the official Bonsai Q1_0 GGUF through pinned PrismML runtime
build `prism-b10660-e311ed3`.

Completed v1 work included:

- model/runtime identity, license, revision, and file-hash capture;
- BitNet baseline, FP4 LM-head, KV8/KV4/KV3, and joint smoke variants;
- Bonsai activation, KV, accumulator-width, and scale-rounding smoke variants;
- native Bonsai Q1_0 prompt and perplexity smoke through the pinned runtime;
- cache-prefix quantization regression coverage;
- initial Catapult3 memory-roof projections.

The v1 decision was intentionally deferred because its roof counted the
Transformer body but omitted the full LM-head projection, and its quality
comparison did not score the same fixed UTF-8 byte span across tokenizers.
Those are methodological limitations of v1, not results to silently reinterpret.

## CPU/RTL v2

The v2 work is tracked by committed code, result JSON/CSV files, and
`docs/experiments/model-selection-cpu-v2.md`. It adds fail-closed artifact
verification, fixed-byte BPB at 512/2048-token contexts, actually executed
joint variants, full on-card e2e body+LM-head roofs, Bankai fixed-point patch
tests, and minimal binary/ternary/TL5 RTL microbenchmarks.

Large checkpoints, GGUF files, runtime archives, generated model captures, and
Quartus build databases remain external artifacts and are never committed.
