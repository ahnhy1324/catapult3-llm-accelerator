# Catapult3 Rev E model selection: CPU v1

Date: 2026-08-31

Base commit: `886a61131bcc73be12f532b36e322f5c0743e3f5`

Work branch: `work/model-selection-cpu-v1`

## Outcome

Select conclusion 3: do not choose the first FPGA e2e model yet.

BitNet 0.7B is the clear throughput candidate, but the available public model
is a reproduction with weak absolute smoke behavior and no publication-grade
quality result here. Bonsai 1.7B has a verified official 248 MB Q1_0 artifact,
good native CPU behavior, and stable A8-g128/KV8 smoke quality, but its FPGA
100 tok/s envelope is narrow and its KV4/KV3 fake-quant results are
catastrophic. The experiment is intentionally not forcing a winner from these
incompatible uncertainties.

The single decisive next experiment is one 4096-predicted-token non-repeated
head-to-head evaluation of the exact hardware profiles: BitNet row16+KV4 and
Bonsai A8-g128+KV8. Use identical source bytes, report bits per UTF-8 byte plus
a fixed task/prompt subset, and evaluate contexts 512 and 2048. This removes
tokenizer bias, tests the quality/throughput trade directly, and is sufficient
to choose between conclusions 1 and 2 without changing RTL.

## Scope and provenance

No RTL, synthesis project, board, bitstream, or production repository was
modified. Models were evaluated sequentially on CPU and original checkpoint
bytes were never changed.

| Candidate | Frozen identity | License | Role |
|---|---|---|---|
| BitNet 0.7B | `1bitLLM/bitnet_b1_58-large@85d0471...` | MIT | public ternary reproduction |
| Bonsai 1.7B packed | `prism-ml/Bonsai-1.7B-gguf@210a9e9...` | Apache-2.0 | official deployment artifact |
| Bonsai 1.7B unpacked | `prism-ml/Bonsai-1.7B-unpacked@a7f720b...` | Apache-2.0 | activation/accumulator reference |
| Prism runtime | `PrismML-Eng/llama.cpp@e311ed3...`, build 10660 | upstream release | official native Q1_0 execution |

The BitNet model is not a Microsoft official checkpoint. Bonsai W1 is binary,
not ternary. Every required file hash is in
`experiments/model_selection/artifact_manifest.json`.

## Reproducibility and health

The environment used Python 3.12.13, PyTorch 2.13.0 CPU, Transformers
4.52.0.dev0, eight threads, seed 20260830, and deterministic PyTorch
algorithms. Exact package versions are retained in `environment.lock.txt`.

Both unpacked checkpoints passed finite, missing/unexpected tensor, all-zero,
and scale checks. BitNet has 728,842,752 parameters and peaked at
5,135,036,416 bytes RSS. Bonsai unpacked has 1,720,028,160 parameters and
peaked at 7,245,250,560 bytes. The native GGUF passed strict runtime loading,
finite PPL, five deterministic prompt generations, and a full-logits hash.

The BitNet baseline and six pre-existing variant logits hashes matched exactly
across the two final deterministic executions. Seventeen unit/schema/roof tests
pass, including cache append semantics and bit-exact Bankai attention/MLP row
flip equivalence.

## CPU-measured quality smoke

The smoke corpus has only 256 predicted tokens and is repeated. PPL is only
comparable within the same tokenizer and backend. Values below diagnose codec
behavior; they do not rank the two models' absolute quality.

### BitNet 0.7B

Baseline PPL was 819.2822. The native BitNet body activation contract was left
unchanged; only LM-head fake quantization and KV cache storage were varied.

| Variant | PPL ratio | top-1 match | logit cosine | KL(base||variant) | exact generation |
|---|---:|---:|---:|---:|---:|
| KV8 | 1.0041 | 1.00 | 0.999925 | 0.000934 | 0.60 |
| KV4 | 1.1535 | 1.00 | 0.999621 | 0.003162 | 0.80 |
| KV3 | 1.1938 | 1.00 | 0.998442 | 0.014823 | 0.80 |
| row16 head | 1.0399 | 0.80 | 0.999300 | 0.009175 | 0.40 |
| row16 + KV4 | 1.1278 | 0.80 | 0.998994 | 0.013115 | 0.40 |
| row16 + KV3 | 0.9947 | 1.00 | 0.997537 | 0.017318 | 0.60 |
| block16x16 head | 1.0034 | 1.00 | 0.998204 | 0.015942 | 0.40 |

The row16+KV3 ratio below one is treated as noise. It conflicts with the larger
logit error and is not an improvement claim. KV saturation was zero for all
variants under the frozen post-round clipping definition.

### Bonsai 1.7B unpacked reference

Baseline PPL was 141.4514. Fake quantization was applied only at the input of
attention/MLP binary-linear projections.

| Variant | PPL ratio | top-1 match | logit cosine | KL(base||variant) | exact generation |
|---|---:|---:|---:|---:|---:|
| A12 per token | 1.0029 | 1.00 | 0.999957 | 0.000241 | 1.00 |
| A10 per token | 0.9939 | 1.00 | 0.999923 | 0.000528 | 1.00 |
| A8 per token | 1.0095 | 1.00 | 0.999466 | 0.001150 | 0.60 |
| A8 g128 | 1.0034 | 1.00 | 0.999868 | 0.001444 | 1.00 |
| KV8 | 1.0096 | 0.80 | 0.999711 | 0.002422 | 0.80 |
| KV4 | 48.2688 | 0.00 | 0.647612 | 6.425608 | 0.00 |
| KV3 | 87.6859 | 0.00 | 0.586917 | 9.390449 | 0.00 |

A8 g128 is the preferred activation candidate from this smoke. KV8 remains a
candidate. KV4/KV3 fail decisively under the specified per-token/per-head
post-RoPE cache codec. Stored cache prefixes are not requantized, so this is
not the earlier adapter failure mode.

Representative layers 0/14/27 sampled one attention and one MLP projection.
Across 216 activation/accumulator/scale cases, INT32/INT24/INT20 produced zero
accumulator saturations. FP-scale mean relative RMSE was 0.001947 for A12,
0.005324 for A10, 0.020312 for A8, and 0.008343 for A8 g128. Unsigned Q4.20
pre-multiply scale and Q12 group-output rounding results are retained row by
row in the result JSON. The unpacked BF16 representation showed at most
0.002201 relative magnitude spread inside nominal weight groups; exact packed
sign/scale behavior is supplied by the native GGUF run.

### Official Bonsai Q1_0 native run

The release ZIP SHA-256 is
`c87e4ae315d17b8ef9695001db7ad0f9eb8ab275c33d11c02395c64d844fe764`.
The GGUF SHA-256 is
`3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3`.

The pinned runtime produced 208.53 tok/s mean generation on this CPU across
five prompts. Its 256-target repeated-text PPL was 107.0377 +/- 25.4871. The
temporary full-logits binary was hashed before deletion as
`57e00d9ca99e5ace7afc2451de1f890359f8efeda1e62c0d1ab98e3fa7badaf1`.
The official completion CLI does not expose prompt top-10 values; those remain
available from the unpacked adapter rather than being fabricated for the
native result.

## Catapult projection

All numbers in this section are estimates, not measurements. The full sweep
uses contexts 128/512/2048/4096; KV8/4/3; 29/31/33/35 GB/s sustained payload;
80/90/95% utilization; and 200/210/225/240 MHz. The raw theoretical bandwidth
is reported separately as 34.128 GB/s for dual x64 DDR4-2133 and 38.394 GB/s
for experimental dual x72 payload.

The summary below fixes 31 GB/s and 90% utilization.

| Model | weight bytes/token | body lanes at 225 MHz | ctx512 KV8 tok/s | ctx2048 KV4 tok/s | ctx4096 KV3 tok/s |
|---|---:|---:|---:|---:|---:|
| BitNet 0.7B | 163,770,638 | 302.0 | 137.86 | 115.07 | 98.47 |
| Bonsai 1.7B | 242,109,632 | 626.3 | 102.58 | 92.17 | 83.56 |
| Bonsai 4B | 565,928,432 | 1,614.8 | 46.17 | 43.33 | 40.79 |
| Bonsai 8B | 1,064,726,912 | 3,087.0 | 25.29 | 24.42 | 23.59 |

BitNet fits both the projected memory and 640/672-lane compute envelopes with
the most margin. Bonsai 1.7B needs about 626 body lanes at 225 MHz and reaches
102.58 tok/s at context 512 only when KV8 is used. At context 2048, even KV3
is 96.87 tok/s, while the measured KV3 quality codec fails. Bonsai 4B and 8B
miss 100 tok/s from weight traffic and body compute alone and should not be the
first Rev E target. Host LM-head-offload flags for every sweep point are in
`hardware_roof.json`.

## Why the decision remains open

- Quality retention: Bonsai A8-g128+KV8 is clean in smoke; BitNet's exact
  useful-quality floor remains unknown. Cross-tokenizer PPL cannot resolve it.
- Traffic and 100 tok/s: BitNet wins clearly. Bonsai 1.7B only has a short-
  context, high-utilization path with the quality-preserving KV8 codec.
- Datapath complexity: BitNet requires ternary decode and a larger KV-head
  footprint; Bonsai has simpler binary dot products but group scales and a
  much larger LM head.
- Verification: both have pinned CPU references, but only Bonsai has an
  official deployment-packed artifact/runtime. The BitNet target would first
  require a canonical packed-image generator.
- Expansion: 4B/8B do not fit this card's plain-decode 100 tok/s target, so
  choosing Bonsai 1.7B does not automatically make those sizes feasible.
- Bankai: the accumulator identity is proven, but behavioral usefulness is not
  evaluated and has no weight in this target choice.

The decisive long run should therefore precede RTL selection. If BitNet
row16+KV4 meets the agreed byte-normalized quality/task floor, choose conclusion
1 because it has the only robust long-context throughput margin. If it misses
and Bonsai's quality margin is material, choose conclusion 2 with context 512
and KV8 as an explicit first-profile limitation.
