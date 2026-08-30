# CPU model-selection smoke summary

All measurements use five fixed prompts and 256 predicted PPL tokens. The PPL
corpus is a repeated engineering text and is not publication-grade. Ratios are
within one model/backend only.

## BitNet 0.7B reproduction

Baseline PPL: 819.2822. Peak RSS: 5,135,036,416 bytes. Wall time: 244.90 s.
The baseline and six pre-existing variant logits hashes were identical across
two final deterministic runs.

| Variant | PPL ratio | top-1 | cosine | KL | exact generation |
|---|---:|---:|---:|---:|---:|
| KV8 | 1.0041 | 1.00 | 0.999925 | 0.000934 | 0.60 |
| KV4 | 1.1535 | 1.00 | 0.999621 | 0.003162 | 0.80 |
| KV3 | 1.1938 | 1.00 | 0.998442 | 0.014823 | 0.80 |
| row16 head | 1.0399 | 0.80 | 0.999300 | 0.009175 | 0.40 |
| row16 + KV4 | 1.1278 | 0.80 | 0.998994 | 0.013115 | 0.40 |
| row16 + KV3 | 0.9947 | 1.00 | 0.997537 | 0.017318 | 0.60 |
| block16x16 head | 1.0034 | 1.00 | 0.998204 | 0.015942 | 0.40 |

The apparent row16+KV3 PPL improvement is sampling noise, not an improvement
claim. All measured KV clipping/saturation rates were zero.

## Bonsai 1.7B

The unpacked BF16 reference baseline PPL is 141.4514. Peak RSS is
7,245,250,560 bytes and wall time is 99.14 s.

| Variant | PPL ratio | top-1 | cosine | KL | exact generation |
|---|---:|---:|---:|---:|---:|
| A12 per-token | 1.0029 | 1.00 | 0.999957 | 0.000241 | 1.00 |
| A10 per-token | 0.9939 | 1.00 | 0.999923 | 0.000528 | 1.00 |
| A8 per-token | 1.0095 | 1.00 | 0.999466 | 0.001150 | 0.60 |
| A8 g128 | 1.0034 | 1.00 | 0.999868 | 0.001444 | 1.00 |
| KV8 | 1.0096 | 0.80 | 0.999711 | 0.002422 | 0.80 |
| KV4 | 48.2688 | 0.00 | 0.647612 | 6.425608 | 0.00 |
| KV3 | 87.6859 | 0.00 | 0.586917 | 9.390449 | 0.00 |

KV4 and KV3 are catastrophic under the frozen per-token/per-head codec. A
cache-prefix regression test rules out accidental repeated quantization.
Changing the codec is future work, not a reinterpretation of these results.

Across 216 representative attention/MLP boundary cases, INT32/INT24/INT20 had
zero accumulator saturations. Mean relative RMSE with FP scale was 0.001947
(A12), 0.005324 (A10), 0.020312 (A8), and 0.008343 (A8 g128). The maximum
within-group magnitude spread in the unpacked BF16 weights was 0.002201.

The pinned official Q1_0 runtime independently passed native loading and
generation. Its repeated-text smoke PPL was 107.0377 +/- 25.4871 and mean
generation rate was 208.53 tok/s on the host. Its temporary 76,445,764-byte
full-logits file hashed to
`57e00d9ca99e5ace7afc2451de1f890359f8efeda1e62c0d1ab98e3fa7badaf1`.

## Catapult assumption-based roof

The table fixes sustained bandwidth at 31 GB/s and payload utilization at 90%.
It is not a board measurement.

| Model | streamed weight MB/token | lanes at 225 MHz for 100 tok/s | ctx 512 KV8 tok/s | ctx 2048 KV4 tok/s | ctx 4096 KV3 tok/s |
|---|---:|---:|---:|---:|---:|
| BitNet 0.7B | 163.77 | 302.0 | 137.86 | 115.07 | 98.47 |
| Bonsai 1.7B | 242.11 | 626.3 | 102.58 | 92.17 | 83.56 |
| Bonsai 4B | 565.93 | 1,614.8 | 46.17 | 43.33 | 40.79 |
| Bonsai 8B | 1,064.73 | 3,087.0 | 25.29 | 24.42 | 23.59 |

At the frozen roof, 4B/8B are excluded as first 100 tok/s targets. BitNet has
substantial margin; Bonsai 1.7B fits 672 body lanes only narrowly and reaches
100 tok/s at context 512 with KV8, not at context 2048.

## Decision

Decision 3: do not choose yet. The smoke data simultaneously favors BitNet's
hardware margin and Bonsai's stronger native model behavior, but it cannot
compare absolute quality across tokenizers or validate long-context behavior.

The one decisive next experiment is a 4096-predicted-token, non-repeated,
byte-normalized head-to-head quality run of the exact hardware profiles:
BitNet row16+KV4 and Bonsai A8-g128+KV8, with the same prompt/task subset and
contexts 512 and 2048. Report bits per UTF-8 byte, task accuracy, within-model
quality loss, and measured maximum RSS. That single run resolves whether the
Bonsai quality advantage is worth its narrow 100 tok/s envelope.
