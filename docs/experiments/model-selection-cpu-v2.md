# Catapult3 Rev E model selection: CPU and RTL v2

Date: 2026-08-31

Base branch/commit: `work/model-selection-cpu-v1` at
`930502ed120becc0eb19ef165feadeba1f6a1738`

Work branch: `work/model-selection-cpu-v2`

## Decision

| Role | Selected model | Reason | Expected fully-on-card tok/s | Main risk |
|---|---|---|---:|---|
| First FPGA bring-up | BitNet 0.7B reproduction | Smallest e2e projection and LM head; only candidate with projected 100 tok/s margin at context 2,048 | 152.1 at ctx512; 115.1 at ctx2048 | Weak checkpoint quality and full-system integration |
| Headline target | Bonsai 1.7B official Q1 g128, A8 g128 + KV8 | Coherent official packed model/runtime and measured joint-profile quality retention | 100.5 at ctx512; 77.2 at ctx2048 | Context-2,048 external bandwidth and binary scale throughput |

The first implementation order is therefore **A: BitNet 0.7B first**. It is a
bring-up and throughput checkpoint, not the public quality demo. The headline
model remains Bonsai 1.7B because the official Q1 artifact generates coherent
English and Korean and its A8 g128 + KV8 joint profile is stable. The headline
claim must initially be scoped to context 512; the fully-on-card context-2,048
roof is a no-go on dual-x64 DDR at the modeled payload rates.

All FPGA rates in this report are `PROJECTED_FPGA`, not board measurements.
The CPU generation rate is diagnostic host performance and is never used as
FPGA throughput evidence.

## What v2 corrected

V1 counted only Transformer-body projections. V2 includes the full LM head,
embedding row lookup, attention QK/AV, normalization/activation, group scales,
streaming top-k, KV reads/writes, and metadata. It also separates memory,
projection, attention, scale, vector, and top-k roofs instead of combining
them into one imaginary lane count.

| Checkpoint | Body major linear elements/token | LM-head elements/token | E2e linear elements/token | Ideal lanes for 100 tok/s at 225 MHz |
|---|---:|---:|---:|---:|
| BitNet 0.7B | 679,477,248 | 49,155,072 | 728,632,320 | 323.84 |
| Bonsai 1.7B | 1,409,286,144 | 310,618,112 | 1,719,904,256 | 764.40 |

These are `CALCULATED_FROM_CONFIG` lower bounds for the projection engine.
They do not include idle cycles caused by attention, scale, top-k, routing, or
memory scheduling.

## Reproducibility and health

The fail-closed manifest pins every consumed model, tokenizer, configuration,
remote-code, GGUF, runtime, and corpus file. A run stops before loading if any
byte count or SHA-256 differs. Safetensors headers are also checked for
duplicate names, invalid shapes, unsupported dtypes, and overlapping ranges.
Loaded models are checked for missing/unexpected/mismatched, nonfinite,
all-zero, duplicate, and abnormal-scale tensors. No all-zero whitelist was
needed for the executed checkpoints (`MEASURED_MODEL_FILE`, `MEASURED_CPU`).

| Artifact | Exact revision | Weight/deployment SHA-256 | License |
|---|---|---|---|
| BitNet 0.7B reproduction | `85d047191dcb224f0e04f20d26110caaf8dc1a47` | `100062646f1f85771ebe297c5e476642d171c2e0e916b2ed8d19dfbe201b4b52` | MIT |
| Bonsai 1.7B unpacked | `a7f720bd688d7563714f3118edd97b83d06f0615` | `cf9a24cbd02e6e257bcfd3177475aaca7f8bd1a63a745441f30d3e40f4313a6b` | Apache-2.0 |
| Bonsai 1.7B Q1 GGUF | `210a9e99f79cb184909d49595906526eb2b3dd9a` | `3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3` | Apache-2.0 |
| PrismML runtime | `e311ed38fe7ab8fb577a5435b049d48b7d040923` | archive `c87e4ae315d17b8ef9695001db7ad0f9eb8ab275c33d11c02395c64d844fe764` | MIT |
| Falcon3 1B Instruct 1.58-bit | `72fd3f95fcd82639c902304919629edda8c6f2b4` | `3d536c681c4722e5263d413d85d243a1302217a245e4b55de93ece8a6ef30b15` | TII Falcon License 2.0 |

The Transformers runtime is pinned at commit
`096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4`. Large model files, caches,
native binaries, full-logit dumps, and Quartus databases remain outside Git.

## Fixed-byte evaluation contract

The evaluated corpus is exactly 24,576 UTF-8 bytes with SHA-256
`c21856723534065f53feb61320d4276722b70680b61834e2f2abeb15ae572f6b`.
It is derived from `Salesforce/wikitext` revision
`b08601e04326c79dfdd32d625aee71d232d685c3` by strict UTF-8 decode, removal
of one leading BOM, CRLF/CR to LF normalization, row joining with LF, and a
codepoint-safe first-24,576-byte cut. BitNet produces 6,262 tokens and Bonsai
5,542, so both exceed the requested 4,096 scored tokens.

Each 512- or 2,048-token evaluation window keeps an overlapping prefix but
scores each target token exactly once. The backend resets only at a window
boundary; it does not present a reset-only score as continuous context. One
EOS token is an unscored warm-up prefix when BOS is unavailable, and no EOS is
appended. BPB is the cross-tokenizer metric; PPL ratios are only compared
within one model/backend.

## CPU results

### Bonsai official Q1 runtime

The pinned native runtime loaded the official 248,302,272-byte Q1_0 GGUF,
produced finite logits with SHA-256
`57e00d9ca99e5ace7afc2451de1f890359f8efeda1e62c0d1ab98e3fa7badaf1`,
and generated coherent English and Korean on all eight prompt categories. It
averaged 202.39 host CPU tok/s, used 658,223,104 bytes peak RSS, and reported
107.038 PPL on the 256-token repeated smoke (`MEASURED_CPU`). The native CLI
does not expose generated token IDs or prompt top-k logits; those fields are
not fabricated and remain available only in the unpacked Transformers path.

### Bonsai unpacked codec matrix

The unpacked model is a semantic boundary reference for activation, KV, and
accumulator experiments; the native Q1 GGUF remains the deployment truth.
Health checks passed with no missing, unexpected, nonfinite, zero,
mismatched, duplicate, or unsupported tensors. The final enriched run took
793.44 s and peaked at 8,111,529,984 bytes RSS (`MEASURED_CPU`).

| Variant | Smoke PPL ratio | Exact greedy agreement | Logit cosine | BPB ctx512 | BPB ctx2048 |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.0000 | 1.00 | 1.000000 | 1.084977 | 0.993415 |
| A8 g128 + KV8 | 1.0158 | 1.00 | 0.999438 | 1.083982 | 0.993713 |
| A10 g128 + KV8 | 1.0128 | 1.00 | 0.999517 | 1.084892 | 0.993928 |
| A8 g128 + K4/V6 | 46.8730 | 0.00 | 0.629 | 2.7936 | 2.7744 |
| A8 g128 + K4/V5 | 42.5950 | 0.00 | 0.623 | 2.7942 | 2.7657 |

A8 g128 + KV8 changes BPB by -0.000995 at context 512 and +0.000298 at
context 2,048. This is the selected headline codec. K4/V6 and K4/V5 are
decisive failures despite their better memory roofs, so they cannot support a
100 tok/s quality claim. A10 g128 + KV8 remains the activation fallback.

Representative layers 0/14/27 covered 216 activation, accumulator, scale,
rounding, and saturation cases. INT32, INT24, and INT20 did not saturate in
those sampled inputs. Unsigned UQ4.20 scale and Q12 group-output rounding were
numerically close to the FP-scale reference. This is a boundary microstudy,
not a full-model proof that INT20 is safe.

### BitNet 0.7B reproduction

The model and safetensors health checks passed. The final enriched run took
777.16 s and peaked at 7,118,209,024 bytes RSS. The 256-token repeated smoke baseline
PPL is 819.282; that absolute number is not compared with Bonsai's tokenizer.

| Variant | Smoke PPL ratio | Exact greedy agreement | Logit cosine | BPB ctx512 | BPB ctx2048 |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.0000 | 1.000 | 1.000000 | 1.147582 | 0.990702 |
| row16 head + KV4 | 1.1278 | 0.750 | 0.999039 | 1.158737 | 0.999080 |
| block16x16 head + KV4 | 0.9194 | 0.875 | 0.997964 | 1.156238 | 1.000128 |
| row16 head + KV3 | 0.9947 | 0.500 | 0.997263 | 1.190254 | 1.023833 |

The long fixed-byte result resolves the misleading short-smoke values below
one. Row16 + KV4 increases BPB by 0.011155/0.008378 at contexts 512/2,048;
block16x16 + KV4 increases it by 0.008656/0.009426. Neither is a measured
improvement. Row16 + KV4 is selected for first bring-up because its long-
context BPB and logit error are slightly better and its scale schedule is
simpler; block16x16 remains a credible head alternative. KV3 degrades BPB by
0.042672/0.033131 and is rejected.

Under the selected joint profiles, Bonsai A8 g128 + KV8 has lower BPB by
0.074755 at context 512 and 0.005368 at context 2,048. BitNet's baseline can
look competitive on the long WikiText span, but the eight prompts expose the
checkpoint's practical limitation: repeated factual text, prompt echo instead
of completing arithmetic/code/technical requests, and incoherent Korean.
Those generation-health failures are why it is bring-up-only.

The checkpoint is a public reproduction rather than a Microsoft release.
Even if its selected codec retains its own baseline, weak absolute generation
or prompt echo makes it unsuitable for the headline demo. It remains useful
as the smallest deterministic ternary e2e bring-up image.

### Limited Falcon3 1B Instruct 1.58-bit candidate

The official Transformers integration passed fail-closed file/header and
loaded-model health checks through eager CPU weight unpack. The bounded run
took 145.59 s, peaked at 3,102,273,536 bytes RSS, produced logits SHA-256
`97c1119cd76288d5c852a2b6045e124409ec4540c04c27b315666eb1ce1a9e06`,
and measured 133.915 PPL on the same 256-token repeated smoke. Its short
English factual, arithmetic, code, and FPGA completions were coherent. The
16-token Korean completions were incomplete and one contained a Unicode
replacement character, so multilingual quality is not established
(`MEASURED_CPU`).

The model was deliberately not expanded to the full fixed-byte codec matrix:
the geometry alone makes the first-card 100 tok/s goal impossible, and the
limited-candidate rule stops work at that blocker rather than spending the
main experiment budget on a no-go target.

The geometry is 1,132,462,080 body linear elements plus an untied FP16
268,435,456-element LM head. Fully on-card traffic is about 782.7 MB/token at
context 512, limiting the x64 31 GB/s, 90%-utilization memory roof to about
35.6 tok/s. It is therefore not a first 100 tok/s candidate regardless of its
bounded CPU smoke quality (`CALCULATED_FROM_CONFIG`, `PROJECTED_FPGA`).

## Fully-on-card Catapult roof

The table fixes dual-x64 payload, 31 GB/s selected sustained bandwidth, 90%
pipeline utilization, and 225 MHz. DDR payload assumptions remain explicit:
dual-x64 theoretical payload is 34.128 GB/s; dual-x72 experimental payload is
38.394 GB/s. A selected sustained rate above the theoretical payload is
rejected rather than silently used.

| Model/profile | Lanes | Context | Projection roof | Memory roof | Final roof | Class |
|---|---:|---:|---:|---:|---:|---|
| BitNet row16 + KV4, direct | 672 | 512 | 207.51 | 152.12 | 152.12 | comfortable |
| BitNet row16 + KV4, TL5 | 672 | 512 | 182.83 | 152.12 | 152.12 | comfortable |
| BitNet row16 + KV4, direct | 672 | 2,048 | 207.51 | 115.10 | 115.10 | borderline |
| BitNet row16 + KV4, TL5 | 672 | 2,048 | 182.83 | 115.10 | 115.10 | borderline |
| Bonsai A8 g128 + KV8 | 768 | 512 | 100.47 | 102.58 | 100.47 | borderline |
| Bonsai A8 g128 + KV8 | 768 | 2,048 | 100.47 | 77.19 | 77.19 | no-go |

Comfortable means every modeled roof is at least 120 tok/s, borderline means
the minimum is 100--120, and no-go is below 100. Host LM-head offload is
retained as a separate diagnostic mode and is not used in the headline
decision. The full 29/31/33/35 GB/s, 80/90/95%, x64/x72, context
128/512/2,048/4,096, 640--896-lane and 200--240 MHz scenario definitions are
reproducible from `hardware_roof.py`; the checked-in compact JSON retains the
decision rows and complete per-component traffic/count decomposition.

The best modeled x72 case (35 GB/s selected, 95%, 896 lanes, 240 MHz) lifts
Bonsai A8 g128 + KV8 to 122.25 tok/s at context 512 but only 91.99 at context
2,048. The K4/V6 codec could cross 100 in an aggressive x72 case, but its
measured quality collapse invalidates that path.

Bonsai 4B and 8B remain geometry-only long-term options. Even with 896 lanes
at 225 MHz, x64 31 GB/s and 90% utilization, their fully-on-card context-512
final roofs are only 46.17 and 25.29 tok/s respectively. No limited generation
run can change that first-card bandwidth/compute no-go.

The TL5 figures include the checked-in single-bank table builder rather than
assuming free lookup-table construction. At 672 lanes, 24 layers require 600
activation tiles and 146,400 non-overlapped build cycles/token. This lowers
the projection roof from 207.51 to 182.83 tok/s, although memory remains the
final bottleneck in the two selected rows. A future dual-bank implementation
may hide some build work, but that is not credited here.

## RTL microbench

Three II=1 kernels were built under `rtl/microbench/`: Bonsai binary g128 with
A8 activations and scale multiply, BitNet direct 5-trits/byte decode, and a
TL5 table path. NumPy golden tests cover random vectors, 640/672 lanes, all
ternary extrema, every one of 243 packed codes, RNE, saturation, and invalid
codes. Pytest currently reports 46 passed (`MEASURED_CPU`).

Quartus Pro 25.1 recognizes the exact `10AXF40GAE` marking as Arria 10 GX
`10AX115_JZ`, speed grade 2. The initial registered but single-stage binary
768-lane reduction compiled and routed at 25,666 ALM, 880 registers, 18 DSP,
and 67.64 MHz. It missed 225 MHz by 10.340 ns; the 15.267 ns worst path was
the reduction from `weight_sign_reg` to `value_pipe[0]`. This measured failure
caused the checked-in staged-reduction rewrite.

The analogous registered single-stage direct-ternary baseline was worse:
640 lanes used 86,134 ALM and 6,939 registers, routed at 1.07 MHz, and missed
225 MHz by 926.288 ns. Its 930.856 ns path traversed 1,289 logic levels from a
packed-weight register to the final value register. This is useful negative
evidence: threshold decode without an explicitly staged tree is not a viable
640-trit/cycle implementation.

A first staged binary rewrite reduced the 768-lane result to 12,020 ALM,
5,564 registers, and 6 DSP, and lifted routed Fmax to 144.55 MHz. It still
missed 225 MHz by 2.474 ns; the remaining 7.080 ns path was the 16-input
activation chunk from an input register to `chunk_reg`. The final checked-in
binary kernel therefore uses explicit pair/8/32/128 balanced stages and a
staged cross-group tree rather than relying on synthesis to rebalance a
procedural accumulator.

The final checked-in kernels were fitted at 225 MHz and their routed Fmax was
used to conservatively classify the requested 200/225/240 MHz points. Separate
fitter seeds were not run for each clock constraint. Every lane point closes
225 MHz; all direct-ternary and TL5 points also exceed 240 MHz. These are
clocked internal microkernel paths. Board pins and the future memory-system
top are intentionally not constrained by this standalone flow.

| Kernel | Useful weights/cycle | ALM | Registers | M20K | DSP | Routed Fmax MHz | 225 MHz slack ns | Peak route | 200/225/240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bonsai binary g128 | 640 | 7,947 | 11,497 | 0 | 5 | 235.02 | +0.189 | 12.8% | pass/pass/fail |
| Bonsai binary g128 | 768 | 9,546 | 13,770 | 0 | 6 | 233.70 | +0.165 | 13.2% | pass/pass/fail |
| Bonsai binary g128 | 896 | 11,090 | 15,997 | 0 | 7 | 230.20 | +0.100 | 14.8% | pass/pass/fail |
| BitNet direct ternary | 640 | 18,819 | 17,270 | 0 | 0 | 259.40 | +0.589 | 15.1% | pass/pass/pass |
| BitNet direct ternary | 672 | 18,656 | 16,292 | 0 | 0 | 288.35 | +0.976 | 13.8% | pass/pass/pass |
| BitNet TL5 | 640 | 19,063 | 32,085 | 128 | 0 | 278.63 | +0.855 | 21.2% | pass/pass/pass |
| BitNet TL5 | 672 | 20,037 | 33,869 | 135 | 0 | 264.48 | +0.663 | 20.5% | pass/pass/pass |

At the same 640 useful weights/cycle, direct ternary costs 2.37 times the ALM
of binary g128; TL5 costs 2.40 times plus 128 M20Ks. Binary additionally uses
one DSP scale pipeline per 128-weight group. At each model's selected e2e
rate, Bonsai binary 768 uses 9,546 ALMs and 6 DSPs, BitNet direct 672 uses
18,656 ALMs and no DSPs, and BitNet TL5 672 uses 20,037 ALMs plus 135 M20Ks.
Thus Bonsai's arithmetic is materially cheaper, but its much larger e2e
matrix count consumes the lane advantage and leaves only a 100.47 tok/s
projection roof at 225 MHz.

The first TL5 implementations missed timing because a binary address drove a
base-3 decode, five-term sum, and all RAM write ports in one cycle. The final
version uses per-bank base-3 odometers, prevents equivalent-state merging, and
registers the table value before RAM write. Its measured build latency is 244
cycles and is charged in the e2e roof. Resource/Fmax non-monotonicity between
640 and 672 lanes is ordinary single-seed place-and-route variance; it is not
interpreted as favorable scaling.

Questa FSE 2024.3 `vlog` compiles all three rewritten kernels and the
testbench with zero errors. On this host `vsim` then fails while loading any
optimized design with Windows error `0x80096010`, even though `vopt` succeeds
and executable signatures validate. Local functional simulation is therefore
`BLOCKED`; the manual GitHub workflow runs the same testbench with Icarus.
No Fmax or functional-pass value is invented for a blocked run.

## Bankai option

Bankai is not a model-selection prerequisite. Integer tests show that XORing
one complete binary weight row is bit-exact to negating that row's wide final
projection contribution only after group scale/rounding and before residual
addition, narrowing, or saturation. Group/final saturation and the signed
two's-complement minimum can break equivalence at other patch points.

Bonsai 1.7B needs a 573,440-bit (71,680-byte) dense row bitmap: 20,480
projection rows/layer times 28 layers. A double buffer is 143,360 bytes. A
token-boundary bank-select swap is one cycle and resident lookup need not
increase inference II, subject to integrated post-fit verification
(`CALCULATED_FROM_CONFIG`, `PROJECTED_FPGA`). No behavioral Bankai benefit is
claimed because none was reproduced.

## Largest remaining risk

The largest single risk is closing 225 MHz for the selected II=1 projection
datapath after it is integrated with real memory, attention, scales, and
top-k. A microkernel post-fit result is necessary but not sufficient. For the
headline Bonsai profile, sustained fully-on-card DDR traffic at context 512 is
the second hard constraint; context 2,048 already fails the modeled memory
roof.

## Reproduction commands

```powershell
C:\msv2env\Scripts\python.exe -m pytest experiments\model_selection\tests rtl\microbench\test_golden.py rtl\microbench\test_quartus_parser.py -q
C:\msv2env\Scripts\python.exe experiments\model_selection\hardware_roof.py --output results\model_selection_v2\hardware_roof.json --csv-output results\model_selection_v2\hardware_roof.csv --no-scenarios
C:\msv2env\Scripts\python.exe rtl\microbench\run_quartus_sweep.py --quartus E:\Altera\quartus\bin64\quartus_sh.exe --build-root C:\quartus-v2 --output results\model_selection_v2\rtl_sweep.json --only representative
```

`scripts/run_model_selection_v2.sh` contains the complete sequential model
commands. The workflow is manual because model downloads/runs do not belong on
every push. Only compact JSON/Markdown/log artifacts are uploaded.
