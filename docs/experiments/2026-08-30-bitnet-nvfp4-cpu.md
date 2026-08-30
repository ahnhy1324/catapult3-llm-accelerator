# BitNet tied embedding / LM-head NVFP4 CPU experiment

Date: 2026-08-30

## Scope

This experiment measures the numerical quality impact of fake-quantizing only the physically tied input embedding / LM-head matrix of `microsoft/bitnet-b1.58-2B-4T`. The packed W1.58 transformer body is unchanged.

This is **not** a native NVFP4 performance benchmark. The quantized matrix is dequantized to BF16 and evaluated through the standard PyTorch/Transformers model path.

## Environment

- GitHub-hosted Ubuntu 24.04 runner
- 4 vCPU AMD EPYC 9V74, AVX2/AVX-512 BF16 exposed
- 15 GiB RAM, 3 GiB swap
- PyTorch 2.13.0+cpu
- Transformers 4.52.0.dev0 from commit `096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4`
- Model dtype: BF16
- Model: `microsoft/bitnet-b1.58-2B-4T`
- Tied matrix: `128256 x 2560` = 328,335,360 values
- Evaluation: three prompt last-token distributions plus 256 predicted WikiText-2 test tokens
- Workflow run: https://github.com/ahnhy1324/catapult3-llm-accelerator/actions/runs/33308934185
- Results artifact: https://github.com/ahnhy1324/catapult3-llm-accelerator/actions/runs/33308934185/artifacts/9731393116

## Quantization layouts

- `row16`: one E4M3-like scale for each row-wise group of 16 E2M1-like values.
- `block16x16`: one E4M3-like scale for each 16 x 16 group of E2M1-like values.
- A tensor-wide FP32 global scale is included in both layouts.

## Results

| Metric | BF16 baseline | row16 | block16x16 |
|---|---:|---:|---:|
| Effective bits / weight | 16 | 4.50000 | 4.03125 |
| Estimated packed tied-matrix size | 626.25 MiB | 176.13 MiB | 157.79 MiB |
| Size reduction vs BF16 | — | 71.875% | 74.805% |
| WikiText slice PPL | 29.1843 | 29.5790 | 30.3196 |
| PPL ratio vs BF16 | 1.0000x | 1.0135x | 1.0389x |
| Prompt top-1 match | — | 100% (3/3) | 100% (3/3) |
| Mean top-5 overlap | — | 100.0% | 93.3% |
| Mean top-10 overlap | — | 90.0% | 96.7% |
| Mean last-logit cosine | — | 0.997801 | 0.994636 |
| Mean KL(base || quantized) | — | 0.010666 | 0.072208 |
| Weight cosine | — | 0.995542 | 0.993303 |
| Weight normalized MSE | — | 0.008913 | 0.013370 |
| Weight saturation fraction | — | 3.299% | 0.234% |

The baseline mean NLL was 3.373632. `row16` increased it to 3.387063; `block16x16` increased it to 3.411795.

## Prompt-level last-token results

### row16

| Prompt | Top-1 | Top-5 overlap | Top-10 overlap | Logit cosine | KL |
|---|---:|---:|---:|---:|---:|
| France | match | 100% | 80% | 0.998827 | 0.008807 |
| Apples | match | 100% | 90% | 0.998509 | 0.012981 |
| FPGA continuation | match | 100% | 100% | 0.996068 | 0.010210 |

### block16x16

| Prompt | Top-1 | Top-5 overlap | Top-10 overlap | Logit cosine | KL |
|---|---:|---:|---:|---:|---:|
| France | match | 80% | 100% | 0.997607 | 0.023189 |
| Apples | match | 100% | 100% | 0.997960 | 0.166051 |
| FPGA continuation | match | 100% | 90% | 0.988340 | 0.027383 |

## Resource use and runtime

The complete compact run, including dependency-ready model download/load, WikiText preparation, baseline evaluation, two full 328M-value quantizations, and both quantized evaluations, took 79.51 seconds wall-clock. Peak RSS was 7,749,024 KiB (about 7.39 GiB), with no swap activity.

The `row16` quantization pass took 11.14 seconds and the `block16x16` pass took 10.57 seconds on the hosted CPU. These are offline conversion times and are not inference throughput measurements.

## Interpretation

`row16` is the stronger first FPGA format. It reduces the tied matrix from 626.25 MiB to 176.13 MiB while increasing this small-slice PPL by only 1.35%. Its prompt KL is about 6.8 times lower than `block16x16`, while costing only 18.34 MiB more storage.

`block16x16` is still viable as a maximum-bandwidth mode, but its 3.89% PPL increase and larger distribution shifts make it a secondary candidate unless calibration, QAT, or selective higher-precision rows recover the loss.

## Limitations

- The PPL sample is only 256 predicted tokens, so the ratio is directional rather than publication-grade.
- Only three prompt distributions were compared in this compact run.
- The experiment fake-quantizes and dequantizes the tied matrix; it does not reproduce every detail of a future FPGA datapath or native NVIDIA NVFP4 arithmetic.
- It measures quality, not speed. Native packed CPU kernels were not used for the FP4 head.
- Longer generation, larger PPL samples, downstream tasks, and row-sensitive mixed-precision experiments remain required before freezing the hardware format.

## Current decision

Proceed with `row16` as the baseline Catapult format for tied embedding / LM-head evaluation. Retain `block16x16` as an optional storage-optimized mode and test whether selective row promotion or lightweight calibration can close its quality gap.
