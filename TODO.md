# Catapult3 Rev E — Low-bit LLM Accelerator TODO

Notion master TODO: https://app.notion.com/p/3cb30fe6e4e68167bf3bde0a4bdb6b5d?pvs=204

## Architecture target
- [ ] Keep the accelerator configurable for low-bit Transformer-family models rather than hardwiring one checkpoint.
- [ ] Primary practical target: ~1B MobileLLM-class model; BitNet 2B as compatibility/reference target.
- [ ] Ternary body baseline: W1.58-style `5 trits / byte`.
- [ ] Reuse the existing low-bit KV attention/IP mathematics where useful; 3-bit KV is the aggressive Catapult operating point.
- [ ] Keep embedding/LM-head precision independent from body precision; evaluate NVFP4-like microscaling and minimal host offload.
- [ ] Optimize single-user / batch=1 first; no multi-user batching requirement.

## DDR / EMIF
- [ ] Reproduce GoldenTop dual DDR4 EMIF and validate both channels at DDR4-2133.
- [ ] Measure sustained sequential read bandwidth per channel and combined.
- [ ] Confirm current x72 physical / ECC-on x64-payload behavior.
- [ ] Test `x72 + ECC off` as full 72-bit payload; characterize calibration, timing and sustained bandwidth.
- [ ] Treat DDR4-2133 as the guaranteed baseline; higher rates are experimental until measured on Rev E.
- [ ] Keep the measured ~266.667 MHz DDR reference clock fixed and test higher DDR memory rates through EMIF/PLL parameter changes rather than changing the board oscillator.
- [ ] Sweep DDR4-2133 → 2200/2266 → 2400, and optionally 2666 only if Quartus/EMIF generation permits it; the installed DRAM is rated for 2666 but FPGA/PHY/PCB margin is unknown.
- [ ] For every overclock point, require repeated power-on calibration, EMIF Toolkit margin checks where available, PRBS/walking-pattern stress, dual-channel simultaneous traffic, sustained bandwidth measurement, and hot-board retest.
- [ ] Record the exact PLL/refclk/memory-clock parameters and Quartus timing/calibration results for every passing/failing point.
- [ ] Schedule long sequential weight bursts and interleave KV without starving weight traffic.
- [ ] Check whether x64→x72 extra bandwidth is enough to hide 3-bit KV traffic at the target decode rate.
- [ ] Record real EMIF stall/burst traces for cycle-level modeling.

## Constant-rate streaming / buffering
- [ ] Decouple ~266.7 MHz EMIF domain from ~210–225 MHz compute domain with async/elastic FIFOs.
- [ ] Start around 640–672 useful ternary weight lanes/cycle and tune from measured bandwidth + post-fit Fmax.
- [ ] Make intermediate external-memory traffic essentially zero.
- [ ] Use distributed line buffers / elastic FIFOs between operators.
- [ ] Determine FIFO depth from realistic DDR stall traces and cycle-level simulation.
- [ ] Implement high/low watermark DDR arbitration between weight and KV traffic.
- [ ] Add counters for FIFO occupancy/underflow, DDR waits, GEMV idle, attention backpressure and stage utilization.
- [ ] Target PE memory-idle ≈ 0 after pipeline fill.
- [ ] Treat DDR as a burst-rate producer and the compute fabric as a constant-rate consumer.

## Fusion / overlap
- [ ] Fuse RMSNorm → activation quantization → projection where practical.
- [ ] Schedule QKV by GQA groups so completed heads can enter RoPE/KV/attention immediately.
- [ ] Overlap current K/V quantize+write with historical QK processing.
- [ ] Evaluate online-softmax / FlashAttention-style streaming; explicitly validate fixed-point rounding semantics.
- [ ] Begin O-proj partial accumulation per completed attention head.
- [ ] Stream FFN by chunk: gate/up → activation → down-proj partial accumulation, no full intermediate materialization.
- [ ] Accumulate next RMSNorm sumsq while residual/projection output is emitted.
- [ ] Prefetch next weight tile whenever attention/other operators leave DDR bandwidth slack.
- [ ] Precompute RoPE constants, addresses/descriptors and quantizer/LUT state ahead of use.
- [ ] Cycle-model all producer/consumer boundaries and identify any unavoidable barriers.

## Direct ternary vs LUT/TL
- [ ] Keep 5-trits/byte as baseline; lossless entropy/pattern compression produced only minor global gains.
- [ ] Preserve weight-compression result as a negative result / no-go for a complex global entropy codec.
- [ ] A/B synthesize direct ternary add/sub vs T-MAC/bitnet.cpp-inspired TL5.
- [ ] TL5 candidate: packed base-3 byte directly indexes one of `3^5 = 243` partial sums for five activations.
- [ ] Measure ALM, M20K/MLAB, Fmax, routing, table-build cost and ops/cycle.
- [ ] Evaluate replicated/banked LUTs, lookup-port pressure, and double-buffered table generation.
- [ ] In x2/x4 reuse modes, share the packed weight byte as the lookup address across multiple activation tables.
- [ ] Prefer LUT/TL only if it meaningfully reduces routing/ALM or enables more reuse pipes; decode DDR ceiling itself does not change.

## Weight compression negative-result notes
- [ ] Keep the measured global ternary distribution / entropy results with the project notes.
- [ ] Global entropy is already close to the 1.6 b/w base-3 packing limit; do not spend large fabric on a general entropy decoder without new evidence.
- [ ] Pattern/dictionary coding should remain off the critical roadmap unless a different model shows materially higher exact-pattern coverage.

## KV / attention
- [ ] Reuse existing KV-cache mathematical/bit-exact contract where possible.
- [ ] Keep K/V precision configurable; retain current K4/V5 paths as references while testing 3-bit mode.
- [ ] Stream KV decode directly into attention where possible.
- [ ] Test compressed-domain / centroid-domain dot products only if compatible with the established KV contract.
- [ ] Spend DSPs aggressively enough that QK/AV/scale/norm latency is hidden behind weight time.
- [ ] Explore 2-way head/group attention pipelines if DSP budget remains ample.
- [ ] Make historical KV readable once and broadcast to multiple verification queries in MTP/prefill modes where dependency rules permit.

## Embedding / LM head
- [ ] Test NVFP4-like E2M1 + block scale on tied embedding/LM-head weights.
- [ ] Implement coefficients using shift/add and reserve DSPs for block scale rather than general FP4 multiplies.
- [ ] Measure PPL/logit error on the chosen checkpoint.
- [ ] Compare on-card NVFP4 head vs minimal host hidden-vector offload vs higher-precision fallback.
- [ ] If offloaded, keep PCIe use minimal: hidden vector/result only; host handles tokenizer/sampler/head.
- [ ] Keep tied embedding row lookup cheap and avoid forcing the whole output matrix onto the critical on-card decode path if it lowers throughput.

## Prefill reuse
- [ ] Same-sequence token tiling only; no multi-user batching requirement.
- [ ] Implement x2 prefill first, then x4 if fit/timing allows.
- [ ] Broadcast one weight stream to multiple activation pipes.
- [ ] Use short reuse banks/line buffers rather than duplicating weight storage.
- [ ] Reuse the same physical multi-position datapath later for MTP verification.
- [ ] Aggressive planning target for ~1B model: ~400–500 tok/s prefill, subject to fit/board data.

## MTP / speculative mode
- [ ] Evaluate D=1 MTP first.
- [ ] Train/distill an MTP1 module for the selected low-bit model.
- [ ] Check whether quantized MTP1 weights can remain resident in M20K after metadata/buffering.
- [ ] Reuse x2 prefill datapath as 2-position target verifier.
- [ ] Broadcast target weights and historical KV across verification positions where dependencies permit.
- [ ] Measure actual second-token acceptance; do not assume 85–90%.
- [ ] Track accepted tokens per target pass and MTP overhead separately.
- [ ] Planning targets only: ~250 tok/s self-contained; ~280 target / ~300 stretch with favorable x72, Fmax, acceptance and minimal LM-head offload.
- [ ] Defer MTP2+ until MTP1 proves worthwhile.

## Optional model-side experiments
- [ ] Explore recursive/weight-shared Transformers separately; requires uptraining/retraining.
- [ ] Quantify external weight traffic saved when a shared physical layer can remain on-chip.
- [ ] Explore layer-specific low-rank adapters / relaxed recursion only as a separate research branch.
- [ ] Treat recursive models as optional model-side specialization, not a requirement for the baseline accelerator.

## Simulation / synthesis
- [ ] Build PCIe-free synthetic command shim for full compute/memory simulation and Arria 10 compilation.
- [ ] Use real dual EMIF IP/memory models where practical.
- [ ] Create block benchmarks: projection, attention, FFN, KV, LM head, TL5.
- [ ] Functional RTL sim for correctness/cycles + post-fit STA for Fmax.
- [ ] Full one-token smoke test after block-level model agrees.
- [ ] Sweep lanes (640/672), target clocks (200/210/220/225), activation pipes (x1/x2/x4), attention width, FIFO depth, TL5/direct ternary.
- [ ] Record ALM/register/M20K/DSP/Fmax/routing for every point.
- [ ] Build cycle model from cycles/pass, actual DDR BW, context, KV precision, MTP acceptance and LM-head placement.
- [ ] Add synthetic DDR profiles for conservative / expected / ideal sustained bandwidth before board measurement.
- [ ] Keep `TOTAL_CYCLES`, weight/KV beats, stalls, FIFO watermarks, stage busy/idle and backpressure counters visible in simulation.

## Resource / performance gates
- [ ] Optimize useful work per external-memory byte, not headline TOPS.
- [ ] Reject wider designs if routing/Fmax loss reduces actual tok/s.
- [ ] Allow aggressive ~75–82% ALM and ~75–85% M20K if >=~210 MHz still closes.
- [ ] Use otherwise-idle DSPs to hide attention/norm latency.
- [ ] Measure decode tok/s, prefill tok/s, J/token, tok/s/W, external GB/token and utilization.
- [ ] Compare fairly against CPUs/Apple Silicon and published FPGA work with model/precision/end-to-end scope stated.

## First-pass PCIe policy
- [ ] PCIe only for command/control, model loading and small result/hidden transfers in the initial resource study.
- [ ] Reserve shell resources but do not depend on host RAM as a third weight channel yet.
- [ ] Evaluate PCIe weight assist / host KV later only after on-card balance is measured.

## Go / no-go gates
- [ ] Dual-EMIF stable + sustained BW known.
- [ ] x72 non-ECC legality/stability known.
- [ ] Direct ternary vs TL5 synth comparison complete.
- [ ] FIFO underflow / PE memory-idle near zero under realistic memory timing.
- [ ] Aggressive configuration fits with acceptable routing and >=~210 MHz.
- [ ] KV integration does not create a new bottleneck.
- [ ] LM-head placement chosen from measured total throughput.
- [ ] MTP acceptance/speedup measured before claiming 250–300 tok/s.

## Working performance targets — planning only
- Plain ~1B single-user decode: ~125–145 tok/s depending on x64/x72 and LM-head placement.
- Same-sequence x4 prefill: ~400–500 tok/s aggressive target.
- MTP1 effective decode: ~250 tok/s self-contained, ~280 target, ~300 stretch under favorable conditions.

## Design principle
Treat Catapult3 as a configurable low-bit Transformer accelerator rather than a single-model hardwired engine: keep model geometry/configuration reasonably flexible while aggressively specializing expensive recurring operations such as ternary linear layers, low-bit KV attention, FP4-like head handling, streaming reuse, operator fusion, and MTP verification.
