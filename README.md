# Catapult3 LLM Accelerator

Experimental low-bit LLM accelerator for Microsoft Catapult3 Rev E / Arria 10.

The project explores a configurable low-bit Transformer datapath rather than a single hardwired checkpoint. The working direction combines a W1.58-style ternary body, low-bit KV attention, optional FP4/NVFP4-like embedding/LM-head handling, aggressive streaming/fusion, same-sequence prefill reuse, and optional MTP verification.

## Current hardware assumptions

- Microsoft Catapult3 Rev E class card
- Arria 10 GX/GT 1150-class fabric
- Dual 72-bit DDR4 interfaces
- GoldenTop reference configuration: DDR4-2133, quarter-rate EMIF, ECC-on 64-bit useful payload per channel
- Experimental target: validate 72-bit payload with ECC disabled
- PCIe kept minimal in the first architecture/resource study

## Working architecture target

- ~640–672 useful ternary weight lanes per compute cycle
- ~210–225 MHz compute domain, decoupled from the ~266.7 MHz EMIF domain
- Deep elastic FIFOs / line buffers to hide DDR burst and refresh stalls
- Intermediate external-memory traffic as close to zero as practical
- GQA-group scheduling and operator overlap
- Streaming attention / online softmax investigation
- Chunk-streamed FFN and partial O-projection accumulation
- x2 then x4 same-sequence prefill reuse
- D=1 MTP as the first speculative mode
- Direct ternary datapath vs TL5/T-MAC-style lookup datapath A/B synthesis

## Planning performance targets

These are architecture targets, not measured claims:

- Plain ~1B single-user decode: ~125–145 tok/s depending on memory mode and LM-head placement
- Same-sequence x4 prefill: ~400–500 tok/s aggressive target
- MTP1 effective decode: ~250 tok/s self-contained, ~280 target, ~300 stretch under favorable conditions

All headline numbers must be replaced by measured cycle counts, post-fit Fmax, real sustained DDR bandwidth, KV cost, and MTP acceptance before being treated as results.

## Documentation and experiments

- [`TODO.md`](TODO.md) — full architecture / validation / synthesis checklist
- [`experiments/nvfp4_bitnet/`](experiments/nvfp4_bitnet/) — BitNet tied embedding/LM-head NVFP4 fake-quant quality demo
- Notion planning page: https://app.notion.com/p/3cb30fe6e4e68167bf3bde0a4bdb6b5d?pvs=204

## Project boundary

This repository is intentionally separate from `ternarycore`. Existing low-bit KV mathematical/bit-exact work may be reused conceptually, but Catapult3-specific RTL, memory architecture, simulation, and performance experiments belong here.
