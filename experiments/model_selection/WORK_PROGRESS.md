# Model-selection CPU v1 progress

## Mandatory resume rule

After every context compaction, read this file and the task request in full
before taking another action. Update this file after each material milestone,
decision, blocker, commit, or change to the next action.

## Objective

Build and run a reproducible CPU comparison of the first Catapult3 Rev E FPGA
target candidates without implementing or modifying RTL. Compare the public
BitNet 0.7B reproduction, official Bonsai 1.7B Q1_0 g128, and the 4B/8B Bonsai
geometry/traffic projections. Produce a single evidence-qualified result
schema, tests, reports, and a justified target decision.

## Immutable anchors

- Target repository: `ahnhy1324/catapult3-llm-accelerator`
- Main base: `886a61131bcc73be12f532b36e322f5c0743e3f5`
- Work branch: `work/model-selection-cpu-v1`
- BitNet 0.7B candidate: `1bitLLM/bitnet_b1_58-large`
  revision `85d047191dcb224f0e04f20d26110caaf8dc1a47`
- Bonsai 1.7B GGUF: `prism-ml/Bonsai-1.7B-gguf`
  revision `210a9e99f79cb184909d49595906526eb2b3dd9a`
- Bonsai 1.7B unpacked reference: `prism-ml/Bonsai-1.7B-unpacked`
  revision `a7f720bd688d7563714f3118edd97b83d06f0615`
- PrismML llama.cpp master observed at start:
  `5ea87ddad22541a37053c7ba92b02ec1923617c6`

## Completed

1. Read the complete attached task request.
2. Verified remote main and cloned the target repository into the workspace.
3. Read `README.md`, `TODO.md`, all files under `experiments/nvfp4_bitnet/`,
   and `docs/experiments/2026-08-30-bitnet-nvfp4-cpu.md` completely.
4. Confirmed there is no applicable `AGENTS.md` in the repository.
5. Created `work/model-selection-cpu-v1` from the clean immutable main base.
6. Verified the public model identities, licenses, revisions, file sizes, and
   LFS SHA-256 values through the Hugging Face API.
7. Confirmed the local host has 31.15 GiB RAM and sufficient disk. No Windows
   or WSL C/C++ toolchain is currently installed; this is a setup item for the
   official Bonsai llama.cpp CPU baseline, not yet a final blocker.
8. Implemented the common result schema, symmetric activation/KV quantizers,
   Q1_0 group-linear reference, fixed-scale/rounding candidates, Bankai XOR
   reference, and hardware roof model.
9. Added BitNet and Bonsai Transformers adapters. BitNet covers baseline,
   row16 head, KV8/KV4/KV3, and row16+KV4/KV3. Bonsai covers BF16 reference,
   A12/A10/A8/A8-g128, KV8/KV4/KV3, and representative INT32/24/20 plus
   group-scale rounding comparisons.
10. Added 13 unit/schema/roof tests; all pass. Generated the first complete
    `results/hardware_roof.json` sweep.
11. Attempted to install the official runtime build toolchain in WSL. Ubuntu
    package DNS resolution currently fails although Windows network access
    works. Continue with checkpoint experiments while resolving the native
    runtime through WSL DNS repair or official prebuilt artifacts.
12. The first BitNet execution exposed a fail-open adapter bug: the frozen 2024
    model converted the supplied quantizing `DynamicCache` into a new default
    cache, producing zero observed KV updates. That result is invalidated and
    must not be cited. The adapter now preserves the custom cache explicitly
    and fails closed when a requested KV variant observes zero updates.
13. Completed valid CPU smoke matrices for the BitNet 0.7B reproduction and
    the unpacked Bonsai 1.7B reference. The result JSON files pass the common
    schema; the unpacked Bonsai result is deliberately labelled `PARTIAL`
    until the official GGUF runtime result is added.
14. Downloaded the official PrismML Windows CPU release
    `prism-b10660-e311ed3`, verified its ZIP SHA-256 as
    `c87e4ae315d17b8ef9695001db7ad0f9eb8ab275c33d11c02395c64d844fe764`,
    and confirmed the executable reports commit
    `e311ed38fe7ab8fb577a5435b049d48b7d040923` (build 10660).
15. Executed the official Q1_0 native baseline: five fixed greedy prompts,
    256 scored tokens, finite PPL, native performance, and a byte hash of the
    temporary full-logits capture are in
    `results/bonsai_1_7b_native_smoke.json`.
16. Regenerated BitNet after the clipping-statistics fix. Baseline and all six
    pre-existing variant logits hashes matched the previous valid run exactly;
    KV clipping rates are zero. Added and measured the required retained
    block16x16 LM-head comparison as a seventh variant.
17. Added the cache-prefix single-quantization regression test, artifact and
    environment manifests, reproduction README, compact result summary, and
    final experiment report. Seventeen tests pass and all three model result
    JSON files validate against the common schema.

## Implementation decisions

- Keep per-model backends separate and normalize only their JSON output.
- Use the official PrismML Q1_0 GGUF runtime for the Bonsai native baseline.
- Use the official unpacked Bonsai reference for activation hooks and
  accumulator/scale experiments; compare backends with numeric metrics rather
  than demanding identical floating-point logits hashes.
- Quantize KV at the cache update boundary, after RoPE for K, using a custom
  dynamic cache so prior cache entries are quantized exactly once.
- Cross-model absolute perplexity is not directly comparable when tokenizers
  differ. Use within-model PPL ratios and clearly label any cross-model metric.
- Bankai XOR equivalence is claimed only at the biasless symmetric binary
  linear accumulator boundary, before nonlinearities or asymmetric clipping.
- Keep checkpoints, caches, build trees, and uncompressed generated captures
  outside Git or under ignored paths. Evaluate models sequentially.

## In progress

- Commit is complete. Push is blocked by the current task environment denying
  outbound connection to `github.com:443`; this is a network-access blocker,
  not a filesystem permission, repository permission, or authentication error.
  The GitHub connector confirmed repository push/admin permission but its
  blob-write action was rejected because the task approval policy is `never`.
  The browser fallback was also explicitly denied GitHub-page access, so no
  further workaround is permitted.

## Next actions

1. Enable outbound `github.com:443` access for this task, or approve the
   GitHub connector's blob/tree/commit/ref write operations.
2. Push `work/model-selection-cpu-v1` to `origin` and verify the remote commit.

## Decision

Conclusion 3: do not choose yet. Run one 4096-predicted-token, non-repeated,
byte-normalized head-to-head comparison of BitNet row16+KV4 and Bonsai
A8-g128+KV8 at contexts 512 and 2048. This is the single decisive experiment
needed before selecting the first RTL target.

## Forbidden scope

- No RTL, Vivado, Quartus project, bitstream, board, or production repository
  modifications.
- Do not edit or repack original checkpoints.
- Do not retain multiple loaded models or quantized copies concurrently.
