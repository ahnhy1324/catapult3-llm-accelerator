#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
artifact_root="${MODEL_SELECTION_ARTIFACT_ROOT:-../model-selection-artifacts}"
result_root="${MODEL_SELECTION_RESULT_ROOT:-results/model_selection_v2}"
manifest="experiments/model_selection/artifact_manifest_v2.json"
corpus="$artifact_root/corpus/fixed-byte-24576.txt"
mkdir -p "$result_root"

case "${1:-all-local}" in
  test)
    "$python_bin" -m pytest experiments/model_selection/tests -q
    ;;
  roof)
    "$python_bin" experiments/model_selection/hardware_roof.py \
      --output "$result_root/hardware_roof.json" \
      --csv-output "$result_root/hardware_roof.csv" \
      --no-scenarios
    ;;
  summary)
    "$python_bin" experiments/model_selection/build_summary.py \
      --results-dir "$result_root" \
      --output "$result_root/summary.json"
    ;;
  rtl-iverilog)
    "$python_bin" rtl/microbench/run_iverilog.py
    ;;
  rtl-quartus)
    "$python_bin" rtl/microbench/run_quartus_sweep.py \
      --build-root "$artifact_root/quartus-v2" \
      --output "$result_root/rtl_sweep.json" \
      --only "${QUARTUS_SWEEP:-representative}"
    ;;
  bitnet)
    "$python_bin" experiments/model_selection/run_bitnet.py \
      --checkpoint-dir "$artifact_root/bitnet-0.7b" \
      --manifest "$manifest" \
      --fixed-byte-corpus "$corpus" \
      --fixed-byte-contexts 512 2048 \
      --fixed-byte-variants row16_head_plus_KV4,block16x16_head_plus_KV4,row16_head_plus_KV3 \
      --output "$result_root/bitnet.json"
    ;;
  bonsai)
    "$python_bin" experiments/model_selection/run_bonsai.py \
      --checkpoint-dir "$artifact_root/bonsai-1.7b-unpacked" \
      --manifest "$manifest" \
      --fixed-byte-corpus "$corpus" \
      --fixed-byte-contexts 512 2048 \
      --fixed-byte-variants A8_g128_plus_KV8,A10_g128_plus_KV8,A8_g128_plus_K4_V6,A8_g128_plus_K4_V5 \
      --output "$result_root/bonsai_transformers.json"
    ;;
  native-bonsai)
    "$python_bin" experiments/model_selection/run_bonsai_native.py \
      --model "$artifact_root/bonsai-1.7b-gguf/Bonsai-1.7B-Q1_0.gguf" \
      --runtime-dir "$artifact_root/prism-llama-runtime/prism-b10660-e311ed3-win-cpu-x64" \
      --runtime-archive "$artifact_root/prism-llama-runtime/llama-prism-b10660-e311ed3-bin-win-cpu-x64.zip" \
      --manifest "$manifest" \
      --output "$result_root/bonsai_native.json"
    ;;
  falcon-limited)
    "$python_bin" experiments/model_selection/run_falcon.py \
      --checkpoint-dir "$artifact_root/falcon3-1b-instruct-1.58bit" \
      --manifest "$manifest" \
      --output "$result_root/falcon3-1b-instruct-limited.json"
    ;;
  all-cpu-transformers)
    # Deliberately sequential: never retain multiple checkpoints in RAM.
    "$0" bitnet
    "$0" bonsai
    "$0" falcon-limited
    ;;
  all-local)
    "$0" test
    "$0" roof
    "$0" summary
    "$0" rtl-iverilog
    ;;
  *)
    echo "usage: $0 {test|roof|summary|rtl-iverilog|rtl-quartus|bitnet|bonsai|native-bonsai|falcon-limited|all-cpu-transformers|all-local}" >&2
    exit 2
    ;;
esac
