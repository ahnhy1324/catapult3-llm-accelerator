#!/usr/bin/env python3
"""Catapult3 Rev E memory and compute roof projection.

All output values carry an evidence class: model-file-derived geometry,
calculation from a frozen numerical contract, or an explicit Catapult
assumption.  No row in this report is a board measurement.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONTEXTS = (128, 512, 2048, 4096)
SUSTAINED_BANDWIDTH_GBPS = (29, 31, 33, 35)
UTILIZATIONS = (0.80, 0.90, 0.95)
CLOCKS_MHZ = (200, 210, 225, 240)
KV_BITS = (8, 4, 3)
TARGET_TOKENS_PER_SECOND = 100.0
CATAPULT_LANES = (640, 672)


@dataclass(frozen=True)
class ModelProfile:
    model_key: str
    model_id: str
    revision: str
    family: str
    weight_format: str
    hidden_size: int
    intermediate_size: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    tied_embeddings: bool
    official_packed_file_bytes: int
    body_weight_count: int
    embedding_weight_count: int
    lm_head_weight_count: int
    other_weight_count: int
    matrix_count: int
    geometry_source: str


def _linear_counts(
    hidden: int,
    intermediate: int,
    layers: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[int, int]:
    query_width = query_heads * head_dim
    kv_width = kv_heads * head_dim
    attention = hidden * query_width + 2 * hidden * kv_width + query_width * hidden
    mlp = 3 * hidden * intermediate
    return layers * (attention + mlp), layers * 7


def _profiles() -> list[ModelProfile]:
    bitnet_body, bitnet_matrices = _linear_counts(1536, 4096, 24, 16, 16, 96)
    bonsai17_body, bonsai17_matrices = _linear_counts(2048, 6144, 28, 16, 8, 128)
    bonsai4_body, bonsai4_matrices = _linear_counts(2560, 9728, 36, 32, 8, 128)
    bonsai8_body, bonsai8_matrices = _linear_counts(4096, 12288, 36, 32, 8, 128)
    return [
        ModelProfile(
            model_key="bitnet-0.7b",
            model_id="1bitLLM/bitnet_b1_58-large",
            revision="85d047191dcb224f0e04f20d26110caaf8dc1a47",
            family="ternary_bitnet_reproduction",
            weight_format="body_5_trits_per_byte_plus_row16_fp4_head",
            hidden_size=1536,
            intermediate_size=4096,
            layers=24,
            query_heads=16,
            kv_heads=16,
            head_dim=96,
            vocab_size=32002,
            tied_embeddings=True,
            official_packed_file_bytes=0,
            body_weight_count=bitnet_body,
            embedding_weight_count=32002 * 1536,
            lm_head_weight_count=32002 * 1536,
            other_weight_count=(3 * 24 + 1) * 1536,
            matrix_count=bitnet_matrices,
            geometry_source="checkpoint_config_calculation",
        ),
        ModelProfile(
            model_key="bonsai-1.7b",
            model_id="prism-ml/Bonsai-1.7B-gguf",
            revision="210a9e99f79cb184909d49595906526eb2b3dd9a",
            family="bonsai_binary_qwen3",
            weight_format="Q1_0_g128",
            hidden_size=2048,
            intermediate_size=6144,
            layers=28,
            query_heads=16,
            kv_heads=8,
            head_dim=128,
            vocab_size=151669,
            tied_embeddings=True,
            official_packed_file_bytes=248_302_272,
            body_weight_count=bonsai17_body,
            embedding_weight_count=151669 * 2048,
            lm_head_weight_count=151669 * 2048,
            other_weight_count=(2 * 28 + 1) * 2048 + 2 * 28 * 128,
            matrix_count=bonsai17_matrices,
            geometry_source="official_config_and_gguf_metadata_calculation",
        ),
        ModelProfile(
            model_key="bonsai-4b",
            model_id="prism-ml/Bonsai-4B-gguf",
            revision="78f2c2bacd0904ffaba24b4873ed975e5818354a",
            family="bonsai_binary_qwen3",
            weight_format="Q1_0_g128",
            hidden_size=2560,
            intermediate_size=9728,
            layers=36,
            query_heads=32,
            kv_heads=8,
            head_dim=128,
            vocab_size=151669,
            tied_embeddings=True,
            official_packed_file_bytes=572_270_624,
            body_weight_count=bonsai4_body,
            embedding_weight_count=151669 * 2560,
            lm_head_weight_count=151669 * 2560,
            other_weight_count=(2 * 36 + 1) * 2560 + 2 * 36 * 128,
            matrix_count=bonsai4_matrices,
            geometry_source="official_config_and_gguf_metadata_calculation",
        ),
        ModelProfile(
            model_key="bonsai-8b",
            model_id="prism-ml/Bonsai-8B-gguf",
            revision="48516770dd04643643e9f9019a2a349cf26c5dbd",
            family="bonsai_binary_qwen3",
            weight_format="Q1_0_g128",
            hidden_size=4096,
            intermediate_size=12288,
            layers=36,
            query_heads=32,
            kv_heads=8,
            head_dim=128,
            vocab_size=151669,
            tied_embeddings=False,
            official_packed_file_bytes=1_158_654_496,
            body_weight_count=bonsai8_body,
            embedding_weight_count=151669 * 4096,
            lm_head_weight_count=151669 * 4096,
            other_weight_count=(2 * 36 + 1) * 4096 + 2 * 36 * 128,
            matrix_count=bonsai8_matrices,
            geometry_source="official_config_and_gguf_metadata_calculation",
        ),
    ]


def _q1_component(weight_count: int) -> tuple[int, int]:
    return math.ceil(weight_count / 8), math.ceil(weight_count / 128) * 2


def _row16_component(weight_count: int) -> tuple[int, int]:
    return math.ceil(weight_count * 4 / 8), math.ceil(weight_count / 16) + 4


def weight_traffic(profile: ModelProfile) -> dict[str, int]:
    if profile.family.startswith("ternary_bitnet"):
        body_weight = math.ceil(profile.body_weight_count / 5)
        body_scale = profile.matrix_count * 2
        head_weight, head_scale = _row16_component(profile.lm_head_weight_count)
        embedding_weight = math.ceil(profile.hidden_size * 4 / 8)
        embedding_scale = math.ceil(profile.hidden_size / 16)
    else:
        body_weight, body_scale = _q1_component(profile.body_weight_count)
        head_weight, head_scale = _q1_component(profile.lm_head_weight_count)
        embedding_weight = math.ceil(profile.hidden_size / 8)
        embedding_scale = math.ceil(profile.hidden_size / 128) * 2
    other = profile.other_weight_count * 2
    return {
        "body_weight_bytes_per_token": body_weight,
        "body_scale_metadata_bytes_per_token": body_scale,
        "embedding_row_bytes_per_token": embedding_weight,
        "embedding_scale_metadata_bytes_per_token": embedding_scale,
        "lm_head_bytes_per_token": head_weight,
        "lm_head_scale_metadata_bytes_per_token": head_scale,
        "other_weight_bytes_per_token": other,
    }


def kv_traffic(profile: ModelProfile, context: int, bits: int) -> dict[str, int]:
    values_per_stored_token = 2 * profile.layers * profile.kv_heads * profile.head_dim
    data_bytes_per_stored_token = math.ceil(values_per_stored_token * bits / 8)
    scale_bytes_per_stored_token = 2 * profile.layers * profile.kv_heads * 2
    return {
        "kv_values_per_stored_token": values_per_stored_token,
        "kv_data_read_bytes_per_token": data_bytes_per_stored_token * context,
        "kv_data_write_bytes_per_token": data_bytes_per_stored_token,
        "kv_scale_read_bytes_per_token": scale_bytes_per_stored_token * context,
        "kv_scale_write_bytes_per_token": scale_bytes_per_stored_token,
    }


def project_profile(profile: ModelProfile) -> dict[str, Any]:
    weights = weight_traffic(profile)
    fixed_weight_bytes = sum(weights.values())
    body_and_other = (
        weights["body_weight_bytes_per_token"]
        + weights["body_scale_metadata_bytes_per_token"]
        + weights["embedding_row_bytes_per_token"]
        + weights["embedding_scale_metadata_bytes_per_token"]
        + weights["other_weight_bytes_per_token"]
    )
    lane_requirements = {
        str(clock): profile.body_weight_count * TARGET_TOKENS_PER_SECOND / (clock * 1e6)
        for clock in CLOCKS_MHZ
    }
    rows: list[dict[str, Any]] = []
    for context in CONTEXTS:
        for kv_bits in KV_BITS:
            kv = kv_traffic(profile, context, kv_bits)
            kv_bytes = sum(value for key, value in kv.items() if key.endswith("bytes_per_token"))
            total = fixed_weight_bytes + kv_bytes
            without_head = body_and_other + kv_bytes
            for bandwidth in SUSTAINED_BANDWIDTH_GBPS:
                raw_tps = bandwidth * 1e9 / total
                raw_without_head = bandwidth * 1e9 / without_head
                for utilization in UTILIZATIONS:
                    achieved = raw_tps * utilization
                    achieved_without_head = raw_without_head * utilization
                    rows.append(
                        {
                            "context": context,
                            "kv_bits": kv_bits,
                            "sustained_bandwidth_GBps": bandwidth,
                            "bandwidth_utilization": utilization,
                            "weight_bytes_per_token": fixed_weight_bytes,
                            **kv,
                            "kv_total_bytes_per_token": kv_bytes,
                            "total_external_bytes_per_token": total,
                            "memory_only_tokens_per_second": raw_tps,
                            "utilized_tokens_per_second": achieved,
                            "host_lm_head_tokens_per_second": achieved_without_head,
                            "plain_decode_100_tps_memory_feasible": achieved >= TARGET_TOKENS_PER_SECOND,
                            "host_lm_head_offload_required_for_100_tps": (
                                achieved < TARGET_TOKENS_PER_SECOND
                                and achieved_without_head >= TARGET_TOKENS_PER_SECOND
                            ),
                            "evidence_class": "CATAPULT_ASSUMPTION_BASED_ESTIMATE",
                        }
                    )
    return {
        "profile": asdict(profile),
        "weight_traffic": weights,
        "packed_model_size": {
            "official_file_bytes": profile.official_packed_file_bytes or None,
            "calculated_streamed_weight_bytes_per_token": fixed_weight_bytes,
            "note": "The BitNet source checkpoint is not deployment-packed; its value is calculated. GGUF file size includes tokenizer/container bytes and, for untied models, the full input embedding that decode reads only by row.",
        },
        "compute": {
            "target_tokens_per_second": TARGET_TOKENS_PER_SECOND,
            "body_weight_operations_per_token": profile.body_weight_count,
            "lane_requirement_by_clock_MHz": lane_requirements,
            "candidate_lane_counts": list(CATAPULT_LANES),
            "fits_672_lanes_by_clock_MHz": {
                clock: requirement <= max(CATAPULT_LANES)
                for clock, requirement in lane_requirements.items()
            },
            "evidence_class": "CATAPULT_ASSUMPTION_BASED_ESTIMATE",
        },
        "rows": rows,
    }


def build_report() -> dict[str, Any]:
    theoretical = {
        "dual_x64_DDR4_2133_GBps": 2 * 8 * 2.133,
        "dual_x72_DDR4_2133_GBps": 2 * 9 * 2.133,
    }
    return {
        "schema_version": "catapult3-hardware-roof-v1",
        "measurement_status": "NO_BOARD_MEASUREMENTS_IN_THIS_REPORT",
        "assumptions": {
            "contexts": list(CONTEXTS),
            "kv_bits": list(KV_BITS),
            "sustained_bandwidth_GBps": list(SUSTAINED_BANDWIDTH_GBPS),
            "utilizations": list(UTILIZATIONS),
            "compute_clocks_MHz": list(CLOCKS_MHZ),
            "theoretical_bandwidth_GBps": theoretical,
            "kv_scale": "one FP16 scale per layer, KV head, token, and K/V tensor",
            "other_weights": "norm weights projected as FP16",
        },
        "models": [project_profile(profile) for profile in _profiles()],
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    report = build_report()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
