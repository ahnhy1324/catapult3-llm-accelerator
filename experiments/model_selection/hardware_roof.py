#!/usr/bin/env python3
"""End-to-end Catapult3 Rev E decode roof model.

Projection, attention, scale, vector, and top-k engines are independently
provisioned. FPGA values are projections, never board measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONTEXTS = (128, 512, 2048, 4096)
SUSTAINED_BANDWIDTH_GBPS = (29, 31, 33, 35)
UTILIZATIONS = (0.80, 0.90, 0.95)
CLOCKS_MHZ = (200, 210, 225, 240)
PROJECTION_LANES = (640, 672, 720, 768, 800, 896)
MEMORY_MODES = {"dual_x64_payload": 34.128, "dual_x72_experimental_payload": 38.394}
TARGET_TPS = 100.0
ATTENTION_ELEMENTS_PER_CYCLE = 256
SCALE_MULTIPLIES_PER_CYCLE = 8
VECTOR_ELEMENTS_PER_CYCLE = 128
TOPK_COMPARISONS_PER_CYCLE = 32
TOP_K = 10


@dataclass(frozen=True)
class ModelProfile:
    model_key: str
    model_id: str
    revision: str
    family: str
    hidden_size: int
    intermediate_size: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    tied_embeddings: bool
    body_weight_elements: int
    lm_head_weight_elements: int
    other_weight_elements: int
    body_matrix_count: int
    official_packed_file_bytes: int


@dataclass(frozen=True)
class Variant:
    key: str
    model_key: str
    activation: str
    head_format: str
    k_bits: int
    v_bits: int
    kv_scale_group: int
    datapaths: tuple[str, ...]


def _linear_counts(hidden: int, intermediate: int, layers: int, query_heads: int, kv_heads: int, head_dim: int) -> tuple[int, int]:
    query_width = query_heads * head_dim
    kv_width = kv_heads * head_dim
    attention = hidden * query_width + 2 * hidden * kv_width + query_width * hidden
    mlp = 3 * hidden * intermediate
    return layers * (attention + mlp), layers * 7


def profiles() -> list[ModelProfile]:
    definitions = [
        ("bitnet-0.7b", "1bitLLM/bitnet_b1_58-large", "85d047191dcb224f0e04f20d26110caaf8dc1a47", "ternary", 1536, 4096, 24, 16, 16, 96, 32002, True, 0),
        ("falcon3-1b-instruct-1.58bit", "tiiuae/Falcon3-1B-Instruct-1.58bit", "72fd3f95fcd82639c902304919629edda8c6f2b4", "ternary", 2048, 8192, 18, 8, 4, 256, 131072, False, 1_357_042_252),
        ("bonsai-1.7b", "prism-ml/Bonsai-1.7B-gguf", "210a9e99f79cb184909d49595906526eb2b3dd9a", "binary_q1_g128", 2048, 6144, 28, 16, 8, 128, 151669, True, 248_302_272),
        ("bonsai-4b", "prism-ml/Bonsai-4B-gguf", "78f2c2bacd0904ffaba24b4873ed975e5818354a", "binary_q1_g128", 2560, 9728, 36, 32, 8, 128, 151669, True, 572_270_624),
        ("bonsai-8b", "prism-ml/Bonsai-8B-gguf", "48516770dd04643643e9f9019a2a349cf26c5dbd", "binary_q1_g128", 4096, 12288, 36, 32, 8, 128, 151669, False, 1_158_654_496),
    ]
    result: list[ModelProfile] = []
    for key, model_id, revision, family, hidden, intermediate, layers, qh, kvh, hd, vocab, tied, packed in definitions:
        body, matrices = _linear_counts(hidden, intermediate, layers, qh, kvh, hd)
        result.append(ModelProfile(
            model_key=key, model_id=model_id, revision=revision, family=family,
            hidden_size=hidden, intermediate_size=intermediate, layers=layers,
            query_heads=qh, kv_heads=kvh, head_dim=hd, vocab_size=vocab,
            tied_embeddings=tied, body_weight_elements=body,
            lm_head_weight_elements=vocab * hidden,
            other_weight_elements=(2 * layers + 1) * hidden + 2 * layers * hd,
            body_matrix_count=matrices, official_packed_file_bytes=packed,
        ))
    return result


def variants() -> list[Variant]:
    return [
        Variant("bitnet-row16-kv4", "bitnet-0.7b", "native_int8", "row16_fp4", 4, 4, 128, ("direct_ternary", "TL5")),
        Variant("bitnet-block16-kv4", "bitnet-0.7b", "native_int8", "block16x16_fp4", 4, 4, 128, ("direct_ternary", "TL5")),
        Variant("bitnet-row16-kv3", "bitnet-0.7b", "native_int8", "row16_fp4", 3, 3, 128, ("direct_ternary", "TL5")),
        Variant("falcon3-1b-official-fp16-head-kv8", "falcon3-1b-instruct-1.58bit", "native_int8", "fp16", 8, 8, 128, ("direct_ternary", "TL5")),
        Variant("bonsai-a8g128-kv8", "bonsai-1.7b", "A8_g128", "q1_g128", 8, 8, 128, ("binary_g128",)),
        Variant("bonsai-a10g128-kv8", "bonsai-1.7b", "A10_g128", "q1_g128", 8, 8, 128, ("binary_g128",)),
        Variant("bonsai-a8g128-k4v6", "bonsai-1.7b", "A8_g128", "q1_g128", 4, 6, 128, ("binary_g128",)),
        Variant("bonsai-a8g128-k4v5", "bonsai-1.7b", "A8_g128", "q1_g128", 4, 5, 128, ("binary_g128",)),
        Variant("bonsai-4b-geometry-kv8", "bonsai-4b", "A8_g128_assumption", "q1_g128", 8, 8, 128, ("binary_g128",)),
        Variant("bonsai-8b-geometry-kv8", "bonsai-8b", "A8_g128_assumption", "q1_g128", 8, 8, 128, ("binary_g128",)),
    ]


def _q1_bytes(elements: int) -> tuple[int, int]:
    return math.ceil(elements / 8), math.ceil(elements / 128) * 2


def _fp4_bytes(elements: int, head_format: str) -> tuple[int, int]:
    data = math.ceil(elements * 4 / 8)
    if head_format == "row16_fp4":
        scales = math.ceil(elements / 16)
    elif head_format == "block16x16_fp4":
        scales = math.ceil(elements / 256)
    else:
        raise ValueError(head_format)
    return data, scales + 4


def memory_components(profile: ModelProfile, variant: Variant, context: int, head_placement: str) -> dict[str, int]:
    if profile.family == "ternary":
        body_weight = math.ceil(profile.body_weight_elements / 5)
        body_metadata = profile.body_matrix_count * 2
        if variant.head_format == "fp16":
            head_weight, head_metadata = profile.lm_head_weight_elements * 2, 0
            embedding_weight, embedding_metadata = profile.hidden_size * 2, 0
        else:
            head_weight, head_metadata = _fp4_bytes(profile.lm_head_weight_elements, variant.head_format)
            embedding_weight = math.ceil(profile.hidden_size * 4 / 8)
            embedding_metadata = math.ceil(profile.hidden_size / 16)
    else:
        body_weight, body_metadata = _q1_bytes(profile.body_weight_elements)
        head_weight, head_metadata = _q1_bytes(profile.lm_head_weight_elements)
        embedding_weight = math.ceil(profile.hidden_size / 8)
        embedding_metadata = math.ceil(profile.hidden_size / 128) * 2
    if head_placement == "minimal_host_lm_head_offload":
        head_weight = head_metadata = 0
    elif head_placement != "fully_on_card":
        raise ValueError(head_placement)
    values = profile.layers * profile.kv_heads * profile.head_dim
    k_data = math.ceil(values * variant.k_bits / 8)
    v_data = math.ceil(values * variant.v_bits / 8)
    groups_per_head = math.ceil(profile.head_dim / variant.kv_scale_group)
    scale_data = 2 * profile.layers * profile.kv_heads * groups_per_head * 2
    kv_read = (k_data + v_data + scale_data) * context
    kv_write = k_data + v_data + scale_data
    other = profile.other_weight_elements * 2
    activation_spill = 0
    total = sum((embedding_weight, embedding_metadata, body_weight, body_metadata, head_weight, head_metadata, kv_read, kv_write, activation_spill, other))
    return {
        "input_embedding_row_bytes_per_token": embedding_weight,
        "input_embedding_metadata_bytes_per_token": embedding_metadata,
        "body_weight_bytes_per_token": body_weight,
        "body_scale_metadata_bytes_per_token": body_metadata,
        "lm_head_bytes_per_token": head_weight,
        "lm_head_scale_metadata_bytes_per_token": head_metadata,
        "kv_read_bytes_per_token": kv_read,
        "kv_write_bytes_per_token": kv_write,
        "activation_spill_bytes_per_token": activation_spill,
        "other_metadata_bytes_per_token": other,
        "total_external_bytes_per_token": total,
        "minimal_host_pcie_bytes_per_token": (2 * profile.hidden_size + 8) if head_placement != "fully_on_card" else 0,
    }


def compute_components(profile: ModelProfile, variant: Variant, context: int, head_placement: str) -> dict[str, int]:
    head = profile.lm_head_weight_elements if head_placement == "fully_on_card" else 0
    projection = profile.body_weight_elements + head
    attention_qk = profile.layers * profile.query_heads * profile.head_dim * context
    norm = (2 * profile.layers + 1) * profile.hidden_size
    activation = 2 * profile.layers * profile.intermediate_size
    topk = profile.vocab_size * TOP_K
    if profile.family == "ternary":
        binary_groups = 0
        scale_multiplies = math.ceil(head / 16) if head and variant.head_format != "fp16" else 0
        ternary_decode = profile.body_weight_elements
    else:
        binary_groups = math.ceil(projection / 128)
        scale_multiplies = binary_groups
        ternary_decode = 0
    return {
        "body_projection_weight_elements_per_token": profile.body_weight_elements,
        "lm_head_weight_elements_per_token": head,
        "e2e_projection_weight_elements_per_token": projection,
        "binary_group_dot_groups_per_token": binary_groups,
        "binary_group_scale_multiplies_per_token": scale_multiplies,
        "ternary_decode_or_lookup_elements_per_token": ternary_decode,
        "qk_elements_per_token": attention_qk,
        "av_elements_per_token": attention_qk,
        "norm_elements_per_token": norm,
        "activation_elements_per_token": activation,
        "streaming_topk_comparisons_per_token": topk,
        "minimal_sampling_operations_per_token": 4 * TOP_K,
    }


def _roof(rate_per_second: float, work_per_token: int) -> float:
    return math.inf if work_per_token == 0 else rate_per_second / work_per_token


def _classification(value: float) -> str:
    if value >= 120.0:
        return "comfortable"
    if value >= 100.0:
        return "borderline"
    return "no-go"


def _projection_cycles_per_token(
    profile: ModelProfile,
    datapath: str,
    lanes: int,
    projection_elements: int,
) -> tuple[float, int]:
    """Return steady weight cycles plus any non-overlapped table-build cost.

    The checked-in TL5 kernel has one table bank.  For each input tile it
    writes all 243 entries through a one-cycle write pipeline before consuming matrix rows.  Six Transformer
    projections consume hidden-size inputs and the down projection consumes
    intermediate-size inputs.  The FP4/FP16 LM head does not use TL5.
    """
    weight_cycles = projection_elements / lanes
    if datapath != "TL5":
        return weight_cycles, 0
    activation_tiles_per_layer = (
        6 * math.ceil(profile.hidden_size / lanes)
        + math.ceil(profile.intermediate_size / lanes)
    )
    build_cycles = 244 * profile.layers * activation_tiles_per_layer
    return weight_cycles + build_cycles, build_cycles


def scenario_rows(profile: ModelProfile, variant: Variant) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context in CONTEXTS:
        for head_placement in ("fully_on_card", "minimal_host_lm_head_offload"):
            memory = memory_components(profile, variant, context, head_placement)
            compute = compute_components(profile, variant, context, head_placement)
            for datapath in variant.datapaths:
                for memory_mode, theoretical_gbps in MEMORY_MODES.items():
                    for bandwidth in SUSTAINED_BANDWIDTH_GBPS:
                        valid_bandwidth = bandwidth <= theoretical_gbps + 1e-12
                        for utilization in UTILIZATIONS:
                            memory_roof = bandwidth * utilization * 1e9 / memory["total_external_bytes_per_token"]
                            for clock in CLOCKS_MHZ:
                                clock_hz = clock * 1e6
                                for lanes in PROJECTION_LANES:
                                    projection_cycles, tl5_build_cycles = _projection_cycles_per_token(
                                        profile,
                                        datapath,
                                        lanes,
                                        compute["e2e_projection_weight_elements_per_token"],
                                    )
                                    roofs = {
                                        "memory_roof_tps": memory_roof,
                                        "projection_compute_roof_tps": clock_hz / projection_cycles,
                                        "attention_compute_roof_tps": _roof(ATTENTION_ELEMENTS_PER_CYCLE * clock_hz, compute["qk_elements_per_token"] + compute["av_elements_per_token"]),
                                        "scale_multiply_roof_tps": _roof(SCALE_MULTIPLIES_PER_CYCLE * clock_hz, compute["binary_group_scale_multiplies_per_token"]),
                                        "vector_compute_roof_tps": _roof(VECTOR_ELEMENTS_PER_CYCLE * clock_hz, compute["norm_elements_per_token"] + compute["activation_elements_per_token"]),
                                        "topk_roof_tps": _roof(TOPK_COMPARISONS_PER_CYCLE * clock_hz, compute["streaming_topk_comparisons_per_token"]),
                                    }
                                    final = min(roofs.values()) if valid_bandwidth else 0.0
                                    bottleneck = min(roofs, key=roofs.get) if valid_bandwidth else "invalid_memory_payload_scenario"
                                    rows.append({
                                        "model_key": profile.model_key, "variant": variant.key,
                                        "datapath": datapath, "head_placement": head_placement,
                                        "context": context, "memory_mode": memory_mode,
                                        "theoretical_payload_GBps": theoretical_gbps,
                                        "selected_sustained_GBps": bandwidth,
                                        "pipeline_utilization": utilization,
                                        "bandwidth_scenario_valid": valid_bandwidth,
                                        "projection_lanes": lanes, "clock_MHz": clock, **roofs,
                                        "projection_weight_cycles_per_token": compute["e2e_projection_weight_elements_per_token"] / lanes,
                                        "tl5_table_build_cycles_per_token": tl5_build_cycles,
                                        "projection_total_cycles_per_token": projection_cycles,
                                        "final_bottleneck_roof_tps": final,
                                        "final_bottleneck": bottleneck,
                                        "classification": _classification(final),
                                        "evidence_tag": "PROJECTED_FPGA" if valid_bandwidth else "BLOCKED",
                                    })
    return rows


def regression_values(profile: ModelProfile) -> dict[str, Any]:
    e2e = profile.body_weight_elements + profile.lm_head_weight_elements
    return {
        "body_major_linear_weight_elements": profile.body_weight_elements,
        "lm_head_weight_elements": profile.lm_head_weight_elements,
        "e2e_linear_weight_elements_per_token": e2e,
        "ideal_lanes_for_100_tps_at_225MHz": e2e * 100 / 225e6,
        "evidence_tag": "CALCULATED_FROM_CONFIG",
    }


def build_report(*, include_scenarios: bool = True) -> dict[str, Any]:
    by_key = {profile.model_key: profile for profile in profiles()}
    models: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    headline_rows: list[dict[str, Any]] = []
    for variant in variants():
        profile = by_key[variant.model_key]
        per_context = {
            str(context): {
                "fully_on_card_memory": memory_components(profile, variant, context, "fully_on_card"),
                "fully_on_card_compute": compute_components(profile, variant, context, "fully_on_card"),
                "minimal_host_lm_head_offload_memory": memory_components(profile, variant, context, "minimal_host_lm_head_offload"),
                "minimal_host_lm_head_offload_compute": compute_components(profile, variant, context, "minimal_host_lm_head_offload"),
            } for context in CONTEXTS
        }
        generated_rows = scenario_rows(profile, variant)
        headline_lanes = 672 if profile.model_key == "bitnet-0.7b" else (640 if profile.model_key == "falcon3-1b-instruct-1.58bit" else (768 if profile.model_key == "bonsai-1.7b" else 896))
        headline_rows.extend(
            row for row in generated_rows
            if row["context"] in (512, 2048)
            and row["head_placement"] == "fully_on_card"
            and row["clock_MHz"] == 225
            and row["projection_lanes"] == headline_lanes
            and (
                (row["memory_mode"] == "dual_x64_payload" and row["selected_sustained_GBps"] == 31 and row["pipeline_utilization"] == 0.90)
                or (row["memory_mode"] == "dual_x72_experimental_payload" and row["selected_sustained_GBps"] == 35 and row["pipeline_utilization"] == 0.95)
            )
        )
        rows = generated_rows if include_scenarios else []
        all_rows.extend(rows)
        models.append({
            "profile": asdict(profile), "variant": asdict(variant),
            "config_regression": regression_values(profile), "per_context": per_context,
            "scenario_row_count": len(rows),
        })
    return {
        "schema_version": "catapult3-hardware-roof-v2",
        "measurement_status": "NO_BOARD_OR_POST_FIT_MEASUREMENTS; ALL THROUGHPUT_VALUES_PROJECTED_FPGA",
        "evidence_tags": ["CALCULATED_FROM_CONFIG", "PROJECTED_FPGA", "ASSUMPTION", "BLOCKED"],
        "assumptions": {
            "contexts": list(CONTEXTS), "sustained_bandwidth_GBps": list(SUSTAINED_BANDWIDTH_GBPS),
            "pipeline_utilizations": list(UTILIZATIONS),
            "memory_modes_theoretical_payload_GBps": MEMORY_MODES,
            "projection_lanes": list(PROJECTION_LANES), "clocks_MHz": list(CLOCKS_MHZ),
            "attention_elements_per_cycle": ATTENTION_ELEMENTS_PER_CYCLE,
            "scale_multiply_pipelines": SCALE_MULTIPLIES_PER_CYCLE,
            "vector_elements_per_cycle": VECTOR_ELEMENTS_PER_CYCLE,
            "topk_comparisons_per_cycle": TOPK_COMPARISONS_PER_CYCLE,
            "streaming_top_k": TOP_K, "activation_spill_bytes_per_token": 0,
            "activation_spill_reason": "ASSUMPTION: fully fused on-card intermediates; any spill reduces memory roof",
            "tl5_note": "TL5 uses the checked-in single-bank 244-cycle pipelined table build for every activation tile; its non-overlapped build cost is included in projection_total_cycles_per_token",
            "evidence_tag": "ASSUMPTION",
        },
        "models": models, "headline_scenario_rows": headline_rows, "scenario_rows": all_rows,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--no-scenarios", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    report = build_report(include_scenarios=not args.no_scenarios)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    csv_rows = report["scenario_rows"] or report["headline_scenario_rows"]
    if args.csv_output and csv_rows:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
