#!/usr/bin/env python3
"""Build the compact v2 decision summary from checked-in result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def variant(result: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in result["variants"] if item["name"] == name)


def bpb_pair(item: dict[str, Any]) -> dict[str, float]:
    return {
        "context_512": item["fixed_byte"]["512"]["bits_per_byte"],
        "context_2048": item["fixed_byte"]["2048"]["bits_per_byte"],
    }


def roof_row(
    rows: list[dict[str, Any]],
    *,
    model_key: str,
    profile: str,
    datapath: str,
    lanes: int,
    context: int,
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["model_key"] == model_key
        and row["variant"] == profile
        and row["datapath"] == datapath
        and row["projection_lanes"] == lanes
        and row["context"] == context
        and row["clock_MHz"] == 225
        and row["head_placement"] == "fully_on_card"
        and row["memory_mode"] == "dual_x64_payload"
        and row["selected_sustained_GBps"] == 31
        and row["pipeline_utilization"] == 0.9
    )


def compact_roof(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "context",
            "projection_compute_roof_tps",
            "attention_compute_roof_tps",
            "memory_roof_tps",
            "scale_multiply_roof_tps",
            "vector_compute_roof_tps",
            "topk_roof_tps",
            "final_bottleneck",
            "final_bottleneck_roof_tps",
            "classification",
            "evidence_tag",
        )
    }


def compact_rtl(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "top",
        "lanes",
        "target_clock_mhz",
        "alm",
        "registers",
        "memory_bits",
        "m20k_blocks",
        "dsp_blocks",
        "fmax_mhz",
        "setup_slack_ns",
        "setup_failing_endpoints",
        "maximum_reported_logic_levels",
        "peak_interconnect_usage_percent",
        "initiation_interval_cycles",
        "target_clock_met",
        "clock_target_results_from_routed_fmax",
        "measurement_kind",
        "evidence_tag",
    )
    return {key: row.get(key) for key in keys}


def build(results_dir: Path) -> dict[str, Any]:
    bitnet = load_json(results_dir / "bitnet.json")
    bonsai = load_json(results_dir / "bonsai_transformers.json")
    bonsai_native = load_json(results_dir / "bonsai_native.json")
    falcon = load_json(results_dir / "falcon3-1b-instruct-limited.json")
    roof = load_json(results_dir / "hardware_roof.json")
    rtl = load_json(results_dir / "rtl_sweep.json")

    bitnet_joint = variant(bitnet, "row16_head_plus_KV4")
    bonsai_joint = variant(bonsai, "A8_g128_plus_KV8")
    bitnet_roofs = [
        compact_roof(
            roof_row(
                roof["headline_scenario_rows"],
                model_key="bitnet-0.7b",
                profile="bitnet-row16-kv4",
                datapath="direct_ternary",
                lanes=672,
                context=context,
            )
        )
        for context in (512, 2048)
    ]
    bonsai_roofs = [
        compact_roof(
            roof_row(
                roof["headline_scenario_rows"],
                model_key="bonsai-1.7b",
                profile="bonsai-a8g128-kv8",
                datapath="binary_g128",
                lanes=768,
                context=context,
            )
        )
        for context in (512, 2048)
    ]

    return {
        "schema_version": "catapult3-model-selection-v2-summary-v1",
        "source_results": [
            "bitnet.json",
            "bonsai_transformers.json",
            "bonsai_native.json",
            "falcon3-1b-instruct-limited.json",
            "hardware_roof.json",
            "rtl_sweep.json",
        ],
        "decision": {
            "first_fpga_bring_up": {
                "model": "BitNet 0.7B public reproduction",
                "profile": "row16 head + KV4; direct ternary datapath",
                "choice": "A",
                "reason": "smallest e2e projection and only selected candidate above 100 tok/s at context 2048",
                "quality_scope": "bring-up only because prompt echo/repetition and weak Korean were measured",
                "evidence_tags": ["MEASURED_CPU", "CALCULATED_FROM_CONFIG", "PROJECTED_FPGA"],
            },
            "headline_target": {
                "model": "Bonsai 1.7B official Q1 g128",
                "profile": "A8 g128 + KV8",
                "reason": "official packed checkpoint/runtime, coherent generation, and stable measured joint codec",
                "claim_scope": "fully on-card context 512; context 2048 is a modeled no-go",
                "evidence_tags": ["MEASURED_CPU", "MEASURED_MODEL_FILE", "PROJECTED_FPGA"],
            },
        },
        "fixed_byte_corpus": {
            "utf8_bytes": 24576,
            "sha256": "c21856723534065f53feb61320d4276722b70680b61834e2f2abeb15ae572f6b",
            "bitnet_tokens": bitnet["baseline"]["fixed_byte"]["512"]["scored_tokens"],
            "bonsai_tokens": bonsai["baseline"]["fixed_byte"]["512"]["scored_tokens"],
            "evidence_tag": "MEASURED_CPU",
        },
        "cpu_quality": {
            "bitnet": {
                "status": bitnet["status"],
                "baseline_bpb": bpb_pair(bitnet["baseline"]),
                "selected_joint_bpb": bpb_pair(bitnet_joint),
                "selected_joint_smoke_ppl_ratio": bitnet_joint["aggregate"]["perplexity_ratio"],
                "selected_joint_exact_generation_agreement": bitnet_joint["aggregate"]["exact_generation_agreement_rate"],
                "peak_rss_bytes": bitnet["performance"]["peak_rss_bytes"],
                "evidence_tag": "MEASURED_CPU",
            },
            "bonsai": {
                "unpacked_status": bonsai["status"],
                "baseline_bpb": bpb_pair(bonsai["baseline"]),
                "selected_joint_bpb": bpb_pair(bonsai_joint),
                "selected_joint_smoke_ppl_ratio": bonsai_joint["aggregate"]["perplexity_ratio"],
                "selected_joint_exact_generation_agreement": bonsai_joint["aggregate"]["exact_generation_agreement_rate"],
                "native_q1_status": bonsai_native["status"],
                "native_q1_cpu_tokens_per_second": bonsai_native["performance"]["mean_native_generation_tokens_per_second"],
                "peak_unpacked_rss_bytes": bonsai["performance"]["peak_rss_bytes"],
                "evidence_tag": "MEASURED_CPU",
            },
            "falcon_limited": {
                "status": falcon["status"],
                "smoke_ppl": falcon["baseline"]["perplexity"]["perplexity"],
                "e2e_linear_weight_elements_per_token": falcon["model"]["geometry"]["e2e_linear_weight_elements_per_token"],
                "disposition": "not expanded because its fully-on-card memory roof is about 35.6 tok/s",
                "evidence_tags": ["MEASURED_CPU", "CALCULATED_FROM_CONFIG", "PROJECTED_FPGA"],
            },
        },
        "fully_on_card_roof": {
            "scenario": {
                "memory": "dual_x64_payload",
                "selected_sustained_GBps": 31,
                "pipeline_utilization": 0.9,
                "clock_MHz": 225,
            },
            "bitnet_row16_kv4_direct_672": bitnet_roofs,
            "bonsai_a8g128_kv8_binary_768": bonsai_roofs,
        },
        "rtl_post_fit": [compact_rtl(row) for row in rtl["rows"]],
        "blockers": {
            "local_rtl_simulation": "Questa vlog/vopt pass, but vsim is blocked by Windows 0x80096010 at design load; GitHub Icarus is authoritative",
            "board_measurement": "not run; Quartus rows are post-fit estimates, not board throughput",
        },
        "largest_remaining_risk": "full-system 225 MHz closure and sustained memory scheduling after projection, attention, scale, and top-k integration",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/model_selection_v2"))
    parser.add_argument("--output", type=Path, default=Path("results/model_selection_v2/summary.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(args.results_dir), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
