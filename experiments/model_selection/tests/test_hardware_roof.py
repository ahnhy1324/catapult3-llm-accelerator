from __future__ import annotations

import math

from hardware_roof import build_report, profiles, scenario_rows, variants


def _regression(model_key: str):
    report = build_report(include_scenarios=False)
    return next(model["config_regression"] for model in report["models"] if model["profile"]["model_key"] == model_key)


def test_bitnet_e2e_projection_regression_includes_lm_head():
    value = _regression("bitnet-0.7b")
    assert value["body_major_linear_weight_elements"] == 679_477_248
    assert value["lm_head_weight_elements"] == 32_002 * 1_536 == 49_155_072
    assert value["e2e_linear_weight_elements_per_token"] == 728_632_320
    assert math.isclose(value["ideal_lanes_for_100_tps_at_225MHz"], 323.8365866666667)


def test_bonsai_e2e_projection_regression_includes_lm_head():
    value = _regression("bonsai-1.7b")
    assert value["body_major_linear_weight_elements"] == 1_409_286_144
    assert value["lm_head_weight_elements"] == 151_669 * 2_048 == 310_618_112
    assert value["e2e_linear_weight_elements_per_token"] == 1_719_904_256
    assert math.isclose(value["ideal_lanes_for_100_tps_at_225MHz"], 764.4018915555556)


def test_falcon3_1b_instruct_geometry_includes_large_untied_head():
    value = _regression("falcon3-1b-instruct-1.58bit")
    assert value["body_major_linear_weight_elements"] == 1_132_462_080
    assert value["lm_head_weight_elements"] == 131_072 * 2_048 == 268_435_456
    assert value["e2e_linear_weight_elements_per_token"] == 1_400_897_536
    assert math.isclose(value["ideal_lanes_for_100_tps_at_225MHz"], 622.6211271111112)


def test_falcon3_official_fp16_head_dominates_traffic():
    report = build_report(include_scenarios=False)
    falcon = next(model for model in report["models"] if model["variant"]["key"] == "falcon3-1b-official-fp16-head-kv8")
    memory = falcon["per_context"]["512"]["fully_on_card_memory"]
    assert memory["body_weight_bytes_per_token"] == math.ceil(1_132_462_080 / 5)
    assert memory["lm_head_bytes_per_token"] == 2 * 268_435_456


def test_embedding_reads_one_row_and_lm_head_reads_full_matrix():
    report = build_report(include_scenarios=False)
    bitnet = next(model for model in report["models"] if model["variant"]["key"] == "bitnet-row16-kv4")
    memory = bitnet["per_context"]["512"]["fully_on_card_memory"]
    assert memory["input_embedding_row_bytes_per_token"] == 1536 // 2
    assert memory["lm_head_bytes_per_token"] == math.ceil((32_002 * 1_536) * 4 / 8)


def test_projection_attention_scale_and_memory_roofs_remain_separate():
    profile = next(item for item in profiles() if item.model_key == "bonsai-1.7b")
    variant = next(item for item in variants() if item.key == "bonsai-a8g128-kv8")
    row = next(
        item for item in scenario_rows(profile, variant)
        if item["context"] == 512
        and item["head_placement"] == "fully_on_card"
        and item["memory_mode"] == "dual_x64_payload"
        and item["selected_sustained_GBps"] == 31
        and item["pipeline_utilization"] == 0.9
        and item["projection_lanes"] == 768
        and item["clock_MHz"] == 225
    )
    assert row["projection_compute_roof_tps"] > 100
    assert row["memory_roof_tps"] > 100
    assert row["final_bottleneck_roof_tps"] == min(
        row["memory_roof_tps"], row["projection_compute_roof_tps"], row["attention_compute_roof_tps"],
        row["scale_multiply_roof_tps"], row["vector_compute_roof_tps"], row["topk_roof_tps"],
    )


def test_x64_35gbps_is_explicitly_invalid_not_silently_used():
    profile = next(item for item in profiles() if item.model_key == "bitnet-0.7b")
    variant = next(item for item in variants() if item.key == "bitnet-row16-kv4")
    row = next(
        item for item in scenario_rows(profile, variant)
        if item["memory_mode"] == "dual_x64_payload" and item["selected_sustained_GBps"] == 35
    )
    assert row["bandwidth_scenario_valid"] is False
    assert row["evidence_tag"] == "BLOCKED"


def test_tl5_single_bank_build_cost_is_not_hidden():
    profile = next(item for item in profiles() if item.model_key == "bitnet-0.7b")
    variant = next(item for item in variants() if item.key == "bitnet-row16-kv4")
    selected = [
        item for item in scenario_rows(profile, variant)
        if item["context"] == 512
        and item["head_placement"] == "fully_on_card"
        and item["memory_mode"] == "dual_x64_payload"
        and item["selected_sustained_GBps"] == 31
        and item["pipeline_utilization"] == 0.9
        and item["projection_lanes"] == 672
        and item["clock_MHz"] == 225
    ]
    direct = next(item for item in selected if item["datapath"] == "direct_ternary")
    tl5 = next(item for item in selected if item["datapath"] == "TL5")
    assert direct["tl5_table_build_cycles_per_token"] == 0
    assert tl5["tl5_table_build_cycles_per_token"] == 244 * 24 * (6 * 3 + 7)
    assert tl5["projection_compute_roof_tps"] < direct["projection_compute_roof_tps"]
