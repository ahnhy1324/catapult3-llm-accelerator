from __future__ import annotations

from hardware_roof import build_report


def test_all_required_sweeps_present():
    report = build_report()
    assert len(report["models"]) == 4
    expected_rows = 4 * 3 * 4 * 3
    for model in report["models"]:
        assert len(model["rows"]) == expected_rows
        assert set(model["compute"]["lane_requirement_by_clock_MHz"]) == {"200", "210", "225", "240"}


def test_theoretical_and_sustained_are_separate():
    report = build_report()
    assumptions = report["assumptions"]
    assert assumptions["theoretical_bandwidth_GBps"]["dual_x64_DDR4_2133_GBps"] == 34.128
    assert assumptions["theoretical_bandwidth_GBps"]["dual_x72_DDR4_2133_GBps"] == 38.394
    assert assumptions["sustained_bandwidth_GBps"] == [29, 31, 33, 35]


def test_bonsai_4b_and_8b_cannot_reach_100_from_weights_alone_at_35gbps():
    report = build_report()
    by_key = {model["profile"]["model_key"]: model for model in report["models"]}
    for key in ("bonsai-4b", "bonsai-8b"):
        weight_bytes = by_key[key]["packed_model_size"]["calculated_streamed_weight_bytes_per_token"]
        assert 35e9 / weight_bytes < 100


def test_bonsai_17_compute_lane_boundary_is_visible():
    report = build_report()
    model = next(item for item in report["models"] if item["profile"]["model_key"] == "bonsai-1.7b")
    lanes = model["compute"]["lane_requirement_by_clock_MHz"]
    assert lanes["200"] > 672
    assert lanes["225"] < 640
