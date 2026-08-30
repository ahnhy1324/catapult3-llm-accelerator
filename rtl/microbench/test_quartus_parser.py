from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_quartus_sweep import parse_reports  # noqa: E402


def test_parse_quartus_25_post_fit_tables(tmp_path):
    project = "kernel_768_225"
    (tmp_path / f"{project}.fit.summary").write_text(
        "Logic utilization (in ALMs) : 25,666 / 427,200 ( 6 % )\n"
        "Total registers : 880\n"
        "Total block memory bits : 0 / 55,562,240 ( 0 % )\n"
        "Total DSP Blocks : 18 / 1,518 ( 1 % )\n",
        encoding="utf-8",
    )
    (tmp_path / f"{project}.fit.rpt").write_text(
        "; M20K blocks ; 3 / 2,713 ;\n"
        "; Average interconnect usage (total/H/V) ; 4.2% / 5.6% / 2.1% ;\n"
        "; Peak interconnect usage (total/H/V) ; 22.0% / 20.6% / 24.4% ;\n",
        encoding="utf-8",
    )
    (tmp_path / f"{project}.sta.rpt").write_text(
        "; Fmax Summary ;\n"
        "; 67.64 MHz ; 67.64 MHz ; clk ; ; Slow 900mV 0C Model ;\n"
        "; Setup Summary ;\n"
        "; clk ; -10.340 ; -960.232 ; 109 ; Slow 900mV 0C Model ;\n"
        "; Hold Summary ;\n"
        "; clk ; 0.023 ; 0.000 ; 0 ; Fast 900mV 0C Model ;\n"
        "; -10.340 ; weight_sign_reg[511] ; value_pipe[0][39] ; clk ; clk ; 4.483 ; -0.059 ; 15.267 ; Slow ;\n"
        "; Data Delay ; 15.267 ; ;\n"
        "; Number of Logic Levels ; ; 12 ;\n",
        encoding="utf-8",
    )
    parsed = parse_reports(tmp_path, project)
    assert parsed["alm"] == 25_666
    assert parsed["registers"] == 880
    assert parsed["dsp_blocks"] == 18
    assert parsed["m20k_blocks"] == 3
    assert parsed["fmax_mhz"] == 67.64
    assert parsed["setup_slack_ns"] == -10.34
    assert parsed["setup_failing_endpoints"] == 109
    assert parsed["hold_slack_ns"] == 0.023
    assert parsed["worst_reported_data_delay_ns"] == 15.267
    assert parsed["maximum_reported_logic_levels"] == 12
    assert parsed["peak_interconnect_usage_percent"] == 22.0
    assert parsed["worst_path_from"] == "weight_sign_reg[511]"
