#!/usr/bin/env python3
"""Run and summarize the Catapult3 microkernel Quartus sweep."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_QUARTUS = Path(r"E:\Altera\quartus\bin64\quartus_sh.exe")


def points() -> list[tuple[str, int, int]]:
    result = []
    for lanes in (640, 768, 896):
        for clock in (200, 225, 240):
            result.append(("bonsai_binary_g128_dot", lanes, clock))
    for top in ("bitnet_direct_ternary", "bitnet_tl5"):
        for lanes in (640, 672):
            for clock in (200, 225, 240):
                result.append((top, lanes, clock))
    return result


def _match(text: str, patterns: list[str]) -> int | float | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def parse_reports(build: Path, project: str) -> dict[str, object]:
    fit_summary = (build / f"{project}.fit.summary").read_text(encoding="utf-8", errors="replace") if (build / f"{project}.fit.summary").exists() else ""
    fit_report = (build / f"{project}.fit.rpt").read_text(encoding="utf-8", errors="replace") if (build / f"{project}.fit.rpt").exists() else ""
    sta_report = (build / f"{project}.sta.rpt").read_text(encoding="utf-8", errors="replace") if (build / f"{project}.sta.rpt").exists() else ""
    combined = fit_summary + "\n" + fit_report
    fmax_row = re.search(
        r";\s*([0-9]+(?:\.[0-9]+)?)\s+MHz\s*;\s*([0-9]+(?:\.[0-9]+)?)\s+MHz\s*;\s*clk\s*;",
        sta_report,
        re.IGNORECASE,
    )
    setup_row = re.search(
        r";\s*clk\s*;\s*(-?[0-9]+(?:\.[0-9]+)?)\s*;\s*(-?[0-9]+(?:\.[0-9]+)?)\s*;\s*([0-9]+)\s*;\s*Slow[^;]*;",
        sta_report,
        re.IGNORECASE,
    )
    hold_offset = sta_report.rfind("; Hold Summary")
    hold_section = sta_report[hold_offset:] if hold_offset >= 0 else ""
    hold_row = re.search(
        r";\s*clk\s*;\s*(-?[0-9]+(?:\.[0-9]+)?)\s*;\s*(-?[0-9]+(?:\.[0-9]+)?)\s*;\s*([0-9]+)\s*;",
        hold_section,
        re.IGNORECASE,
    )
    data_delays = [float(value) for value in re.findall(r";\s*Data Delay\s*;\s*([0-9]+(?:\.[0-9]+)?)", sta_report)]
    logic_levels = [int(value) for value in re.findall(r";\s*Number of Logic Levels\s*;\s*(?:;\s*)?([0-9]+)", sta_report)]
    average_interconnect = re.search(r"Average interconnect usage \(total/H/V\)\s*;\s*([0-9.]+)%", fit_report)
    peak_interconnect = re.search(r"Peak interconnect usage \(total/H/V\)\s*;\s*([0-9.]+)%", fit_report)
    worst_path = re.search(
        r";\s*(-?[0-9]+(?:\.[0-9]+)?)\s*;\s*([^;]+?)\s*;\s*([^;]+?)\s*;\s*clk\s*;\s*clk\s*;",
        sta_report,
        re.IGNORECASE,
    )
    return {
        "alm": _match(combined, [r"Logic utilization \(in ALMs\)\s*:\s*([0-9,]+)", r"ALMs needed[^:]*:\s*([0-9,]+)"]),
        "registers": _match(combined, [r"Total registers[^:]*:\s*([0-9,]+)", r"Dedicated logic registers[^:]*:\s*([0-9,]+)"]),
        "memory_bits": _match(combined, [r"Total block memory bits[^:]*:\s*([0-9,]+)"]),
        "m20k_blocks": _match(combined, [r"M20K blocks\s*[;:]\s*([0-9,]+)", r"M20Ks\s*[;:]\s*([0-9,]+)", r"Total RAM Blocks\s*[;:]\s*([0-9,]+)"]),
        "dsp_blocks": _match(combined, [r"Total DSP Blocks\s*:\s*([0-9,]+)", r"DSP Blocks Needed[^:]*:\s*([0-9,]+)"]),
        "fmax_mhz": float(fmax_row.group(1)) if fmax_row else None,
        "restricted_fmax_mhz": float(fmax_row.group(2)) if fmax_row else None,
        "setup_slack_ns": float(setup_row.group(1)) if setup_row else None,
        "setup_tns_ns": float(setup_row.group(2)) if setup_row else None,
        "setup_failing_endpoints": int(setup_row.group(3)) if setup_row else None,
        "hold_slack_ns": float(hold_row.group(1)) if hold_row else None,
        "hold_tns_ns": float(hold_row.group(2)) if hold_row else None,
        "hold_failing_endpoints": int(hold_row.group(3)) if hold_row else None,
        "worst_reported_data_delay_ns": max(data_delays) if data_delays else None,
        "maximum_reported_logic_levels": max(logic_levels) if logic_levels else None,
        "average_interconnect_usage_percent": float(average_interconnect.group(1)) if average_interconnect else None,
        "peak_interconnect_usage_percent": float(peak_interconnect.group(1)) if peak_interconnect else None,
        "worst_path_from": worst_path.group(2).strip() if worst_path else None,
        "worst_path_to": worst_path.group(3).strip() if worst_path else None,
        "fit_summary": f"{project}.fit.summary (external Quartus build artifact)",
        "sta_report": f"{project}.sta.rpt (external Quartus build artifact)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quartus", type=Path, default=DEFAULT_QUARTUS)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=("all", "representative"), default="all")
    parser.add_argument("--parse-existing", action="store_true")
    parser.add_argument("--point", nargs=3, metavar=("TOP", "LANES", "CLOCK_MHZ"))
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    selected = points()
    if args.only == "representative":
        selected = [
            ("bonsai_binary_g128_dot", 768, 225),
            ("bitnet_direct_ternary", 640, 225),
            ("bitnet_tl5", 640, 225),
        ]
    if args.point:
        selected = [(args.point[0], int(args.point[1]), int(args.point[2]))]
    rows = []
    if args.append and args.output.exists():
        rows = json.loads(args.output.read_text(encoding="utf-8")).get("rows", [])
    for top, lanes, clock in selected:
        project = f"{top}_{lanes}_{clock}"
        rows = [row for row in rows if (row["top"], row["lanes"], row["target_clock_mhz"]) != (top, lanes, clock)]
        build = (args.build_root / project).resolve()
        command = [str(args.quartus), "-t", str(HERE / "synth_microbench.tcl"), top, str(lanes), str(clock), str(build)]
        started = time.perf_counter()
        if args.parse_existing:
            flow_report = build / f"{project}.flow.rpt"
            existing_text = flow_report.read_text(encoding="utf-8", errors="replace") if flow_report.exists() else ""
            returncode = 0 if re.search(r"Flow Status\s*;\s*Successful", existing_text) else 1
            log_tail = existing_text[-4000:]
        else:
            completed = subprocess.run(command, text=True, capture_output=True)
            returncode = completed.returncode
            log_tail = (completed.stdout + completed.stderr)[-4000:]
        row = {
            "top": top,
            "lanes": lanes,
            "target_clock_mhz": clock,
            "device": "10AXF40GAE (Quartus-resolved 10AX115_JZ)",
            "quartus": "25.1.0 Build 129 Patch 0.36 SC Pro",
            "exit_code": returncode,
            "wall_seconds": time.perf_counter() - started,
            "status": "PASS" if returncode == 0 else "FAIL",
            "log_tail": log_tail,
            "evidence_tag": "PROJECTED_FPGA",
            "measurement_kind": "QUARTUS_POST_FIT_TIMING_ESTIMATE",
            "initiation_interval_cycles": 1,
            "useful_weight_elements_per_cycle": lanes,
            "groups_per_cycle": lanes / (128 if top == "bonsai_binary_g128_dot" else 5),
            "scale_multiplies_per_cycle": lanes / 128 if top == "bonsai_binary_g128_dot" else 0,
        }
        row.update(parse_reports(build, project))
        row["target_clock_met"] = (
            row["setup_slack_ns"] is not None and row["setup_slack_ns"] >= 0
        )
        row["clock_target_results_from_routed_fmax"] = {
            str(candidate): bool(row["fmax_mhz"] is not None and row["fmax_mhz"] >= candidate)
            for candidate in (200, 225, 240)
        }
        row["clock_target_method"] = (
            "Compare each requested 200/225/240 MHz target with the final routed Fmax from the 225 MHz constrained fit; "
            "separate fitter seeds were not run."
        )
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"schema_version": "catapult3-rtl-sweep-v1", "rows": rows}, indent=2) + "\n", encoding="utf-8")
    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
