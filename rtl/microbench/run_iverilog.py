#!/usr/bin/env python3
"""Generate the same golden vectors and run the RTL with Icarus Verilog."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from run_questa import HERE, generate_include


def _resolve_tool(explicit: Path | None, name: str) -> str:
    if explicit:
        return str(explicit.resolve())
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(name)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iverilog", type=Path)
    parser.add_argument("--vvp", type=Path)
    parser.add_argument("--keep-work", type=Path)
    args = parser.parse_args()
    if args.keep_work:
        work = args.keep_work.resolve()
        work.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="catapult3-iverilog-")
        work = Path(cleanup.name)
    vectors = generate_include(work / "generated_vectors.svh")
    output = work / "microbench.vvp"
    compile_command = [
        _resolve_tool(args.iverilog, "iverilog"),
        "-g2012",
        "-Wall",
        "-s",
        "tb_microbench",
        "-I",
        str(work),
        "-o",
        str(output),
        str(HERE / "bonsai_binary_g128_dot.sv"),
        str(HERE / "bitnet_direct_ternary.sv"),
        str(HERE / "bitnet_tl5.sv"),
        str(HERE / "tb_microbench.sv"),
    ]
    subprocess.run(compile_command, cwd=work, check=True)
    completed = subprocess.run(
        [_resolve_tool(args.vvp, "vvp"), str(output)],
        cwd=work,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0 or "MICROBENCH_PASS" not in completed.stdout:
        raise RuntimeError(
            f"Icarus simulation failed ({completed.returncode})\n{completed.stdout}\n{completed.stderr}"
        )
    print(json.dumps({
        "status": "PASS",
        **vectors,
        "simulator": "Icarus Verilog",
        "evidence_tag": "MEASURED_CPU",
        "log": completed.stdout.strip(),
    }, indent=2))
    if cleanup:
        cleanup.cleanup()


if __name__ == "__main__":
    main()
