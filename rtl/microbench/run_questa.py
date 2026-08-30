#!/usr/bin/env python3
"""Generate NumPy golden vectors and run the Questa functional microbench."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from golden import binary_g128_dot, pack_trits, ternary_dot


HERE = Path(__file__).resolve().parent


def _sv_assign(name: str, values) -> list[str]:
    lines = []
    array = np.asarray(values)
    for index in np.ndindex(array.shape):
        suffix = "".join(f"[{value}]" for value in index)
        lines.append(f"    {name}{suffix} = {int(array[index])};")
    return lines


def generate_include(path: Path) -> dict[str, int]:
    rng = np.random.default_rng(20260831)
    binary_vectors = 6
    binary_activation = rng.integers(-128, 128, size=(binary_vectors, 128), dtype=np.int16)
    binary_activation[0] = 127
    binary_activation[1] = -128
    binary_sign = rng.choice([-1, 1], size=(binary_vectors, 128)).astype(np.int8)
    binary_sign[0] = 1
    binary_sign[1] = -1
    binary_scale = np.array([3, -3, 257, 128, 1, 511], dtype=np.int32)
    binary_expected = []
    binary_saturation = []
    for index in range(binary_vectors):
        value, saturation = binary_g128_dot(
            binary_activation[index], binary_sign[index], [binary_scale[index]],
            accumulator_bits=14, scale_fraction_bits=1, output_bits=16,
        )
        binary_expected.append(value)
        binary_saturation.append(int(saturation))

    ternary_activation = np.array([-128, 127, -64, 63, 0, 1, -1, 42, -42, 7], dtype=np.int16)
    ternary_trits = np.array([
        [1] * 10,
        [-1] * 10,
        [0] * 10,
        [-1, 0, 1, 1, -1, 0, 1, -1, 0, 1],
        rng.integers(-1, 2, size=10),
        rng.integers(-1, 2, size=10),
    ], dtype=np.int8)
    packed = np.asarray([pack_trits(row) for row in ternary_trits], dtype=np.int16)
    ternary_expected = [ternary_dot(ternary_activation, row, output_bits=16)[0] for row in ternary_trits]

    lines = [
        f"localparam int BINARY_VECTORS = {binary_vectors};",
        f"localparam int TERNARY_VECTORS = {len(ternary_trits)};",
        "integer binary_activation_vector [0:BINARY_VECTORS-1][0:127];",
        "integer binary_sign_vector [0:BINARY_VECTORS-1][0:127];",
        "integer binary_scale_vector [0:BINARY_VECTORS-1];",
        "integer binary_expected_vector [0:BINARY_VECTORS-1];",
        "integer binary_saturation_vector [0:BINARY_VECTORS-1];",
        "integer ternary_activation_vector [0:9];",
        "integer ternary_packed_vector [0:TERNARY_VECTORS-1][0:1];",
        "integer ternary_expected_vector [0:TERNARY_VECTORS-1];",
        "initial begin",
    ]
    lines += _sv_assign("binary_activation_vector", binary_activation)
    lines += _sv_assign("binary_sign_vector", binary_sign)
    lines += _sv_assign("binary_scale_vector", binary_scale)
    lines += _sv_assign("binary_expected_vector", binary_expected)
    lines += _sv_assign("binary_saturation_vector", binary_saturation)
    lines += _sv_assign("ternary_activation_vector", ternary_activation)
    lines += _sv_assign("ternary_packed_vector", packed)
    lines += _sv_assign("ternary_expected_vector", ternary_expected)
    lines += ["end"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"binary_vectors": binary_vectors, "ternary_vectors": len(ternary_trits)}


def _tool(name: str, directory: Path | None) -> str:
    if directory:
        candidate = directory / f"{name}.exe"
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(name)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questa-bin", type=Path, default=Path(os.environ["QUESTA_BIN"]) if "QUESTA_BIN" in os.environ else None)
    parser.add_argument("--license-file", type=Path)
    parser.add_argument("--keep-work", type=Path)
    args = parser.parse_args()
    if args.keep_work:
        work = args.keep_work.resolve()
        work.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="catapult3-rtl-")
        work = Path(cleanup.name)
    vectors = generate_include(work / "generated_vectors.svh")
    vlog = _tool("vlog", args.questa_bin)
    vsim = _tool("vsim", args.questa_bin)
    vlib = _tool("vlib", args.questa_bin)
    environment = os.environ.copy()
    if args.license_file:
        environment["LM_LICENSE_FILE"] = str(args.license_file.resolve())
        environment["MGLS_LICENSE_FILE"] = str(args.license_file.resolve())
    subprocess.run([vlib, "work"], cwd=work, env=environment, check=True)
    subprocess.run([vlog, "-work", "work", "+acc", "-sv", f"+incdir+{work}", str(HERE / "bonsai_binary_g128_dot.sv"), str(HERE / "bitnet_direct_ternary.sv"), str(HERE / "bitnet_tl5.sv"), str(HERE / "tb_microbench.sv")], cwd=work, env=environment, check=True)
    completed = subprocess.run(
        [vsim, "-c", "work.tb_microbench", "-do", "run -all; quit -f"],
        cwd=work,
        env=environment,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"vsim exited {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    if "MICROBENCH_PASS" not in completed.stdout:
        raise RuntimeError(completed.stdout + completed.stderr)
    print(json.dumps({"status": "PASS", **vectors, "simulator": "Questa", "evidence_tag": "MEASURED_CPU"}, indent=2))
    if cleanup:
        cleanup.cleanup()


if __name__ == "__main__":
    main()
