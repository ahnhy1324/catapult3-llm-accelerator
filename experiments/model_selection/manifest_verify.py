#!/usr/bin/env python3
"""Fail-closed artifact verification for model-selection v2.

The verifier hashes the bytes that a runner is about to load.  It does not
trust a Hugging Face cache name, an LFS pointer, or a previously generated
result JSON.  Large files are streamed so verification stays within the
16-GiB CI memory envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_SAFETENSOR_DTYPES = {
    "BOOL",
    "F8_E4M3",
    "F8_E5M2",
    "F16",
    "BF16",
    "F32",
    "F64",
    "I8",
    "I16",
    "I32",
    "I64",
    "U8",
    "U16",
    "U32",
    "U64",
}


class ManifestVerificationError(RuntimeError):
    """Raised when an artifact differs from the pinned manifest."""


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "catapult3-model-selection-artifacts-v2":
        raise ManifestVerificationError(f"unsupported manifest schema: {value.get('schema_version')!r}")
    return value


def _artifact(manifest: dict[str, Any], artifact_key: str) -> dict[str, Any]:
    matches = [item for item in manifest.get("artifacts", []) if item.get("key") == artifact_key]
    if len(matches) != 1:
        raise ManifestVerificationError(
            f"manifest must contain exactly one artifact key {artifact_key!r}; found {len(matches)}"
        )
    return matches[0]


def verify_artifact(
    manifest_path: Path,
    artifact_key: str,
    root: Path,
    *,
    required_roles: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Verify every pinned file in one artifact group.

    ``required_roles`` connects the manifest to the loader contract.  A runner
    fails before model construction if a role it can load is not represented.
    """
    manifest = load_manifest(manifest_path)
    artifact = _artifact(manifest, artifact_key)
    root = root.resolve()
    files = artifact.get("files", [])
    roles = [str(item.get("role", "")) for item in files]
    if len(roles) != len(set(roles)):
        raise ManifestVerificationError(f"duplicate file role in {artifact_key}")
    if required_roles is not None:
        missing_roles = sorted(set(required_roles) - set(roles))
        if missing_roles:
            raise ManifestVerificationError(f"manifest lacks required roles for {artifact_key}: {missing_roles}")

    verified: list[dict[str, Any]] = []
    for record in files:
        relative = Path(record["name"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ManifestVerificationError(f"artifact path escapes root: {relative}") from exc
        if not path.is_file():
            raise ManifestVerificationError(f"missing {artifact_key}/{relative}")
        actual_bytes = path.stat().st_size
        expected_bytes = int(record["bytes"])
        if actual_bytes != expected_bytes:
            raise ManifestVerificationError(
                f"size mismatch for {artifact_key}/{relative}: {actual_bytes} != {expected_bytes}"
            )
        actual_hash = sha256_file(path)
        expected_hash = str(record["sha256"]).lower()
        if actual_hash != expected_hash:
            raise ManifestVerificationError(
                f"SHA-256 mismatch for {artifact_key}/{relative}: {actual_hash} != {expected_hash}"
            )
        verified.append(
            {
                "role": record["role"],
                "name": relative.as_posix(),
                "byte_size": actual_bytes,
                "sha256": actual_hash,
                "verification": "MEASURED_LOCAL_BYTES",
            }
        )
    return verified


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestVerificationError(f"duplicate safetensors header key: {key}")
        result[key] = value
    return result


def inspect_safetensors_header(path: Path) -> dict[str, Any]:
    """Inspect shape, offsets, duplicate names, and dtypes without loading data."""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ManifestVerificationError(f"truncated safetensors prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes <= 0 or header_bytes > file_size - 8:
            raise ManifestVerificationError(f"invalid safetensors header length: {header_bytes}")
        raw_header = handle.read(header_bytes)
    header = json.loads(raw_header, object_pairs_hook=_reject_duplicate_json_keys)
    tensor_count = 0
    unsupported: list[dict[str, str]] = []
    invalid_shapes: list[str] = []
    invalid_offsets: list[str] = []
    intervals: list[tuple[int, int, str]] = []
    for name, record in header.items():
        if name == "__metadata__":
            continue
        tensor_count += 1
        dtype = str(record.get("dtype"))
        if dtype not in SUPPORTED_SAFETENSOR_DTYPES:
            unsupported.append({"tensor": name, "dtype": dtype})
        shape = record.get("shape")
        if not isinstance(shape, list) or any(not isinstance(value, int) or value < 0 for value in shape):
            invalid_shapes.append(name)
        offsets = record.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > file_size - 8 - header_bytes
        ):
            invalid_offsets.append(name)
        else:
            intervals.append((offsets[0], offsets[1], name))
    intervals.sort()
    overlap = [
        (left[2], right[2])
        for left, right in zip(intervals, intervals[1:])
        if left[1] > right[0]
    ]
    if unsupported or invalid_shapes or invalid_offsets or overlap:
        raise ManifestVerificationError(
            "invalid safetensors header: "
            f"unsupported={unsupported}, invalid_shapes={invalid_shapes}, "
            f"invalid_offsets={invalid_offsets}, overlap={overlap}"
        )
    return {
        "tensor_count": tensor_count,
        "duplicate_tensor_names": [],
        "unsupported_dtypes": unsupported,
        "invalid_shapes": invalid_shapes,
        "overlapping_data_ranges": overlap,
        "evidence_tag": "MEASURED_MODEL_FILE",
    }


def inspect_gguf_magic(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        magic = handle.read(4)
        version_raw = handle.read(4)
    if magic != b"GGUF" or len(version_raw) != 4:
        raise ManifestVerificationError(f"invalid GGUF prefix: {path}")
    return {
        "magic": "GGUF",
        "version": struct.unpack("<I", version_raw)[0],
        "quantization_type_check": "DEFERRED_TO_PINNED_STRICT_RUNTIME_LOADER",
        "evidence_tag": "MEASURED_MODEL_FILE",
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-key", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--safetensors", type=Path)
    parser.add_argument("--gguf", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result: dict[str, Any] = {
        "artifact_key": args.artifact_key,
        "files": verify_artifact(args.manifest, args.artifact_key, args.root),
        "evidence_tag": "MEASURED_MODEL_FILE",
    }
    if args.safetensors:
        safetensors_path = args.safetensors
        if not safetensors_path.is_absolute():
            safetensors_path = args.root / safetensors_path
        result["safetensors"] = inspect_safetensors_header(safetensors_path)
    if args.gguf:
        gguf_path = args.gguf
        if not gguf_path.is_absolute():
            gguf_path = args.root / gguf_path
        result["gguf"] = inspect_gguf_magic(gguf_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
