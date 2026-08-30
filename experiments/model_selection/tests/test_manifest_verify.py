from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from manifest_verify import ManifestVerificationError, inspect_safetensors_header, verify_artifact


def _manifest(path: Path, payload: Path, *, digest: str | None = None) -> Path:
    value = {
        "schema_version": "catapult3-model-selection-artifacts-v2",
        "artifacts": [
            {
                "key": "unit",
                "files": [
                    {
                        "role": "weight",
                        "name": payload.name,
                        "bytes": payload.stat().st_size,
                        "sha256": digest or hashlib.sha256(payload.read_bytes()).hexdigest(),
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_manifest_verifier_accepts_exact_bytes(tmp_path: Path):
    payload = tmp_path / "weights.bin"
    payload.write_bytes(b"exact")
    records = verify_artifact(_manifest(tmp_path / "manifest.json", payload), "unit", tmp_path)
    assert records[0]["verification"] == "MEASURED_LOCAL_BYTES"


def test_manifest_verifier_fails_closed_on_hash_and_missing_role(tmp_path: Path):
    payload = tmp_path / "weights.bin"
    payload.write_bytes(b"wrong")
    manifest = _manifest(tmp_path / "manifest.json", payload, digest="0" * 64)
    with pytest.raises(ManifestVerificationError, match="SHA-256 mismatch"):
        verify_artifact(manifest, "unit", tmp_path)
    exact = _manifest(tmp_path / "manifest2.json", payload)
    with pytest.raises(ManifestVerificationError, match="required roles"):
        verify_artifact(exact, "unit", tmp_path, required_roles={"weight", "config"})


def test_safetensors_header_health(tmp_path: Path):
    header = json.dumps(
        {"tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")
    health = inspect_safetensors_header(path)
    assert health["tensor_count"] == 1
    assert health["unsupported_dtypes"] == []
