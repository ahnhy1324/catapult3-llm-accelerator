from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest


def test_minimal_result_validates():
    schema = json.loads((Path(__file__).parents[1] / "result_schema.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": "catapult3-model-selection-result-v1",
        "run_id": "unit-test",
        "run_mode": "smoke",
        "status": "PASS",
        "evidence_scope": ["CPU_MEASURED"],
        "environment": {},
        "model": {"candidate": "x", "model_id": "x/y", "architecture": "x", "parameter_class": "tiny"},
        "checkpoint": {
            "revision": "0123456789",
            "files": [{"name": "x", "byte_size": 1, "sha256": "0" * 64, "verification": "LOCAL_BYTES"}],
        },
        "backend": {"name": "test", "revision": "test", "execution_path": "test"},
        "health": {
            "finite": True,
            "all_zero_tensors": [],
            "missing_tensors": [],
            "unexpected_tensors": [],
            "abnormal_scales": [],
        },
        "baseline": {"logits_sha256": "1" * 64, "prompts": [], "perplexity": None},
        "variants": [],
        "performance": {"wall_seconds": 0.0, "peak_rss_bytes": 0},
        "artifacts": [],
        "blockers": [],
    }
    jsonschema.validate(result, schema)


@pytest.mark.parametrize(
    "name",
    [
        "bitnet_0_7b_smoke.json",
        "bonsai_1_7b_reference_smoke.json",
        "bonsai_1_7b_native_smoke.json",
    ],
)
def test_checked_in_cpu_result_validates(name: str):
    root = Path(__file__).parents[1]
    schema = json.loads((root / "result_schema.json").read_text(encoding="utf-8"))
    result = json.loads((root / "results" / name).read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)


def test_v2_schema_requires_explicit_evidence_tags():
    root = Path(__file__).parents[1]
    schema = json.loads((root / "result_schema_v2.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": "catapult3-model-selection-result-v2",
        "run_id": "unit-v2",
        "run_mode": "metadata_only",
        "status": "BLOCKED",
        "evidence_tags": ["MEASURED_MODEL_FILE", "BLOCKED"],
        "environment": {},
        "model": {"model_id": "x/y", "architecture": "x", "parameter_class": "tiny"},
        "checkpoint": {
            "revision": "0123456789",
            "manifest": "artifact_manifest_v2.json",
            "files": [{"name": "x", "byte_size": 1, "sha256": "0" * 64, "verification": "MEASURED_LOCAL_BYTES"}],
        },
        "backend": {"name": "test", "revision": "test", "execution_path": "test"},
        "health": {"finite": True, "all_zero_tensors": [], "missing_tensors": [], "unexpected_tensors": [], "abnormal_scales": []},
        "baseline": {"logits_sha256": "1" * 64, "prompts": [], "perplexity": None},
        "variants": [],
        "performance": {"wall_seconds": 0.0, "peak_rss_bytes": 0},
        "artifacts": [],
        "blockers": [],
    }
    jsonschema.validate(result, schema)


@pytest.mark.parametrize(
    "name",
    [
        "bitnet.json",
        "bonsai_transformers.json",
        "bonsai_native.json",
        "falcon3-1b-instruct-limited.json",
    ],
)
def test_checked_in_v2_cpu_result_validates(name: str):
    repository = Path(__file__).parents[3]
    schema = json.loads(
        (repository / "experiments" / "model_selection" / "result_schema_v2.json").read_text(encoding="utf-8")
    )
    result = json.loads(
        (repository / "results" / "model_selection_v2" / name).read_text(encoding="utf-8")
    )
    jsonschema.validate(result, schema)
