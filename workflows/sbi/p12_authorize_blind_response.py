#!/usr/bin/env python3
"""Freeze authorization for truth-free ph001 random-response construction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "p12a_candidate": "p12a-production-candidate-frozen-v1",
    "p12f_selection": "p12f-no-field-finalist-v1",
    "blind_input": "p10-phase-input-complete-v1",
}


def _resolved_artifact(path: str, authority_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    repository_candidate = REPO_ROOT / candidate
    if repository_candidate.exists():
        return repository_candidate
    return authority_path.parent / candidate


def validate_blind_authority(path: Path | None) -> dict:
    """Fail closed unless the posterior and field-selection decisions are frozen."""
    if path is None:
        raise PermissionError("ph001 P3b-R requires an explicit frozen blind authority")
    authority_path = Path(path)
    authority = json.loads(authority_path.read_text())
    if (
        authority.get("schema_version") != "p12-blind-response-build-authority-v1"
        or authority.get("phase") != "ph001"
        or authority.get("pass") is not True
        or authority.get("ph001_opened") is not False
        or authority.get("open_count") != 0
        or authority.get("truth_files_read") != []
    ):
        raise PermissionError("blind response-build authority is not sealed and valid")
    artifacts = authority.get("artifacts", {})
    for name, schema in EXPECTED.items():
        record = artifacts.get(name, {})
        artifact_path = _resolved_artifact(record.get("path", ""), authority_path)
        if not artifact_path.is_file() or sha256(artifact_path) != record.get("sha256"):
            raise PermissionError(f"blind authority artifact changed: {name}")
        payload = json.loads(artifact_path.read_text())
        if payload.get("schema_version") != schema or payload.get("pass") is not True:
            raise PermissionError(f"blind authority artifact is invalid: {name}")
        if name != "p12f_selection" and payload.get("truth_files_read", []):
            raise PermissionError(f"blind authority artifact is not truth-free: {name}")
        if payload.get("open_count", 0) != 0:
            raise PermissionError(f"blind authority artifact was opened: {name}")
        if payload.get("ph001_opened", payload.get("sealed_phase_opened", False)):
            raise PermissionError(f"blind authority artifact opened ph001: {name}")
    blind_input = json.loads(
        _resolved_artifact(artifacts["blind_input"]["path"], authority_path).read_text()
    )
    if (
        blind_input.get("phase") != "ph001"
        or blind_input.get("role") != "sealed_blind"
        or not all(blind_input.get("blind_gates", {}).values())
    ):
        raise PermissionError("ph001 observed-only input contract is not sealed")
    return authority


def build_authority(
    *,
    p12a_candidate: Path,
    p12f_selection: Path,
    blind_input: Path,
    output: Path,
) -> dict:
    paths = {
        "p12a_candidate": Path(p12a_candidate),
        "p12f_selection": Path(p12f_selection),
        "blind_input": Path(blind_input),
    }
    payloads = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for name, expected_schema in EXPECTED.items():
        payload = payloads[name]
        if payload.get("schema_version") != expected_schema or payload.get("pass") is not True:
            raise RuntimeError(f"cannot authorize blind response from invalid {name}")
        if name != "p12f_selection" and payload.get("truth_files_read", []):
            raise PermissionError(f"cannot authorize from non-blind {name}")
        if payload.get("open_count", 0) != 0:
            raise PermissionError(f"cannot authorize from opened {name}")
        if payload.get("ph001_opened", payload.get("sealed_phase_opened", False)):
            raise PermissionError(f"cannot authorize after ph001 opening: {name}")
    if (
        payloads["blind_input"].get("phase") != "ph001"
        or payloads["blind_input"].get("role") != "sealed_blind"
        or not all(payloads["blind_input"].get("blind_gates", {}).values())
    ):
        raise PermissionError("ph001 observed-only input contract is not sealed")
    marker = {
        "schema_version": "p12-blind-response-build-authority-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "phase": "ph001",
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "truth_files_read": [],
        "ph001_opened": False,
        "open_count": 0,
        "pass": True,
    }
    atomic_json(output, marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p12a-candidate", type=Path, required=True)
    parser.add_argument("--p12f-selection", type=Path, required=True)
    parser.add_argument("--blind-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_authority(
        p12a_candidate=args.p12a_candidate,
        p12f_selection=args.p12f_selection,
        blind_input=args.blind_input,
        output=args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
