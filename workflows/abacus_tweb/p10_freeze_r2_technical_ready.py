#!/usr/bin/env python3
"""Freeze R2 technical readiness without authorizing a science run."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    paths = {
        "assignment_policy": ROOT / "r2_assignment_response_v1/R2_ASSIGNMENT_POLICY_READY.json",
        "overlays": ROOT / "r2_assignment_response_v1/R2_ASSIGNMENT_OVERLAYS_READY.json",
        "loader": ROOT / "training_contract_r2_assignment/TRAINING_LOADER_READY.json",
        "loader_smoke": ROOT / "training_contract_r2_assignment/R2_LOADER_SMOKE.json",
        "throughput": ROOT / (
            "response_training/p10_r2_canary_1000_v1/unet/seed_42/"
            "THROUGHPUT_CANARY_REPORT.json"
        ),
    }
    records = {name: load(path) for name, path in paths.items()}
    for name, record in records.items():
        if not record.get("pass"):
            raise RuntimeError(f"{name} is not passing")
        if record.get("ph001_opened") or record.get("blind_phase_opened"):
            raise RuntimeError(f"{name} opened ph001")
    loader_hash = sha256(paths["loader"])
    if records["throughput"]["training_loader_sha256"] != loader_hash:
        raise RuntimeError("throughput canary did not use the current R2 loader")
    if int(records["throughput"]["patch_updates"]) != 1000:
        raise RuntimeError("R2 throughput canary did not complete 1,000 patches")

    output = ROOT / "training_contract_r2_assignment/P10_R2_TECHNICAL_READY.json"
    payload = {
        "schema_version": "p10-r2-technical-ready-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "patch_updates_per_second": records["throughput"]["patch_updates_per_second"],
        "ph001_opened": False,
        "technical_ready": True,
        "science_run_authorized": False,
        "science_blockers": [
            "freeze the final R0/R1 deterministic decision",
            "write P10_VIEW_LADDER_READY.json under the registered paired-view contract",
        ],
        "pass": True,
    }
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
