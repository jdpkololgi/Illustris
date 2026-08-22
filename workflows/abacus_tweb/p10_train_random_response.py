#!/usr/bin/env python3
"""Capacity-matched P10 R1 trainer using the stored P3b-R response view.

This wrapper deliberately leaves ``p10_train_arm_a.py`` byte-for-byte unchanged
while outstanding P12 cross-fits still resume against its frozen source hash.
Only the loader class and response-view provenance are replaced here.
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb import p10_train_arm_a as trainer
from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import sha256


def requested_model(arguments: list[str]) -> str | None:
    for index, value in enumerate(arguments):
        if value == "--model" and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith("--model="):
            return value.split("=", 1)[1]
    return None


if requested_model(sys.argv[1:]) not in (None, "unet"):
    raise SystemExit("P3b-R R1 is registered only for the capacity-matched U-PATCH")

trainer.P10PhaseBalancedLoader = P10RandomResponseLoader
_source_contract = trainer.source_contract
_atomic_json = trainer.atomic_json


def response_source_contract() -> dict[str, str]:
    result = _source_contract()
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "workflows/abacus_tweb/p3br_training_contract.py",
    ):
        result[str(path.relative_to(REPO_ROOT))] = sha256(path)
    return result


def response_atomic_json(path: Path, payload: dict) -> None:
    if (
        Path(path).name == "run_manifest.json"
        or payload.get("stage") == "P10 deterministic multi-phase Arm A final-view R0"
    ):
        payload = dict(payload)
        payload["schema_version"] = "p10-p3br-r1-run-v1"
        payload["stage"] = "P10 deterministic multi-phase random-response R1"
        payload["view"] = (
            "R1 capacity-matched: BRIGHT counts, random-support apodization, "
            "random-derived BRIGHT log-count ratio"
        )
        payload["response_scope"] = (
            "random M/angular targetability only; full C_fibre and C_z are not claimed"
        )
    _atomic_json(path, payload)


trainer.source_contract = response_source_contract
trainer.atomic_json = response_atomic_json


if __name__ == "__main__":
    trainer.main()
