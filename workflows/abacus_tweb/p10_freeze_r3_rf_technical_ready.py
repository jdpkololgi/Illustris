#!/usr/bin/env python3
"""Freeze P10 R3-RF technical readiness and measured 1,000-patch throughput."""
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
    contract = ROOT / "training_contract_r3_random_field"
    run = ROOT / "response_training/p10_r3_rf_canary_1000_v1/unet/seed_42"
    paths = {
        "products": ROOT / "r3_random_field_v1/R3_RANDOM_FIELD_PRODUCTS_READY.json",
        "loader": contract / "TRAINING_LOADER_READY.json",
        "loader_smoke": contract / "R3_RF_LOADER_SMOKE.json",
        "run_manifest": run / "run_manifest.json",
        "canary_marker": run / "TECHNICAL_CANARY_COMPLETE.json",
        "checkpoint": run / "arm_a_checkpoint.pt",
        "loss_trace": run / "loss_trace.jsonl",
    }
    records = {name: load(path) for name, path in paths.items() if path.suffix == ".json"}
    for name in ("products", "loader", "loader_smoke", "canary_marker"):
        record = records[name]
        if not record.get("pass"):
            raise RuntimeError(f"{name} does not pass")
        if record.get("ph001_opened") or record.get("ph001_product_built"):
            raise RuntimeError(f"{name} opened or built ph001")
    canary = records["canary_marker"]
    if int(canary.get("global_step", -1)) != 1000:
        raise RuntimeError("R3-RF canary did not complete exactly 1,000 updates")
    elapsed = paths["canary_marker"].stat().st_mtime - paths["run_manifest"].stat().st_mtime
    if elapsed <= 0:
        raise RuntimeError("invalid R3-RF canary wall time")
    report = {
        "schema_version": "p10-r3-rf-throughput-canary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "R3-RF six-channel U-PATCH",
        "device_contract": "one Perlmutter A100; no DDP",
        "patch_updates": 1000,
        "wall_seconds_from_artifact_mtime": elapsed,
        "patch_updates_per_second": 1000.0 / elapsed,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "ph001_opened": False,
        "pass": True,
    }
    report_path = run / "THROUGHPUT_CANARY_REPORT.json"
    atomic_json(report_path, report)
    technical = {
        "schema_version": "p10-r3-rf-technical-ready-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "products": {"path": str(paths["products"]), "sha256": sha256(paths["products"])},
            "loader": {"path": str(paths["loader"]), "sha256": sha256(paths["loader"])},
            "loader_smoke": {
                "path": str(paths["loader_smoke"]),
                "sha256": sha256(paths["loader_smoke"]),
            },
            "throughput": {"path": str(report_path), "sha256": sha256(report_path)},
        },
        "patch_updates_per_second": report["patch_updates_per_second"],
        "ph001_opened": False,
        "technical_ready": True,
        "science_run_authorized": True,
        "science_contract": (
            "priority high-S/N uncompressed random-response arm; matched R1/R2/BF "
            "optimizer-update and ph006 evaluation contract"
        ),
        "pass": True,
    }
    output = contract / "P10_R3_RF_TECHNICAL_READY.json"
    atomic_json(output, technical)
    print(json.dumps(technical, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

