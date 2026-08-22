#!/usr/bin/env python3
"""Publish compact, hash-verified P3b-R evidence into the Git repository.

Large HDF5 response overlays remain on pscratch.  This exporter copies only the
scientific decision, per-phase manifests/QA, and frozen R1 loader metadata after
the complete pipeline marker has verified every visible phase.  The sealed
ph001 phase is rejected by construction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def copy_verified(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    destination_hash = sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"evidence copy hash mismatch: {source} -> {destination}")
    return {
        "source": str(source),
        "source_sha256": source_hash,
        "tracked": str(destination.relative_to(REPO_ROOT)),
        "tracked_sha256": destination_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "docs/evidence/p3br"
    )
    args = parser.parse_args()

    complete_path = args.root / "training_contract/P3BR_PIPELINE_COMPLETE.json"
    complete = load(complete_path)
    if not complete.get("pass") or complete.get("ph001_opened"):
        raise RuntimeError("P3b-R pipeline is not complete or opened ph001")
    expected = set(PHASES)
    if set(complete.get("products", {})) != expected:
        raise RuntimeError("P3b-R complete marker does not contain the visible phases")

    sources = {
        "random_density_decision": args.root
        / "training_contract/P3BR_RANDOM_DENSITY_DECISION.json",
        "pipeline_complete": complete_path,
        "r1_training_loader": args.root
        / "training_contract_r1_random/TRAINING_LOADER_READY.json",
        "r1_adapter_inventory": args.root
        / "training_contract_r1_random/adapter_inventory.json",
        "r1_field_transform": args.root
        / "training_contract_r1_random/transforms/field/field_transform.json",
    }
    for phase in PHASES:
        response = args.root / phase / "p3b_random_response_v1"
        sources[f"{phase}_manifest"] = response / "manifest.json"
        sources[f"{phase}_qa"] = response / "qa.json"

    records = {}
    for label, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = load(source)
        if label.endswith(("_manifest", "_qa")) and not payload.get("pass"):
            raise RuntimeError(f"non-passing evidence source: {source}")
        if payload.get("ph001_opened") or payload.get("sealed_phase_opened"):
            raise RuntimeError(f"blind phase contamination in {source}")
        records[label] = copy_verified(source, args.output / f"{label}.json")

    evidence = {
        "schema_version": "p3br-tracked-evidence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_root": str(args.root),
        "visible_phases": list(PHASES),
        "blind_phase": "ph001",
        "ph001_product_built": False,
        "ph001_opened": False,
        "records": records,
        "pass": True,
    }
    atomic_json(args.output / "evidence_manifest.json", evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
