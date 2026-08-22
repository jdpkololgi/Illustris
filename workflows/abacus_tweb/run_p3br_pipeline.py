#!/usr/bin/env python3
"""Resume P3b-R maps, decision, overlays, and the R1 loader to completion."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")
CANARY = ("ph000", "ph006")


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(json.dumps({"command": command}) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def parallel(commands: list[tuple[list[str], Path]], workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, command, log): command for command, log in commands}
        for future in as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--map-workers", type=int, default=2)
    parser.add_argument("--overlay-workers", type=int, default=3)
    args = parser.parse_args()
    builder = REPO_ROOT / "workflows/abacus_tweb/p3br_build_random_response.py"
    preparer = REPO_ROOT / "workflows/abacus_tweb/p3br_prepare_r1_contract.py"
    exporter = REPO_ROOT / "workflows/abacus_tweb/p3br_export_evidence.py"
    logs = args.root / "p3br_logs"
    parallel([
        ([args.python, str(builder), "maps", "--root", str(args.root), "--phase", phase],
         logs / f"{phase}_canary_maps.log")
        for phase in CANARY
    ], args.map_workers)
    run(
        [args.python, str(builder), "decision", "--root", str(args.root)],
        logs / "random_density_decision.log",
    )
    decision_path = args.root / "training_contract/P3BR_RANDOM_DENSITY_DECISION.json"
    decision = json.loads(decision_path.read_text())
    n_random = int(decision["selected_realisation_count"])
    selected_snapshots = "1,4,18" if n_random == 18 else "1,4"
    parallel([
        ([
            args.python, str(builder), "maps", "--root", str(args.root),
            "--phase", phase, "--snapshots", selected_snapshots,
        ], logs / f"{phase}_selected_maps.log")
        for phase in PHASES if phase not in CANARY
    ], args.map_workers)
    parallel([
        ([
            args.python, str(builder), "overlay", "--root", str(args.root),
            "--phase", phase, "--decision", str(decision_path),
        ], logs / f"{phase}_overlay.log")
        for phase in PHASES
    ], args.overlay_workers)
    run(
        [
            args.python, str(preparer), "--root", str(args.root),
            "--base-contract", str(args.root / "training_contract"),
            "--output", str(args.root / "training_contract_r1_random"),
        ],
        logs / "r1_training_contract.log",
    )
    products = {}
    for phase in PHASES:
        manifest = args.root / phase / "p3b_random_response_v1/manifest.json"
        qa = args.root / phase / "p3b_random_response_v1/qa.json"
        row = json.loads(manifest.read_text())
        qa_row = json.loads(qa.read_text())
        if not row.get("pass") or not qa_row.get("pass"):
            raise RuntimeError(f"{phase} response product does not pass")
        products[phase] = {
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "qa": str(qa),
            "qa_sha256": sha256(qa),
        }
    contract = args.root / "training_contract_r1_random/TRAINING_LOADER_READY.json"
    contract_row = json.loads(contract.read_text())
    if not contract_row.get("pass") or contract_row.get("ph001_opened"):
        raise RuntimeError("R1 training contract is not ready or opened ph001")
    marker = {
        "schema_version": "p3br-pipeline-complete-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_random_ids": decision["selected_random_ids"],
        "decision": str(decision_path),
        "decision_sha256": sha256(decision_path),
        "products": products,
        "r1_training_contract": str(contract),
        "r1_training_contract_sha256": sha256(contract),
        "ph001_product_built": False,
        "ph001_opened": False,
        "pass": True,
    }
    atomic_json(args.root / "training_contract/P3BR_PIPELINE_COMPLETE.json", marker)
    run(
        [args.python, str(exporter), "--root", str(args.root)],
        logs / "tracked_evidence.log",
    )
    print(json.dumps(marker, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
