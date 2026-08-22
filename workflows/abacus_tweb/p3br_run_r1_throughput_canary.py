#!/usr/bin/env python3
"""Run and freeze the registered 1,000-patch P3b-R R1 throughput canary."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_canary(marker: dict, *, expected_updates: int) -> None:
    if not marker.get("pass"):
        raise RuntimeError("R1 technical canary did not pass")
    if int(marker.get("global_step", -1)) != expected_updates:
        raise RuntimeError(
            f"R1 canary stopped at {marker.get('global_step')} rather than "
            f"{expected_updates} updates"
        )
    if int(marker.get("cursor", -1)) != expected_updates:
        raise RuntimeError("fresh R1 canary cursor does not match its update budget")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract-root", type=Path, default=ROOT / "training_contract_r1_random"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "response_training")
    parser.add_argument("--run-name", default="p3br_r1_canary_1000_v1")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.updates != 1000:
        parser.error("the frozen P3b-R throughput gate is exactly 1,000 patches")

    ready_path = args.contract_root / "TRAINING_LOADER_READY.json"
    ready = load(ready_path)
    if not ready.get("pass") or ready.get("ph001_opened"):
        raise RuntimeError("R1 loader is not ready or opened ph001")

    run = args.output_root / args.run_name / "unet" / f"seed_{args.seed}"
    report_path = run / "THROUGHPUT_CANARY_REPORT.json"
    marker_path = run / "TECHNICAL_CANARY_COMPLETE.json"
    if report_path.is_file():
        report = load(report_path)
        if not report.get("pass"):
            raise RuntimeError("existing R1 throughput report is not passing")
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    trainer = REPO_ROOT / "workflows/abacus_tweb/p10_train_random_response.py"
    command = [
        sys.executable,
        str(trainer),
        "--model", "unet",
        "--contract-root", str(args.contract_root),
        "--seed", str(args.seed),
        "--epochs", "20",
        "--min-epochs", "10",
        "--disable-early-stopping",
        "--lr", "0.002",
        "--loss-log-every", "25",
        "--checkpoint-every", "250",
        "--stop-after-updates", str(args.updates),
        "--run-name", args.run_name,
        "--output-root", str(args.output_root),
        "--auto-resume",
    ]
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    subprocess.run(command, check=True)
    elapsed = time.monotonic() - started
    marker = load(marker_path)
    validate_canary(marker, expected_updates=args.updates)
    checkpoint = run / "arm_a_checkpoint.pt"
    loss_trace = run / "loss_trace.jsonl"
    report = {
        "schema_version": "p3br-r1-throughput-canary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started_utc,
        "model": "capacity-matched R1 U-PATCH",
        "device_contract": "one Perlmutter A100; no DDP",
        "patch_updates": args.updates,
        "wall_seconds": elapsed,
        "patch_updates_per_second": args.updates / elapsed,
        "technical_marker": str(marker_path),
        "technical_marker_sha256": sha256(marker_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "loss_trace": str(loss_trace),
        "loss_trace_sha256": sha256(loss_trace),
        "training_loader": str(ready_path),
        "training_loader_sha256": sha256(ready_path),
        "ph001_opened": False,
        "pass": True,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
