#!/usr/bin/env python3
"""Idempotently finish the frozen P8.9 D0 downstream evaluation.

This driver never trains.  It owns the complete stitch -> global FFT evaluation
-> learned-context chain with one filesystem lock, resumes an interrupted
stitch, and skips a stage only after validating its completion artifact against
the selected epoch-16 checkpoint or stitched-field manifest.
"""
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

from workflows.abacus_tweb.p8_deterministic_common import (
    acquire_run_lock,
    atomic_json,
    sha256,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_density_phys_v1")
RUN = ROOT / "d0_runs/rotation_0/seed_42/scientific_v1"
STITCHED = ROOT / "d0_stitched/rotation_0/seed_42"
EVALUATION = ROOT / "d0_evaluation/rotation_0/seed_42"
CONTEXT = ROOT / "d0_learned_context/rotation_0/seed_42"
OUTPUT = ROOT / "d0_downstream/rotation_0/seed_42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--stitched", type=Path, default=STITCHED)
    parser.add_argument("--evaluation", type=Path, default=EVALUATION)
    parser.add_argument("--context", type=Path, default=CONTEXT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def validate_training(run: Path) -> dict:
    summary_path = run / "training_summary.json"
    marker = run / "D0_TRAINING_SCHEDULE_COMPLETE"
    checkpoint = run / "best_checkpoint.pt"
    if not marker.exists() or not summary_path.exists() or not checkpoint.exists():
        raise RuntimeError("P8.9 scientific training is not complete")
    summary = load_json(summary_path)
    if summary.get("status") != "D0_TRAINING_SCHEDULE_COMPLETE":
        raise RuntimeError("P8.9 training summary is not complete")
    if int(summary.get("epochs_completed", 0)) != 20:
        raise RuntimeError("P8.9 training did not complete the frozen 20 epochs")
    if int(summary.get("best_epoch", -1)) != 16:
        raise RuntimeError("selected P8.9 checkpoint is not the registered epoch 16")
    return {
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "best_epoch": int(summary["best_epoch"]),
        "best_macro_shell_r2_delta_r7": float(
            summary["best_macro_shell_r2_delta_r7"]
        ),
    }


def stitched_complete(stitched: Path, checkpoint_sha: str) -> bool:
    marker = stitched / "D0_STITCHED_FIELD_READY"
    manifest_path = stitched / "stitched_field_manifest.json"
    if not marker.exists() or not manifest_path.exists():
        return False
    manifest = load_json(manifest_path)
    if manifest.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("completed stitched field belongs to a different checkpoint")
    if manifest.get("status") != "PASS":
        raise RuntimeError("stitched field manifest is not PASS")
    for row in manifest.get("support_coverage", {}).values():
        if float(row.get("coverage_fraction", 0.0)) != 1.0:
            raise RuntimeError("completed stitched field has incomplete science support")
    return True


def evaluation_complete(evaluation: Path, stitched_manifest_sha: str) -> bool:
    marker = evaluation / "D0_FIELD_DOWNSTREAM_EVALUATED"
    report_path = evaluation / "field_downstream_metrics.json"
    if not marker.exists() or not report_path.exists():
        return False
    report = load_json(report_path)
    if report.get("status") != "PASS":
        raise RuntimeError("downstream field evaluation is not PASS")
    recorded = report.get("inputs", {}).get("stitched_manifest_sha256")
    if recorded != stitched_manifest_sha:
        raise RuntimeError("downstream evaluation belongs to a different stitched field")
    return True


def context_complete(context: Path, stitched_manifest_sha: str) -> bool:
    marker = context / "D0_LEARNED_CONTEXT_COMPLETE"
    report_path = context / "learned_context_report.json"
    if not marker.exists() or not report_path.exists():
        return False
    report = load_json(report_path)
    if report.get("status") != "PASS":
        raise RuntimeError("learned-context report is not PASS")
    recorded = report.get("inputs", {}).get("stitched_manifest_sha256")
    if recorded != stitched_manifest_sha:
        raise RuntimeError("learned-context result belongs to a different stitched field")
    return True


def run_stage(command: list[str], name: str) -> dict:
    started = time.time()
    print(json.dumps({"stage": name, "command": command}), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"P8.9 downstream stage {name} failed: {completed.returncode}")
    return {"status": "completed", "elapsed_seconds": float(time.time() - started)}


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("P8.9 downstream evaluation requires interactive CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_lock(
        args.output / ".downstream.lock",
        purpose="P8.9 stitch, global FFT evaluation, and learned context",
    )
    started = time.time()
    training = validate_training(args.run)
    stages = {}
    checkpoint = Path(training["checkpoint"])

    if stitched_complete(args.stitched, training["checkpoint_sha256"]):
        stages["stitch"] = {"status": "reused_hash_validated"}
    else:
        stages["stitch"] = run_stage([
            str(args.python), "-u", "-m",
            "workflows.abacus_tweb.p8_infer_stitched_density",
            "--checkpoint", str(checkpoint),
            "--output", str(args.stitched),
            "--device", args.device,
            "--resume",
            "--progress-every", str(args.progress_every),
        ], "stitch")
    if not stitched_complete(args.stitched, training["checkpoint_sha256"]):
        raise RuntimeError("stitch stage exited without a valid completion contract")
    stitched_manifest = args.stitched / "stitched_field_manifest.json"
    stitched_sha = sha256(stitched_manifest)

    if evaluation_complete(args.evaluation, stitched_sha):
        stages["evaluation"] = {"status": "reused_hash_validated"}
    else:
        stages["evaluation"] = run_stage([
            str(args.python), "-u", "-m",
            "workflows.abacus_tweb.p8_evaluate_stitched_density",
            "--stitched", str(args.stitched),
            "--output", str(args.evaluation),
            "--device", args.device,
        ], "evaluation")
    if not evaluation_complete(args.evaluation, stitched_sha):
        raise RuntimeError("evaluation stage exited without a valid completion contract")

    if context_complete(args.context, stitched_sha):
        stages["learned_context"] = {"status": "reused_hash_validated"}
    else:
        stages["learned_context"] = run_stage([
            str(args.python), "-u", "-m",
            "workflows.abacus_tweb.p8_learned_field_context",
            "--stitched", str(args.stitched),
            "--output", str(args.context),
            "--device", args.device,
        ], "learned_context")
    if not context_complete(args.context, stitched_sha):
        raise RuntimeError("context stage exited without a valid completion contract")

    artifacts = {
        "stitched_manifest": str(stitched_manifest),
        "stitched_manifest_sha256": stitched_sha,
        "field_downstream_metrics": str(
            args.evaluation / "field_downstream_metrics.json"
        ),
        "field_downstream_metrics_sha256": sha256(
            args.evaluation / "field_downstream_metrics.json"
        ),
        "learned_context_report": str(args.context / "learned_context_report.json"),
        "learned_context_report_sha256": sha256(
            args.context / "learned_context_report.json"
        ),
    }
    report = {
        "schema_version": "p8-density-downstream-run-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_sha(),
        "status": "PASS",
        "model": "U-DENSITY-PHYS-v1",
        "rotation": 0,
        "seed": 42,
        "training": training,
        "stages": stages,
        "artifacts": artifacts,
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_json(args.output / "downstream_run_manifest.json", report)
    (args.output / "D0_DOWNSTREAM_COMPLETE").write_text(
        f"checkpoint={training['checkpoint_sha256']} stitched={stitched_sha}\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    run_lock.close()


if __name__ == "__main__":
    main()
