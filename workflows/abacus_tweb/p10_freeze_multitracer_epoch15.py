#!/usr/bin/env python3
"""Freeze the bounded P10 BRIGHT+FAINT Proxy/Null diagnostic at epoch 15.

This script never changes a checkpoint.  It validates that both matched runs
completed exactly the registered fifteen epochs, identifies the best validation
row inside that budget, and writes immutable terminal markers used by the
interactive supervisor.  It deliberately refuses an overshot epoch-16 history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_NAMES = {
    "proxy": "p10_bf_proxy_v1",
    "null": "p10_bf_null_v1",
}
REQUIRED_EPOCHS = 15
EXPECTED_CORES = 84_446


def sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def load_history(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, REQUIRED_EPOCHS + 1)):
        raise RuntimeError(
            f"{path} must contain exactly epochs 1..{REQUIRED_EPOCHS}; got {epochs}"
        )
    for row in rows:
        if not bool(row.get("complete_epoch_coverage")):
            raise RuntimeError(f"epoch {row['epoch']} lacks complete coverage")
        if int(row.get("unique_cores_seen", -1)) != EXPECTED_CORES:
            raise RuntimeError(f"epoch {row['epoch']} has the wrong core count")
        if int(row.get("repeat_cores", -1)) != 0:
            raise RuntimeError(f"epoch {row['epoch']} repeats cores")
    return rows


def freeze_run(run_root: Path, view: str) -> dict[str, Any]:
    output = run_root / RUN_NAMES[view] / "unet_multitracer" / "seed_42"
    marker = output / "EPOCH15_FROZEN.json"
    if marker.exists():
        payload = json.loads(marker.read_text())
        if not payload.get("pass"):
            raise RuntimeError(f"existing marker is not passing: {marker}")
        terminal = output / "ARM_A_TRAINING_COMPLETE.json"
        if not terminal.exists():
            atomic_json(terminal, {
                **payload,
                "status": "FROZEN_REGISTERED_EPOCH15",
                "terminal_marker_semantics": "bounded diagnostic, not a 20-epoch completion",
            })
        return payload

    history_path = output / "epoch_history.jsonl"
    checkpoint_path = output / "best_checkpoint.pt"
    report_path = output / "best_validation_report.json"
    cursor_checkpoint = output / "arm_a_checkpoint.pt"
    for path in (history_path, checkpoint_path, report_path, cursor_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    history = load_history(history_path)
    best = max(history, key=lambda row: float(row["primary_macro_r2_lambda1"]))
    report = json.loads(report_path.read_text())
    report_score = float(report["primary_macro_r2_lambda1"])
    if abs(report_score - float(best["primary_macro_r2_lambda1"])) > 1e-10:
        raise RuntimeError(
            f"best report score {report_score} does not match epoch history best "
            f"{best['primary_macro_r2_lambda1']}"
        )

    payload = {
        "schema_version": "p10-bf-epoch15-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "view": view,
        "run_name": RUN_NAMES[view],
        "registered_stop_epoch": REQUIRED_EPOCHS,
        "epochs_completed": REQUIRED_EPOCHS,
        "global_steps": int(history[-1]["global_step"]),
        "best_epoch": int(best["epoch"]),
        "best_primary_macro_r2_lambda1": float(best["primary_macro_r2_lambda1"]),
        "best_first_three_macro_r2_lambda1": float(
            best["diagnostic_first_three_shell_macro_r2_lambda1"]
        ),
        "best_per_shell_lambda1_r2": best["per_shell_lambda1_r2"],
        "history_path": str(history_path),
        "history_sha256": sha256(history_path),
        "best_checkpoint": str(checkpoint_path),
        "best_checkpoint_sha256": sha256(checkpoint_path),
        "best_validation_report": str(report_path),
        "best_validation_report_sha256": sha256(report_path),
        "cursor_checkpoint_sha256": sha256(cursor_checkpoint),
        "training_phases": ["ph000", "ph002", "ph003", "ph004", "ph005"],
        "validation_phase": "ph006",
        "sealed_phase_opened": False,
        "interpretation": "bounded multitracer information diagnostic; non-production",
        "pass": True,
    }
    atomic_json(marker, payload)
    # The legacy trainer and any already-running supervisor know only this
    # terminal filename.  Publishing the scientifically explicit freeze payload
    # here prevents a later invocation from resuming into epoch 16.
    atomic_json(output / "ARM_A_TRAINING_COMPLETE.json", {
        **payload,
        "status": "FROZEN_REGISTERED_EPOCH15",
        "terminal_marker_semantics": "bounded diagnostic, not a 20-epoch completion",
    })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/"
            "p12_and_multitracer_training"
        ),
    )
    parser.add_argument("--view", choices=tuple(RUN_NAMES))
    args = parser.parse_args()
    if args.view is not None:
        print(json.dumps(freeze_run(args.run_root, args.view), indent=2, sort_keys=True))
        return
    runs = {view: freeze_run(args.run_root, view) for view in RUN_NAMES}
    aggregate = {
        "schema_version": "p10-bf-epoch15-pair-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registered_stop_epoch": REQUIRED_EPOCHS,
        "runs": runs,
        "sealed_phase_opened": False,
        "pass": all(row["pass"] for row in runs.values()),
    }
    marker = args.run_root / "P10_BF_EPOCH15_FROZEN.json"
    atomic_json(marker, aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
