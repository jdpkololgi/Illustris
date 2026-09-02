#!/usr/bin/env python3
"""Checkpoint-only posterior-draw resolution audit for a frozen P12-F run.

This utility never changes the parent canary marker.  It reloads the frozen
checkpoint and validation-core identities, increases only the Monte Carlo draw
count, and writes a separately labelled diagnostic with full parent hashes.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_train_conditional_field_flow import (
    ConditionalVelocityUNet,
    DEFAULT_CONTRACT,
    DEFAULT_PHASE_ROOT,
    FieldTargetStore,
    evaluate,
    scientific_pass,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--contract-root", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--phase-root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_parent(
    run_dir: Path, manifest: dict, checkpoint: dict, config: dict
) -> None:
    if int(config["canary"]["posterior_draws"]) <= 0:
        raise RuntimeError("invalid frozen parent draw count")
    if config["roles"]["sealed_blind_test"] != "ph001":
        raise RuntimeError("unexpected blind-phase contract")
    if config["roles"]["validation_and_selection"] != "ph006":
        raise RuntimeError("checkpoint-only rescore is restricted to ph006")
    if manifest.get("ph001_opened") is not False:
        raise PermissionError("parent manifest does not seal ph001")
    if checkpoint.get("ph001_opened") is not False:
        raise PermissionError("parent checkpoint does not seal ph001")
    config_path = Path(manifest["config"])
    if manifest["config_sha256"] != sha256(config_path):
        raise RuntimeError("parent config hash drift")
    if checkpoint["config_sha256"] != manifest["config_sha256"]:
        raise RuntimeError("checkpoint/config hash mismatch")
    parent_report = run_dir / "P12F_CANARY_REPORT.json"
    if not parent_report.exists():
        raise RuntimeError("parent P12-F report is missing")
    report = json.loads(parent_report.read_text())
    if report.get("ph001_opened") is not False:
        raise PermissionError("parent report does not seal ph001")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    manifest = json.loads(manifest_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    validate_parent(run_dir, manifest, checkpoint, config)
    if args.draws <= int(config["canary"]["posterior_draws"]):
        raise ValueError("rescore draws must exceed the frozen parent draw count")

    rescore_config = copy.deepcopy(config)
    rescore_config["canary"]["posterior_draws"] = int(args.draws)
    output = run_dir / f"rescore_draws_{args.draws}"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"non-empty rescore output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    loader = P10RandomResponseLoader(args.contract_root, include_blind=False)
    visible = tuple(config["roles"]["training"]) + ("ph006",)
    store = FieldTargetStore(args.phase_root, visible)
    try:
        model = ConditionalVelocityUNet(
            condition_channels=3, base=int(config["model"]["unet_base"])
        ).to(args.device)
        model.load_state_dict(checkpoint["model"])
        validation_ids = np.asarray(
            manifest["selected_core_ids"]["ph006"], dtype=np.int64
        )
        report, arrays = evaluate(
            model,
            loader,
            store,
            validation_ids,
            checkpoint["target_scaler"],
            rescore_config,
            args.device,
        )
    finally:
        store.close()
        loader.close()

    passed, reasons = scientific_pass(report, config)
    report.update(
        {
            "schema_version": "p12f-field-flow-draw-resolution-rescore-v1",
            "rescore_created_utc": utc_now(),
            "parent_run": str(run_dir),
            "parent_manifest_sha256": sha256(manifest_path),
            "parent_checkpoint_sha256": sha256(checkpoint_path),
            "parent_report_sha256": sha256(run_dir / "P12F_CANARY_REPORT.json"),
            "frozen_parent_draws": int(config["canary"]["posterior_draws"]),
            "rescore_draws": int(args.draws),
            "same_validation_cores": True,
            "same_checkpoint": True,
            "same_physics_and_gates": True,
            "rescore_gate_pass": bool(passed),
            "rescore_gate_failure_reasons": reasons,
            "parent_marker_unchanged": True,
            "ph001_opened": False,
        }
    )
    arrays_path = output / "ph006_posterior_samples.npz"
    np.savez_compressed(arrays_path, **arrays)
    report["posterior_samples"] = str(arrays_path.resolve())
    report["posterior_samples_sha256"] = sha256(arrays_path)
    atomic_json(output / "P12F_RESCORE_REPORT.json", report)
    marker = {
        "schema_version": "p12f-field-flow-draw-resolution-rescore-complete-v1",
        "created_utc": utc_now(),
        "rescore_gate_pass": bool(passed),
        "failure_reasons": reasons,
        "draws": int(args.draws),
        "report": str((output / "P12F_RESCORE_REPORT.json").resolve()),
        "report_sha256": sha256(output / "P12F_RESCORE_REPORT.json"),
        "parent_marker_unchanged": True,
        "ph001_opened": False,
    }
    atomic_json(output / "P12F_RESCORE_COMPLETE.json", marker)
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
