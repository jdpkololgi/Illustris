#!/usr/bin/env python3
"""Fit the P12-F G1 radial residual filter on registered training cores only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_train_unet_patch import CHANNELS, model_inputs
from workflows.sbi.p12f_gaussian_controls import (
    ConditionalGaussianUNet,
    finalize_residual_filter,
    residual_filter_accumulator,
    update_residual_filter_accumulator,
)
from workflows.sbi.p12f_train_conditional_field_flow import (
    FieldTargetStore,
    target_tensor,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f_matched_challengers_v1.json"
DEFAULT_CONTRACT = Path(
    "/global/homes/d/dkololgi/p11_contracts/"
    "training_contract_r1_random_repair_v2_20260901"
)
DEFAULT_PHASE_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract-root", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--phase-root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--gaussian-checkpoint", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("G1 filter fitting requires a compute GPU")
    config = json.loads(args.config.read_text())
    manifest = json.loads(args.run_manifest.read_text())
    checkpoint = torch.load(
        args.gaussian_checkpoint, map_location=args.device, weights_only=False
    )
    if (
        checkpoint.get("schema_version")
        != "p12f-matched-challenger-checkpoint-v1"
        or checkpoint.get("method") != "gaussian"
    ):
        raise RuntimeError("G1 filter requires a matched Gaussian checkpoint")
    if checkpoint.get("ph001_opened") or manifest.get("ph001_opened"):
        raise PermissionError("ph001 appeared in the Gaussian training provenance")
    if checkpoint.get("frozen_digest") != manifest.get("frozen_digest"):
        raise RuntimeError("Gaussian checkpoint/run manifest digest mismatch")
    phases = tuple(config["roles"]["training"])
    selected = {
        phase: np.asarray(
            manifest["frozen"]["selected_core_ids"][phase], dtype=np.int64
        )
        for phase in phases
    }
    expected = int(config["matched_contract"]["training_cores_per_phase"])
    if any(len(values) != expected for values in selected.values()):
        raise RuntimeError("G1 filter training-core count is not the matched contract")
    if any("ph001" in phase for phase in selected):
        raise PermissionError("ph001 entered G1 filter fitting")

    loader = P10RandomResponseLoader(args.contract_root, include_blind=False)
    store = FieldTargetStore(args.phase_root, phases)
    model = ConditionalGaussianUNet(
        condition_channels=3,
        base=int(config["matched_contract"]["unet_base"]),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    scaler = checkpoint["target_scaler"]
    normalization = loader.field_normalization
    parent = json.loads((REPO_ROOT / config["parent_flow_config"]).read_text())
    halo = int(parent["patch"]["context_halo_voxels"])
    alignment = int(parent["patch"]["alignment_voxels"])
    accumulator = residual_filter_accumulator(
        int(config["methods"]["gaussian_correlated_g1"]["radial_bins"])
    )
    supported_voxels = 0
    unsupported_voxels = 0
    with torch.inference_mode():
        ordinal = 0
        total = sum(len(values) for values in selected.values())
        for phase in phases:
            adapter = loader.field_adapter(phase)
            for core_id in selected[phase]:
                patch = adapter.extract(
                    int(core_id),
                    halo,
                    CHANNELS,
                    alignment_voxels=alignment,
                )
                condition, _ = model_inputs(
                    patch, normalization, args.device
                )
                target_data = store.extract(phase, patch)
                target = target_tensor(
                    target_data["delta"], scaler, args.device
                )
                mean, log_std = model(condition)
                normalized = (
                    (target - mean) / torch.exp(log_std)
                )[0, 0].detach().cpu().numpy()
                support = np.asarray(target_data["support"], dtype=bool)
                supported_voxels += int(support.sum())
                unsupported_voxels += int((~support).sum())
                normalized = np.where(support, normalized, 0.0)
                update_residual_filter_accumulator(
                    accumulator, normalized
                )
                ordinal += 1
                if ordinal == 1 or ordinal % 100 == 0 or ordinal == total:
                    print(
                        json.dumps(
                            {
                                "fields": ordinal,
                                "total": total,
                                "phase": phase,
                                "core_id": int(core_id),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    contract = finalize_residual_filter(accumulator)
    contract.update(
        {
            "created_utc": utc_now(),
            "checkpoint": str(args.gaussian_checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.gaussian_checkpoint),
            "run_manifest": str(args.run_manifest.resolve()),
            "run_manifest_sha256": sha256(args.run_manifest),
            "config_sha256": sha256(args.config),
            "training_ready_sha256": sha256(
                args.contract_root / "TRAINING_LOADER_READY.json"
            ),
            "target_scaler": scaler,
            "training_phases": list(phases),
            "training_core_ids": {
                phase: values.tolist() for phase, values in selected.items()
            },
            "supported_voxels": supported_voxels,
            "unsupported_voxels_zeroed": unsupported_voxels,
            "truth_files_read": [
                f"{phase} delta_R7 training targets" for phase in phases
            ],
            "validation_phase_read": False,
            "ph001_opened": False,
            "pass": True,
        }
    )
    atomic_json(args.output, contract)
    store.close()
    loader.close()
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
