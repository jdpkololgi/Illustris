#!/usr/bin/env python3
"""Export leakage-safe U-PATCH summaries for P12 posterior fitting.

The checkpoint must have omitted the exported phase from deterministic
training.  The exact 32-dimensional latent consumed by the point head, its
base prediction, ordered truth and deployable response covariates are written
in one parent-keyed shard.  ph001 is refused unconditionally.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb import p8_train_unet_patch as unet_impl
from workflows.abacus_tweb.p8_deterministic_common import (
    increments_to_eigenvalues,
    sha256,
    unscale_increments,
)
from workflows.abacus_tweb.p8_train_patch_recovery import torch_load
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader, atomic_json


def validate_oof_checkpoint(checkpoint: dict, export_phase: str, latent_size: int) -> None:
    if checkpoint.get("schema_version") != "p10-arm-a-best-v1":
        raise RuntimeError("unsupported deterministic checkpoint schema")
    if checkpoint.get("model") != "unet":
        raise RuntimeError("P12 baseline requires a U-PATCH checkpoint")
    training_phases = tuple(checkpoint.get("training_phases", ()))
    if export_phase in training_phases:
        raise RuntimeError(f"{export_phase} is in-sample for this checkpoint")
    if checkpoint.get("validation_phase") != export_phase:
        raise RuntimeError("export phase is not the checkpoint's frozen omitted phase")
    weight = checkpoint["state_dict"].get("unet.output.weight")
    if weight is None or int(weight.shape[0]) != latent_size:
        raise RuntimeError("checkpoint latent width does not match the P12 contract")


def ntilde_at_rows(selection: dict, cap: np.ndarray, redshift: np.ndarray) -> np.ndarray:
    result = np.empty(len(redshift), dtype=np.float32)
    for cap_id, name in ((0, "SGC"), (1, "NGC")):
        chosen = cap == cap_id
        curve = selection["rotations"]["0"]["caps"][name]
        result[chosen] = np.interp(
            np.clip(redshift[chosen], curve["grid_z"][0], curve["grid_z"][-1]),
            curve["grid_z"],
            curve["ntilde"],
        ).astype(np.float32)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--latent-channels", type=int, default=32)
    args = parser.parse_args()
    if args.phase == "ph001":
        raise PermissionError("ph001 summaries remain sealed until the final pipeline is frozen")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P12 summary export requires a GPU interactive allocation")

    loader = P10PhaseBalancedLoader(args.contract_root, include_blind=False)
    if loader.validation_phase != args.phase:
        raise RuntimeError("contract validation phase is not the requested omitted phase")
    checkpoint = torch_load(args.checkpoint, args.device)
    validate_oof_checkpoint(checkpoint, args.phase, args.latent_channels)
    model = unet_impl.UPatch(base=args.base, latent_channels=args.latent_channels).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalization = checkpoint["normalization"]
    scaler = checkpoint["scaler"]
    phase_root = args.contract_root / "phases" / args.phase
    record = loader.phase_records[args.phase]
    assignment = np.load(Path(record["inputs"]["assignment"]), mmap_mode="r")
    authoritative = np.asarray(assignment["supervised_eligible"], dtype=bool)
    expected_parent = np.asarray(assignment["parent_node_id"][authoritative], dtype=np.int64)
    n_rows = len(expected_parent)
    output = args.output_root / args.phase
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "parent_node_id": np.lib.format.open_memmap(output / "parent_node_id.npy", mode="w+", dtype=np.int64, shape=(n_rows,)),
        "latent": np.lib.format.open_memmap(output / "latent_32d.npy", mode="w+", dtype=np.float32, shape=(n_rows, args.latent_channels)),
        "base_prediction": np.lib.format.open_memmap(output / "base_prediction_eigenvalues.npy", mode="w+", dtype=np.float32, shape=(n_rows, 3)),
    }
    cursor = 0
    adapter = loader.field_adapter(args.phase)
    with torch.inference_mode():
        for ref in loader.validation_refs():
            patch = adapter.extract(
                ref.core_id,
                unet_impl.HALO_VOXELS,
                unet_impl.CHANNELS,
                alignment_voxels=unet_impl.ALIGNMENT_VOXELS,
            )
            values, points = unet_impl.model_inputs(patch, normalization, args.device)
            latent = model.sample_latent(values, points)
            scaled = model.head(latent)
            eigen = increments_to_eigenvalues(
                unscale_increments(scaled.cpu().numpy(), scaler)
            ).astype(np.float32)
            stop = cursor + len(patch.authoritative_parent_id)
            arrays["parent_node_id"][cursor:stop] = patch.authoritative_parent_id
            arrays["latent"][cursor:stop] = latent.cpu().numpy().astype(np.float32)
            arrays["base_prediction"][cursor:stop] = eigen
            cursor = stop
    if cursor != n_rows:
        raise RuntimeError(f"summary export row count mismatch: {cursor} != {n_rows}")
    for array in arrays.values():
        array.flush()
    found_parent = np.asarray(arrays["parent_node_id"])
    if len(np.unique(found_parent)) != n_rows or not np.array_equal(
        np.sort(found_parent), np.sort(expected_parent)
    ):
        raise RuntimeError("summary parent set does not match authoritative omitted phase")

    truth_by_parent = loader.targets_by_parent(args.phase)
    truth = np.asarray(truth_by_parent[found_parent], dtype=np.float32)
    truth_path = output / "truth_eigenvalues.npy"
    np.save(truth_path, truth, allow_pickle=False)
    parent_redshift = np.load(phase_root / "parent_redshift.npy", mmap_mode="r")
    redshift = np.asarray(parent_redshift[found_parent], dtype=np.float32)
    parent_to_assignment = np.full(len(truth_by_parent), -1, dtype=np.int64)
    parent_to_assignment[np.asarray(assignment["parent_node_id"], dtype=np.int64)] = np.arange(len(assignment))
    row = parent_to_assignment[found_parent]
    if np.any(row < 0):
        raise RuntimeError("exported parent lacks P4 assignment")
    cap = np.asarray(assignment["cap"][row], dtype=np.uint8)
    shell = np.asarray(assignment["shell"][row], dtype=np.int8)
    boundary = np.asarray(
        assignment["distance_to_conservative_fold_boundary_mpc"][row], dtype=np.float32
    )
    selection = json.loads(
        (args.contract_root / "transforms/field/selection_manifest.json").read_text()
    )
    response_path = output / "response_covariates.npz"
    np.savez_compressed(
        response_path,
        redshift=redshift,
        ntilde_mpc3=ntilde_at_rows(selection, cap, redshift),
        cap=cap,
        shell=shell,
        distance_to_fold_boundary_mpc=boundary,
    )
    manifest = {
        "schema_version": "p12-unet-oof-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "training_phases": list(checkpoint["training_phases"]),
        "out_of_fold": True,
        "phase_is_conditioning_feature": False,
        "sealed_phase_opened": False,
        "rows": n_rows,
        "latent_dimensions": args.latent_channels,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "contract_root": str(args.contract_root),
        "arrays": {
            "parent_node_id": str(output / "parent_node_id.npy"),
            "latent": str(output / "latent_32d.npy"),
            "base_prediction": str(output / "base_prediction_eigenvalues.npy"),
            "truth": str(truth_path),
            "response": str(response_path),
        },
        "array_sha256": {
            name: sha256(Path(path))
            for name, path in {
                "parent_node_id": output / "parent_node_id.npy",
                "latent": output / "latent_32d.npy",
                "base_prediction": output / "base_prediction_eigenvalues.npy",
                "truth": truth_path,
                "response": response_path,
            }.items()
        },
        "pass": True,
    }
    atomic_json(output / "OOF_SUMMARY_COMPLETE.json", manifest)
    assignment.close()
    for field in loader._field.values():
        field.close()
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
