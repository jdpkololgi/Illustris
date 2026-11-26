#!/usr/bin/env python3
"""Export leakage-safe U-PATCH summaries for P12 posterior fitting.

The checkpoint must have omitted the exported phase from deterministic
training.  The exact 32-dimensional latent consumed by the point head, its
base prediction, ordered truth and deployable response covariates are written
in one parent-keyed shard. Reserved phases are refused using the bound policy;
historical contracts retain the legacy ph001 restriction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

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


def validate_export_phase(manifest: dict, phase: str) -> None:
    """Reject reserved/in-sample exports before loading any phase payload."""
    if not manifest.get("pass"):
        raise ValueError("Export requires a passing input contract")
    roles = manifest["roles"]
    if manifest.get("phase_policy"):
        from shared.phase_policy import check_roles
        check_roles(manifest, roles, omitted=manifest.get("out_of_fold_phase"))
    elif phase == "ph001":
        raise PermissionError("ph001 remains sealed in historical contracts")
    reserved = list(roles.get("sealed_blind_test", [])) + list(roles.get("reserved_confirmation", []))
    validation = roles["validation_and_selection"]
    validation = [validation] if isinstance(validation, str) else validation
    if phase in reserved or phase in roles["training"] or phase not in validation:
        raise PermissionError("Export phase must be non-reserved and out of training")


def validate_oof_checkpoint(checkpoint: dict, export_phase: str, latent_size: int) -> None:
    if checkpoint.get("schema_version") != "p10-arm-a-best-v1":
        raise RuntimeError("unsupported deterministic checkpoint schema")
    if checkpoint.get("model") != "unet":
        raise RuntimeError("P12 baseline requires a U-PATCH checkpoint")
    training_phases = tuple(checkpoint.get("training_phases", ()))
    if export_phase in training_phases:
        raise RuntimeError(f"{export_phase} is in-sample for this checkpoint")
    validation_phases = checkpoint.get("validation_phases") or [checkpoint.get("validation_phase")]
    if export_phase not in validation_phases:
        raise RuntimeError("export phase is not one of the checkpoint's registered validation phases")
    weight = checkpoint["state_dict"].get("unet.output.weight")
    if weight is None or int(weight.shape[0]) != latent_size:
        raise RuntimeError("checkpoint latent width does not match the P12 contract")


def terminal_export_checkpoint(state: dict, completion: dict, loader, epochs: int) -> dict:
    """Select fixed-budget terminal weights without looking at omitted scores."""
    args = state.get('frozen_arguments', {})
    if (state.get('schema_version') != 'p10-arm-a-checkpoint-v1' or state.get('model') != 'unet'
            or epochs < 1 or not args.get('disable_early_stopping')
            or args.get('epochs') != epochs or completion.get('epochs_completed') != epochs
            or state.get('global_step') != completion.get('global_steps')
            or state.get('global_step') != epochs * completion.get('training_cores_per_epoch', 0)
            or state.get('resume', {}).get('epoch') != epochs
            or state.get('resume', {}).get('cursor') != completion.get('training_cores_per_epoch')):
        raise RuntimeError('A complete fixed-budget terminal encoder is required')
    if state.get('source_contract') != completion.get('source_contract'):
        raise RuntimeError('Terminal checkpoint/completion source mismatch')
    if tuple(completion['training_phases']) != loader.training_phases:
        raise RuntimeError('Terminal encoder training phases mismatch')
    for key, relative in [('training_ready_marker_sha256', 'TRAINING_LOADER_READY.json'),
                          ('target_scaler_sha256','transforms/target_scaler.json'),
                          ('field_transform_sha256','transforms/field/field_transform.json')]:
        if completion[key] != sha256(loader.root / relative):
            raise RuntimeError('Terminal encoder input binding mismatch: ' + key)
    return {'schema_version':'p10-arm-a-best-v1', 'model':'unet',
            'state_dict':state['model_state'], 'training_phases':list(loader.training_phases),
            'validation_phases':list(loader.validation_phases),
            'scaler':loader.target_scaler, 'normalization':loader.field_normalization,
            'selection_policy':'terminal_fixed_budget', 'terminal_epochs':epochs}


def ntilde_at_rows(selection: dict, cap: np.ndarray, redshift: np.ndarray) -> np.ndarray:
    if selection.get("schema") == "p12a-matched-cic-selection-v1":
        spec = selection["selection"]
        edges = np.asarray(spec["bin_edges"], dtype=float)
        density = np.asarray(spec["density"], dtype=float)
        valid = np.asarray(spec["valid"], dtype=bool)
        c, z = np.asarray(cap), np.asarray(redshift)
        if not (selection.get("pass") and spec["density_unit"] == "Mpc^-3"
                and edges.ndim == 1 and np.all(np.diff(edges) > 0)
                and density.shape == valid.shape == (2, len(edges)-1)
                and c.shape == z.shape and c.dtype.kind in "iu"
                and np.isin(c, [0, 1]).all()):
            raise ValueError("Invalid corrected selection contract or cap")
        b = np.searchsorted(edges, z, side="right") - 1
        if not ((b >= 0) & (b < len(edges)-1)).all() or not valid[c,b].all():
            raise ValueError("Export rows outside supported selection bins")
        values = density[c,b]
        if not (np.isfinite(values) & (values >= 0)).all():
            raise ValueError("Invalid corrected selection density")
        return values.astype(np.float32)
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


def parent_to_assignment_index(assignment, n_parent: int) -> np.ndarray:
    """Map canonical parent IDs to rows in a P4 assignment archive.

    ``assignment`` is normally an ``NpzFile``.  Its ``len`` is the number of
    arrays in the archive, not the number of assignment rows, so row indices
    must be derived from ``parent_node_id`` itself.
    """
    parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    if parent.ndim != 1:
        raise RuntimeError("assignment parent_node_id must be one-dimensional")
    if np.any(parent < 0) or np.any(parent >= n_parent):
        raise RuntimeError("assignment parent_node_id lies outside the parent catalogue")
    if len(np.unique(parent)) != parent.size:
        raise RuntimeError("assignment parent_node_id must be unique")
    result = np.full(n_parent, -1, dtype=np.int64)
    result[parent] = np.arange(parent.size, dtype=np.int64)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-halo", type=int, default=24)
    parser.add_argument("--terminal-epochs", type=int)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--recover-ph006-job-61930825", action="store_true",
                        help="Finalize the saved predictions from the verified failed development export")
    args = parser.parse_args()
    if args.context_halo < 0:
        raise ValueError("context halo must be nonnegative")
    output = args.output_root / args.phase
    recover = args.recover_ph006_job_61930825
    if recover:
        base = Path("/pscratch/sd/d/dkololgi/abacus")
        if not (output.resolve() == base/"p12a_lowz_development_summaries_61930825/ph006"
                and args.contract_root.resolve() == base/"p12a_lowz_20261029_v1/training_contract_v1"
                and args.checkpoint.resolve() == base/"p12a_lowz_full_seed42_61548250/full/unet/seed_42/arm_a_checkpoint.pt"
                and args.terminal_epochs == 4 and args.context_halo == 48
                and not (output/"OOF_SUMMARY_COMPLETE.json").exists()):
            raise ValueError("Recovery inputs do not match original job")
    if not recover and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite export directory: {output}")
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.contract_root / "TRAINING_LOADER_READY.json").read_text())
    validate_export_phase(manifest, args.phase)
    if manifest.get("phase_policy") and (args.context_halo != 48 or args.terminal_epochs is None):
        raise ValueError("Shared campaign export requires halo48 and terminal epochs")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P12 summary export requires a GPU interactive allocation")

    loader = P10PhaseBalancedLoader(args.contract_root, include_blind=False)
    if args.phase not in loader.validation_phases:
        raise RuntimeError("contract validation phase is not the requested omitted phase")
    if loader.phase_records[args.phase].get('truth_lineage', {}).get('label_policy_version') == 'galaxy-realspace-all-populations-v2':
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch_load(args.checkpoint, args.device)
    completion_path = args.checkpoint.parent / 'ARM_A_TRAINING_COMPLETE.json'
    if args.terminal_epochs is not None:
        checkpoint = terminal_export_checkpoint(checkpoint, json.loads(completion_path.read_text()),
                                                loader, args.terminal_epochs)
    elif loader.manifest.get('checkpoint_policy'):
        raise RuntimeError('Expanded OOF contracts require --terminal-epochs; omitted-phase best selection is forbidden')
    validate_oof_checkpoint(checkpoint, args.phase, args.latent_channels)
    if tuple(checkpoint['training_phases']) != loader.training_phases:
        raise RuntimeError('Checkpoint training phases do not match its export contract')
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
    if recover:
        arrays = {key: np.load(output/path, mmap_mode="r") for key, path in
                  [("parent_node_id","parent_node_id.npy"), ("latent","latent_32d.npy"),
                   ("base_prediction","base_prediction_eigenvalues.npy")]}
        for key, shape in [("parent_node_id",(n_rows,)), ("latent",(n_rows,32)),
                           ("base_prediction",(n_rows,3))]:
            arr = arrays[key]
            if arr.shape != shape or not all(np.isfinite(arr[i:i+100000]).all()
                                             for i in range(0,n_rows,100000)):
                raise ValueError("Invalid saved array: " + key)
    else:
        arrays = {
        "parent_node_id": np.lib.format.open_memmap(output / "parent_node_id.npy", mode="w+", dtype=np.int64, shape=(n_rows,)),
        "latent": np.lib.format.open_memmap(output / "latent_32d.npy", mode="w+", dtype=np.float32, shape=(n_rows, args.latent_channels)),
        "base_prediction": np.lib.format.open_memmap(output / "base_prediction_eigenvalues.npy", mode="w+", dtype=np.float32, shape=(n_rows, 3)),
    }
    cursor = n_rows if recover else 0
    adapter = loader.field_adapter(args.phase)
    with torch.inference_mode():
        for ref in (() if recover else loader.validation_refs(args.phase)):
            patch = adapter.extract(
                ref.core_id,
                args.context_halo,
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
            if cursor == len(patch.authoritative_parent_id) or cursor // 100000 != (cursor-len(patch.authoritative_parent_id)) // 100000:
                print(json.dumps(dict(phase=args.phase, rows=cursor, total=n_rows, elapsed_seconds=time.monotonic()-started)), flush=True)
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
    if recover:
        if not np.array_equal(np.load(truth_path, mmap_mode="r"), truth):
            raise ValueError("Saved truth does not match recovered parent order")
    else:
        np.save(truth_path, truth, allow_pickle=False)
    parent_redshift = np.load(phase_root / "parent_redshift.npy", mmap_mode="r")
    redshift = np.asarray(parent_redshift[found_parent], dtype=np.float32)
    parent_to_assignment = parent_to_assignment_index(assignment, len(truth_by_parent))
    row = parent_to_assignment[found_parent]
    if np.any(row < 0):
        raise RuntimeError("exported parent lacks P4 assignment")
    cap = np.asarray(assignment["cap"][row], dtype=np.uint8)
    shell = np.asarray(assignment["shell"][row], dtype=np.int8)
    # Optional historical split diagnostic. The posterior's physical support
    # distance comes from the random catalogue through a separate lookup.
    split_diagnostic = {}
    if "distance_to_conservative_fold_boundary_mpc" in assignment:
        split_diagnostic["distance_to_fold_boundary_mpc"] = np.asarray(
            assignment["distance_to_conservative_fold_boundary_mpc"][row], dtype=np.float32
        )
    elif not loader.manifest.get("phase_policy"):
        raise KeyError("Historical assignment lacks its fold-boundary diagnostic")
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
        **split_diagnostic,
    )
    manifest = {
        "schema_version": "p12-unet-oof-summary-v1",
        "context_halo_voxels": args.context_halo,
        "alignment_voxels": unet_impl.ALIGNMENT_VOXELS,
        "precision": {"cudnn_tf32": torch.backends.cudnn.allow_tf32, "matmul_tf32": False,
                      "strict_deterministic": torch.are_deterministic_algorithms_enabled()},
        "export_source_sha256": sha256(Path(__file__)),
        "prediction_source_job": "61930825" if recover else None,
        "elapsed_seconds": time.monotonic() - started,
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
        "checkpoint_selection": checkpoint.get('selection_policy', 'validation_best'),
        "terminal_epochs": args.terminal_epochs,
        "completion_sha256": sha256(completion_path) if args.terminal_epochs is not None else None,
        "contract_marker_sha256": sha256(args.contract_root / 'TRAINING_LOADER_READY.json'),
        "truth_lineage": record.get('truth_lineage'),
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
