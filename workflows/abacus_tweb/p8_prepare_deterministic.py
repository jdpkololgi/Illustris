#!/usr/bin/env python3
"""Freeze the deterministic P8 target, rotation, and sampling contracts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import fitsio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import (
    SHELL_NAMES,
    atomic_json,
    authoritative_mask,
    fit_target_scaler,
    fold_roles,
    sha256,
    shell_weights,
)


DEFAULT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
DEFAULT_TARGETS = Path(
    "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/"
    "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_"
    "ngrid2048_thr0p2_halo_xcom.fits"
)
DEFAULT_INDEX = Path(
    "/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/canonical_index.npz"
)
DEFAULT_ASSIGNMENT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz"
)
DEFAULT_CORES = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/cores.npz")
DEFAULT_ROTATIONS = Path(
    "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/rotations.json"
)
DEFAULT_CONTRACT = Path("docs/evidence/contracts/p8_target_metric_contract_v1.json")
P0_EVIDENCE = Path("docs/evidence/p0/evidence_freeze.json")


def write_parent_arrays(
    targets_path: Path,
    index_path: Path,
    output: Path,
) -> dict:
    eigen_path = output / "parent_eigenvalues.npy"
    z_path = output / "parent_redshift.npy"
    targetid_path = output / "parent_targetid.npy"
    index = np.load(index_path, mmap_mode="r")
    n = len(index["parent_node_id"])
    if not np.array_equal(np.asarray(index["parent_node_id"]), np.arange(n)):
        raise RuntimeError("P1b parent_node_id is not exact FITS-row identity")
    columns = fitsio.read(
        targets_path,
        columns=["TARGETID", "Z", "LAMBDA1", "LAMBDA2", "LAMBDA3"],
    )
    if len(columns) != n:
        raise RuntimeError(f"target FITS has {len(columns)} rows, P1b has {n}")
    targetid = np.asarray(columns["TARGETID"], dtype=np.int64)
    if not np.array_equal(targetid, np.asarray(index["targetid"], dtype=np.int64)):
        raise RuntimeError("target FITS TARGETID does not align with P1b canonical index")

    eigen = np.lib.format.open_memmap(
        eigen_path, mode="w+", dtype=np.float32, shape=(n, 3)
    )
    for column, name in enumerate(("LAMBDA1", "LAMBDA2", "LAMBDA3")):
        eigen[:, column] = np.asarray(columns[name], dtype=np.float32)
    eigen.flush()
    redshift = np.lib.format.open_memmap(z_path, mode="w+", dtype=np.float32, shape=(n,))
    redshift[:] = np.asarray(columns["Z"], dtype=np.float32)
    redshift.flush()
    saved_targetid = np.lib.format.open_memmap(
        targetid_path, mode="w+", dtype=np.int64, shape=(n,)
    )
    saved_targetid[:] = targetid
    saved_targetid.flush()
    del eigen, redshift, saved_targetid, columns
    return {
        "parent_rows": int(n),
        "parent_eigenvalues": str(eigen_path),
        "parent_eigenvalues_sha256": sha256(eigen_path),
        "parent_redshift": str(z_path),
        "parent_redshift_sha256": sha256(z_path),
        "parent_targetid": str(targetid_path),
        "parent_targetid_sha256": sha256(targetid_path),
    }


def prepare_rotation(
    rotation: int,
    *,
    output: Path,
    rotations: dict,
    assignment,
    cores,
    truth: np.ndarray,
) -> dict:
    train_folds, validation_fold, development_fold = fold_roles(rotations, rotation)
    auth = authoritative_mask(assignment)
    folds = np.asarray(assignment["fold"], dtype=np.int8)
    parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    shell = np.asarray(assignment["shell"], dtype=np.int8)
    train_rows = np.flatnonzero(auth & np.isin(folds, train_folds))
    validation_rows = np.flatnonzero(auth & (folds == validation_fold))
    development_rows = np.flatnonzero(auth & (folds == development_fold))
    scaler = fit_target_scaler(np.asarray(truth[parent[train_rows]], dtype=np.float64))
    weights, shell_count = shell_weights(shell[train_rows])

    row_weight = np.zeros(len(parent), dtype=np.float32)
    row_weight[train_rows] = weights
    core_weight = np.bincount(
        np.asarray(assignment["core_id"], dtype=np.int64),
        weights=row_weight,
        minlength=len(cores["core_id"]),
    ).astype(np.float64)
    core_fold = np.asarray(cores["fold"], dtype=np.int8)
    train_cores = np.flatnonzero(np.isin(core_fold, train_folds) & (core_weight > 0))
    validation_cores = np.flatnonzero(
        (core_fold == validation_fold) & (np.asarray(cores["active_count"]) > 0)
    )

    rotation_dir = output / f"rotation_{rotation}"
    rotation_dir.mkdir(parents=True, exist_ok=True)
    np.save(rotation_dir / "active_training_weight.npy", row_weight)
    np.save(rotation_dir / "training_core_id.npy", train_cores.astype(np.int32))
    np.save(rotation_dir / "training_core_weight.npy", core_weight[train_cores])
    np.save(rotation_dir / "validation_core_id.npy", validation_cores.astype(np.int32))
    target_scaler_path = rotation_dir / "target_scaler.json"
    atomic_json(target_scaler_path, scaler)
    role_path = rotation_dir / "roles.json"
    role = {
        "rotation": int(rotation),
        "train_folds": list(train_folds),
        "validation_fold": int(validation_fold),
        "development_test_fold": int(development_fold),
        "train_authoritative_rows": int(len(train_rows)),
        "validation_authoritative_rows": int(len(validation_rows)),
        "development_authoritative_rows": int(len(development_rows)),
        "training_shell_counts": shell_count,
        "training_cores": int(len(train_cores)),
        "validation_cores": int(len(validation_cores)),
        "patch_sampling": "probability proportional to sum of N_s^-0.5 core weights",
        "loss": "weighted mean over authoritative core rows only",
        "target_scaler": str(target_scaler_path),
        "target_scaler_sha256": sha256(target_scaler_path),
    }
    atomic_json(role_path, role)
    role["roles_sha256"] = sha256(role_path)
    return role


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--canonical-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--assignment", type=Path, default=DEFAULT_ASSIGNMENT)
    parser.add_argument("--cores", type=Path, default=DEFAULT_CORES)
    parser.add_argument("--rotations", type=Path, default=DEFAULT_ROTATIONS)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()

    started = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    parent = write_parent_arrays(args.targets, args.canonical_index, args.output_root)
    assignment = np.load(args.assignment, mmap_mode="r")
    cores = np.load(args.cores, mmap_mode="r")
    rotations = json.loads(args.rotations.read_text())
    truth = np.load(parent["parent_eigenvalues"], mmap_mode="r")
    index = np.load(args.canonical_index, mmap_mode="r")
    auth = authoritative_mask(assignment)
    active_parent = np.asarray(assignment["parent_node_id"][auth], dtype=np.int64)
    if not np.all(np.asarray(index["valid_target"][active_parent], dtype=bool)):
        raise RuntimeError("an authoritative P4 row lacks a valid target")
    active_truth = np.asarray(truth[active_parent], dtype=np.float64)
    if not np.all(np.isfinite(active_truth)):
        raise RuntimeError("non-finite target in authoritative P4 rows")
    if np.any(active_truth[:, 1] < active_truth[:, 0]) or np.any(
        active_truth[:, 2] < active_truth[:, 1]
    ):
        raise RuntimeError("canonical target ordering is broken")

    roles = {
        str(rotation): prepare_rotation(
            rotation,
            output=args.output_root,
            rotations=rotations,
            assignment=assignment,
            cores=cores,
            truth=truth,
        )
        for rotation in range(5)
    }
    p0 = json.loads(P0_EVIDENCE.read_text())
    manifest = {
        "schema_version": 1,
        "stage": "P8 deterministic target and rotation freeze",
        "status": "P8_TARGETS_READY",
        "target": {
            "representation": "linear increments",
            "definition": ["lambda1", "lambda2-lambda1", "lambda3-lambda2"],
            "inverse": "cumulative sum; no post-hoc sorting",
            "truth_ordering_verified": True,
            "epoch": "z=0.2 snapshot",
            "smoothing": "7 Mpc/h Gaussian",
        },
        "primary_metric": "mean spatial-fold equal-shell macro R2(lambda1)",
        "screen_rotations": [0, 2],
        "screen_rotation_reason": (
            "pre-registered non-adjacent validation folds 1 and 3; both contain "
            "both caps and all four shells"
        ),
        "parent_arrays": parent,
        "rotations": roles,
        "inputs": {
            "targets": str(args.targets),
            "targets_sha256": sha256(args.targets),
            "canonical_index": str(args.canonical_index),
            "canonical_index_sha256": sha256(args.canonical_index),
            "assignment": str(args.assignment),
            "assignment_sha256": sha256(args.assignment),
            "cores": str(args.cores),
            "cores_sha256": sha256(args.cores),
            "rotations": str(args.rotations),
            "rotations_sha256": sha256(args.rotations),
            "contract": str(args.contract),
            "contract_sha256": sha256(args.contract),
        },
        "classical_gate": {
            "role": "hard learned-model adoption baseline",
            "matched_fullcap_rows_required_before_model_ranking": True,
            "historical_p0_is_adoption_eligible": False,
            "historical_p0_reason": "different 219929-row wedge split, retained for context only",
            "historical_p0_methods": sorted(p0["methods"]),
            "expected_artifact": str(args.output_root / "classical" / "classical_summary.json"),
        },
        "gates": {
            "canonical_fits_alignment": True,
            "all_authoritative_targets_finite": True,
            "all_authoritative_targets_ordered": True,
            "five_rotations_frozen": True,
            "all_rotations_have_four_training_shells": all(
                set(row["training_shell_counts"]) == set(SHELL_NAMES)
                for row in roles.values()
            ),
            "screen_rotations_frozen_before_training": True,
        },
        "elapsed_seconds": time.time() - started,
    }
    manifest["pass"] = all(manifest["gates"].values())
    path = args.output_root / "prepared_manifest.json"
    atomic_json(path, manifest)
    (args.output_root / "P8_TARGETS_READY").write_text(
        f"prepared_manifest_sha256={sha256(path)}\n"
        f"screen_rotations=0,2\n"
        f"authoritative_rows={len(active_parent)}\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
