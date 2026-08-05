#!/usr/bin/env python3
"""Preflight the hard lambda1 bound for the P8 U+CIC corrective model.

The bound is selected from training-fold residuals only.  Validation truth is then
used solely to ask whether a registered bound could possibly satisfy the existing
no-sparse-shell-degradation adoption gate, even with an oracle correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import (
    SHELL_NAMES,
    atomic_json,
    authoritative_mask,
    fold_roles,
)


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
ROTATIONS = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/rotations.json")
RECOVERY_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1")


def r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.sum((truth - truth.mean()) ** 2)
    return float(1.0 - np.sum((truth - prediction) ** 2) / denominator)


def best_unet_shell_scores(recovery_root: Path, rotation: int) -> dict[str, float]:
    path = (
        recovery_root
        / "convergence_extension_v1"
        / "unet"
        / f"rotation_{rotation}"
        / "seed_42"
        / "epoch_history.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    best = max(rows, key=lambda row: row["primary_macro_r2_lambda1"])
    return {
        "epoch": int(best["epoch"]),
        "primary_macro_r2_lambda1": float(best["primary_macro_r2_lambda1"]),
        "per_shell_lambda1_r2": {
            key: float(value) for key, value in best["per_shell_lambda1_r2"].items()
        },
    }


def indexed_anchor(p8_root: Path, rotation: int) -> tuple[np.ndarray, np.ndarray]:
    directory = p8_root / "classical" / f"rotation_{rotation}"
    parent = np.load(directory / "active_parent_node_id.npy", mmap_mode="r")
    eigen = np.load(
        directory / "cic_train_affine_active_eigenvalues.npy", mmap_mode="r"
    )
    order = np.argsort(parent)
    return np.asarray(parent)[order], np.asarray(eigen)[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--rotations", type=Path, default=ROTATIONS)
    parser.add_argument("--recovery-root", type=Path, default=RECOVERY_ROOT)
    parser.add_argument("--screen-rotations", type=int, nargs="+", default=(0, 2))
    parser.add_argument("--bounds", type=float, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--allowed-sparse-degradation", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=P8_ROOT / "classical" / "residual_bound_feasibility.json",
    )
    args = parser.parse_args()

    assignment = np.load(args.assignment, mmap_mode="r")
    truth = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    rotations = json.loads(args.rotations.read_text())
    authoritative = authoritative_mask(assignment)
    row_fold = np.asarray(assignment["fold"], dtype=np.int8)
    row_shell = np.asarray(assignment["shell"], dtype=np.int8)
    parent_id = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    report = {
        "schema_version": 1,
        "selection_policy": (
            "choose a common bound at or above the largest rotation-level training "
            "99th percentile; validation truth is used only for feasibility"
        ),
        "allowed_sparse_degradation": float(args.allowed_sparse_degradation),
        "rotations": {},
    }

    training_p99 = []
    for rotation in args.screen_rotations:
        train_folds, validation_fold, _ = fold_roles(rotations, rotation)
        anchor_parent, anchor_eigen = indexed_anchor(args.p8_root, rotation)
        sigma = float(
            json.loads(
                (args.p8_root / f"rotation_{rotation}" / "target_scaler.json").read_text()
            )["std"][0]
        )

        train_mask = authoritative & np.isin(row_fold, train_folds)
        train_parent = parent_id[train_mask]
        train_index = np.searchsorted(anchor_parent, train_parent)
        if np.any(train_index >= len(anchor_parent)) or np.any(
            anchor_parent[train_index] != train_parent
        ):
            raise RuntimeError(f"rotation {rotation} training anchor lookup failed")
        train_truth = np.asarray(truth[train_parent, 0], dtype=np.float64)
        train_cic = np.asarray(anchor_eigen[train_index, 0], dtype=np.float64)
        standardized = np.abs(train_truth - train_cic) / sigma
        quantiles = np.quantile(standardized, (0.5, 0.9, 0.95, 0.99, 0.999))
        training_p99.append(float(quantiles[3]))

        validation_mask = authoritative & (row_fold == validation_fold)
        validation_parent = parent_id[validation_mask]
        validation_index = np.searchsorted(anchor_parent, validation_parent)
        if np.any(validation_index >= len(anchor_parent)) or np.any(
            anchor_parent[validation_index] != validation_parent
        ):
            raise RuntimeError(f"rotation {rotation} validation anchor lookup failed")
        validation_truth = np.asarray(truth[validation_parent, 0], dtype=np.float64)
        validation_cic = np.asarray(anchor_eigen[validation_index, 0], dtype=np.float64)
        validation_shell = row_shell[validation_mask]
        unet = best_unet_shell_scores(args.recovery_root, rotation)

        bounds = {}
        for bound in args.bounds:
            oracle = validation_cic + np.clip(
                validation_truth - validation_cic, -bound * sigma, bound * sigma
            )
            shell_scores = {
                SHELL_NAMES[shell]: r2(
                    validation_truth[validation_shell == shell],
                    oracle[validation_shell == shell],
                )
                for shell in range(4)
            }
            sparse_floor = (
                unet["per_shell_lambda1_r2"][SHELL_NAMES[3]]
                - args.allowed_sparse_degradation
            )
            bounds[str(float(bound))] = {
                "oracle_per_shell_lambda1_r2": shell_scores,
                "oracle_sparse_shell_pass": bool(
                    shell_scores[SHELL_NAMES[3]] >= sparse_floor
                ),
                "required_sparse_shell_floor": float(sparse_floor),
            }

        report["rotations"][str(rotation)] = {
            "lambda1_training_sigma": sigma,
            "training_abs_residual_sigma_quantiles": {
                name: float(value)
                for name, value in zip(("q50", "q90", "q95", "q99", "q999"), quantiles)
            },
            "training_abs_residual_sigma_maximum": float(standardized.max()),
            "standalone_unet": unet,
            "bounds": bounds,
        }

    selected_bound = float(np.ceil(max(training_p99)))
    report["selected_common_lambda1_max_sigma"] = selected_bound
    report["selected_from_training_only"] = True
    report["v1_one_sigma_feasible"] = bool(
        all(
            row["bounds"]["1.0"]["oracle_sparse_shell_pass"]
            for row in report["rotations"].values()
        )
    )
    selected_key = str(selected_bound)
    report["selected_bound_feasible"] = bool(
        all(
            row["bounds"][selected_key]["oracle_sparse_shell_pass"]
            for row in report["rotations"].values()
        )
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
