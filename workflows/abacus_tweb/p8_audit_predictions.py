#!/usr/bin/env python3
"""Audit P8 predictions versus sampling, graph support, and spatial residuals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import atomic_json


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
P5_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p5_graph_patch_adapter")
POINTS = Path(
    "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
    "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"
)


def model_paths(root: Path, model: str, rotation: int, seed: int):
    if model == "CLASSICAL-CIC":
        directory = root / "classical" / f"rotation_{rotation}"
        return (
            directory / "validation_parent_node_id.npy",
            directory / "cic_train_affine_eigenvalues.npy",
            directory / "cic_diagnostic_report.json",
        )
    slug = model[0].lower() + "_patch"
    directory = root / slug / f"rotation_{rotation}" / f"seed_{seed}"
    return (
        directory / "best_validation_parent_node_id.npy",
        directory / "best_validation_eigenvalues.npy",
        directory / "diagnostic_report.json",
    )


def binned_metrics(covariate: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> dict:
    finite = np.isfinite(covariate)
    edges = np.unique(np.quantile(covariate[finite], np.linspace(0.0, 1.0, 6)))
    if len(edges) < 3:
        return {"status": "insufficient covariate variation"}
    result = []
    for index in range(len(edges) - 1):
        selected = finite & (covariate >= edges[index])
        selected &= covariate <= edges[index + 1] if index == len(edges) - 2 else covariate < edges[index + 1]
        y, p = truth[selected, 0], prediction[selected, 0]
        result.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "n": int(selected.sum()),
            "r2_lambda1": float(r2_score(y, p)) if selected.sum() > 2 and np.var(y) > 0 else None,
            "mae_lambda1": float(np.mean(np.abs(p - y))) if selected.any() else None,
            "bias_lambda1": float(np.mean(p - y)) if selected.any() else None,
        })
    return {"binning": "validation-fold quintiles", "bins": result}


def moran_superblocks(
    residual: np.ndarray,
    xyz: np.ndarray,
    superblock: np.ndarray,
    cap: np.ndarray,
    neighbours: int = 8,
) -> dict:
    unique, inverse = np.unique(superblock, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    mean_residual = np.bincount(inverse, weights=residual) / count
    centroid = np.column_stack(
        [np.bincount(inverse, weights=xyz[:, axis]) / count for axis in range(3)]
    )
    block_cap = np.asarray([
        int(np.bincount(cap[inverse == index]).argmax()) for index in range(len(unique))
    ])
    centred = mean_residual - mean_residual.mean()
    numerator = 0.0
    total_weight = 0.0
    for cap_id in (0, 1):
        rows = np.flatnonzero(block_cap == cap_id)
        if len(rows) < 2:
            continue
        k = min(neighbours + 1, len(rows))
        _, near = cKDTree(centroid[rows]).query(centroid[rows], k=k)
        near = np.atleast_2d(near)
        for local, neighbour_local in enumerate(near):
            source = rows[local]
            for other_local in np.asarray(neighbour_local).reshape(-1)[1:]:
                target = rows[int(other_local)]
                numerator += centred[source] * centred[target]
                total_weight += 1.0
    denominator = float(np.sum(centred**2))
    value = (
        float(len(unique) / total_weight * numerator / denominator)
        if total_weight > 0 and denominator > 0
        else None
    )
    return {
        "statistic": "Moran I of mean lambda1 residual on 8-nearest superblocks within cap",
        "superblocks": int(len(unique)),
        "directed_adjacencies": int(total_weight),
        "moran_i": value,
    }


def strict_four_hop_parent(p5_root: Path, validation_core: np.ndarray, size: int) -> np.ndarray:
    offsets = np.load(p5_root / "core_active_offsets.npy", mmap_mode="r")
    parent = np.load(p5_root / "core_active_parent.npy", mmap_mode="r")
    eligible = np.load(p5_root / "core_active_eligible.npy", mmap_mode="r")
    safe = np.load(p5_root / "core_active_safe4hop.npy", mmap_mode="r")
    output = np.zeros(size, dtype=bool)
    for core_id in validation_core:
        start, stop = int(offsets[core_id]), int(offsets[core_id + 1])
        output[np.asarray(parent[start:stop][eligible[start:stop] & safe[start:stop]], dtype=np.int64)] = True
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("CLASSICAL-CIC", "G-PATCH", "U-PATCH", "F-PATCH"), required=True)
    parser.add_argument("--rotation", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--p5-root", type=Path, default=P5_ROOT)
    parser.add_argument("--points", type=Path, default=POINTS)
    args = parser.parse_args()
    parent_path, prediction_path, output = model_paths(
        args.p8_root, args.model, args.rotation, args.seed
    )
    parent = np.load(parent_path)
    prediction = np.load(prediction_path)
    truth_all = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    truth = np.asarray(truth_all[parent], dtype=np.float64)
    assignment = np.load(args.assignment, mmap_mode="r")
    active_parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    parent_to_active = np.full(len(truth_all), -1, dtype=np.int64)
    parent_to_active[active_parent] = np.arange(len(active_parent))
    active = parent_to_active[parent]
    if np.any(active < 0):
        raise RuntimeError("prediction parent is absent from P4 assignment")
    raw_graph = np.load(args.p5_root / "node_features.npy", mmap_mode="r")
    exposure = np.load(args.p8_root / "parent_exposure_apodized.npy", mmap_mode="r")
    points = np.load(args.points, mmap_mode="r")
    residual = np.asarray(prediction[:, 0] - truth[:, 0], dtype=np.float64)
    validation_core = np.load(
        args.p8_root / f"rotation_{args.rotation}" / "validation_core_id.npy"
    )
    strict = strict_four_hop_parent(args.p5_root, validation_core, len(truth_all))
    strict_selected = strict[parent]
    strict_shell = np.asarray(assignment["shell"][active], dtype=np.int8)
    strict_by_shell = {}
    for shell_id in range(4):
        selected = strict_selected & (strict_shell == shell_id)
        strict_by_shell[str(shell_id)] = {
            "n": int(selected.sum()),
            "retained_fraction": float(np.mean(strict_selected[strict_shell == shell_id])),
            "r2_lambda1": (
                float(r2_score(truth[selected, 0], prediction[selected, 0]))
                if selected.sum() > 2 and np.var(truth[selected, 0]) > 0 else None
            ),
        }
    payload = {
        "schema_version": 1,
        "model": args.model,
        "rotation": args.rotation,
        "seed": args.seed,
        "n": int(len(parent)),
        "covariates": {
            "graph_degree": binned_metrics(np.asarray(raw_graph[parent, 0]), truth, prediction),
            "graph_density": binned_metrics(np.asarray(raw_graph[parent, 2]), truth, prediction),
            "exposure_apodized": binned_metrics(np.asarray(exposure[parent]), truth, prediction),
            "fold_boundary_distance_mpc": binned_metrics(
                np.asarray(assignment["distance_to_conservative_fold_boundary_mpc"][active]),
                truth,
                prediction,
            ),
        },
        "residual_spatial_correlation": moran_superblocks(
            residual,
            np.asarray(points[parent, :3], dtype=np.float64),
            np.asarray(assignment["superblock_id"][active], dtype=np.int32),
            np.asarray(assignment["cap"][active], dtype=np.int8),
        ),
        "four_hop_isolation_diagnostic": {
            "primary_gate": False,
            "n": int(strict_selected.sum()),
            "retained_fraction": float(strict_selected.mean()),
            "by_shell": strict_by_shell,
        },
    }
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
