#!/usr/bin/env python3
"""Fit and materialise the frozen training-only G-PATCH feature transforms."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
import sys

import numpy as np
from sklearn.preprocessing import PowerTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    authoritative_mask,
    fold_roles,
    sha256,
)


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
P5_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p5_graph_patch_adapter")
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
ROTATIONS = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/rotations.json")
SELECTION = Path(
    "/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/"
    "fullcap_selection_v1/selection_manifest.json"
)
POINTS = Path(
    "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
    "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"
)
SI_COLUMNS = (0, 2, 3, 4, 5, 6)


def transform_si(values: np.ndarray, cap: np.ndarray, medians: dict) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    for cap_id in (0, 1):
        selected = cap == cap_id
        for column in SI_COLUMNS:
            output[selected, column] /= max(
                float(medians[str(cap_id)][str(column)]), 1e-9
            )
    return output


def curve_values(redshift: np.ndarray, cap: np.ndarray, selection: dict, rotation: int) -> np.ndarray:
    result = np.empty(len(redshift), dtype=np.float64)
    row = selection["rotations"][str(rotation)]["caps"]
    for cap_id, name in ((0, "SGC"), (1, "NGC")):
        selected = cap == cap_id
        grid_z = np.asarray(row[name]["grid_z"], dtype=np.float64)
        ntilde = np.asarray(row[name]["ntilde"], dtype=np.float64)
        result[selected] = np.interp(
            np.clip(redshift[selected], grid_z[0], grid_z[-1]), grid_z, ntilde
        )
    return result


def edge_fit(
    *,
    pairs: np.ndarray,
    attrs: np.ndarray,
    parent_train: np.ndarray,
    parent_cap: np.ndarray,
    output: Path,
    chunk: int,
) -> dict:
    counts = {0: 0, 1: 0}
    for start in range(0, len(pairs), chunk):
        stop = min(start + chunk, len(pairs))
        block = np.asarray(pairs[start:stop], dtype=np.int64)
        internal = parent_train[block[:, 0]] & parent_train[block[:, 1]]
        for cap_id in (0, 1):
            counts[cap_id] += int(
                np.sum(internal & (parent_cap[block[:, 0]] == cap_id))
            )
    length_paths = {}
    length_arrays = {}
    cursors = {0: 0, 1: 0}
    for cap_id in (0, 1):
        path = output / f"edge_length_training_cap{cap_id}.npy"
        length_paths[cap_id] = path
        length_arrays[cap_id] = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(counts[cap_id],)
        )
    for start in range(0, len(pairs), chunk):
        stop = min(start + chunk, len(pairs))
        block = np.asarray(pairs[start:stop], dtype=np.int64)
        edge = np.asarray(attrs[start:stop], dtype=np.float32)
        internal = parent_train[block[:, 0]] & parent_train[block[:, 1]]
        for cap_id in (0, 1):
            selected = internal & (parent_cap[block[:, 0]] == cap_id)
            n = int(selected.sum())
            length_arrays[cap_id][cursors[cap_id] : cursors[cap_id] + n] = edge[selected, 0]
            cursors[cap_id] += n
    result = {}
    for cap_id in (0, 1):
        length_arrays[cap_id].flush()
        median = float(np.median(length_arrays[cap_id]))
        result[str(cap_id)] = {
            "training_internal_undirected_edges": counts[cap_id],
            "edge_length_si_median": median,
            "temporary_exact_median_array": str(length_paths[cap_id]),
        }

    accum = {
        cap_id: {"n": 0, "length_sum": 0.0, "length_sum2": 0.0,
                 "density_sum": 0.0, "density_sum2": 0.0}
        for cap_id in (0, 1)
    }
    for start in range(0, len(pairs), chunk):
        stop = min(start + chunk, len(pairs))
        block = np.asarray(pairs[start:stop], dtype=np.int64)
        edge = np.asarray(attrs[start:stop], dtype=np.float64)
        internal = parent_train[block[:, 0]] & parent_train[block[:, 1]]
        for cap_id in (0, 1):
            selected = internal & (parent_cap[block[:, 0]] == cap_id)
            log_length = np.log(
                np.maximum(edge[selected, 0] / result[str(cap_id)]["edge_length_si_median"], 1e-6)
            )
            log_density = np.log(np.maximum(edge[selected, 4], 1e-6))
            # The production graph is bidirectional. Length is duplicated; reverse
            # density contrast is the reciprocal after the registered clamp.
            reverse_density = np.log(1.0 / np.maximum(edge[selected, 4], 1e-6))
            row = accum[cap_id]
            row["n"] += 2 * len(log_length)
            row["length_sum"] += 2.0 * float(log_length.sum(dtype=np.float64))
            row["length_sum2"] += 2.0 * float(np.square(log_length).sum(dtype=np.float64))
            both_density = np.concatenate((log_density, reverse_density))
            row["density_sum"] += float(both_density.sum(dtype=np.float64))
            row["density_sum2"] += float(np.square(both_density).sum(dtype=np.float64))
    for cap_id in (0, 1):
        row = accum[cap_id]
        n = max(row["n"], 1)
        length_mean = row["length_sum"] / n
        density_mean = row["density_sum"] / n
        result[str(cap_id)].update({
            "directed_fit_edges": row["n"],
            "log_length_mean": length_mean,
            "log_length_std": max(
                max(row["length_sum2"] / n - length_mean**2, 0.0) ** 0.5,
                1e-6,
            ),
            "log_density_contrast_mean": density_mean,
            "log_density_contrast_std": max(
                max(row["density_sum2"] / n - density_mean**2, 0.0) ** 0.5,
                1e-6,
            ),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, required=True)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--p5-root", type=Path, default=P5_ROOT)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--rotations", type=Path, default=ROTATIONS)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--points", type=Path, default=POINTS)
    parser.add_argument("--chunk", type=int, default=5_000_000)
    args = parser.parse_args()
    started = time.time()
    output = args.p8_root / "g_patch_features" / f"rotation_{args.rotation}"
    output.mkdir(parents=True, exist_ok=True)
    assignment = np.load(args.assignment, mmap_mode="r")
    rotations = json.loads(args.rotations.read_text())
    selection = json.loads(args.selection.read_text())
    train_folds, validation_fold, development_fold = fold_roles(rotations, args.rotation)
    auth = authoritative_mask(assignment)
    active_parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    active_fold = np.asarray(assignment["fold"], dtype=np.int8)
    train_rows = auth & np.isin(active_fold, train_folds)
    train_parent = active_parent[train_rows]
    points = np.load(args.points, mmap_mode="r")
    cap_all = np.asarray(points[:, 3], dtype=np.int8)
    redshift_all = np.load(args.p8_root / "parent_redshift.npy", mmap_mode="r")
    raw = np.load(args.p5_root / "node_features.npy", mmap_mode="r")

    medians = {}
    for cap_id in (0, 1):
        selected_parent = train_parent[cap_all[train_parent] == cap_id]
        medians[str(cap_id)] = {
            str(column): float(np.median(raw[selected_parent, column]))
            for column in SI_COLUMNS
        }
    train_si = transform_si(
        np.asarray(raw[train_parent], dtype=np.float64), cap_all[train_parent], medians
    )
    if np.any(train_si + 1e-6 <= 0):
        raise RuntimeError("Box-Cox input is not positive after registered epsilon")
    node_power = PowerTransformer(method="box-cox").fit(train_si + 1e-6)
    ntilde_train = curve_values(
        np.asarray(redshift_all[train_parent]), cap_all[train_parent], selection, args.rotation
    )
    log_ntilde = np.log(np.maximum(ntilde_train, 1e-12))
    ntilde_mean, ntilde_std = float(log_ntilde.mean()), float(log_ntilde.std())
    if not np.isfinite(ntilde_std) or ntilde_std <= 0:
        raise RuntimeError("invalid training-node ntilde scaler")

    transformed_path = output / "node_features_8d.npy"
    transformed = np.lib.format.open_memmap(
        transformed_path, mode="w+", dtype=np.float32, shape=(len(raw), 8)
    )
    for start in range(0, len(raw), args.chunk):
        stop = min(start + args.chunk, len(raw))
        cap = cap_all[start:stop]
        si = transform_si(np.asarray(raw[start:stop]), cap, medians)
        transformed[start:stop, :7] = node_power.transform(si + 1e-6).astype(np.float32)
        ntilde = curve_values(
            np.asarray(redshift_all[start:stop]), cap, selection, args.rotation
        )
        transformed[start:stop, 7] = (
            (np.log(np.maximum(ntilde, 1e-12)) - ntilde_mean) / ntilde_std
        ).astype(np.float32)
    transformed.flush()
    del transformed, train_si
    scaler_path = output / "node_power_transformer.pkl"
    with scaler_path.open("wb") as handle:
        pickle.dump(node_power, handle)

    parent_train = np.zeros(len(raw), dtype=bool)
    parent_train[train_parent] = True
    edge = edge_fit(
        pairs=np.load(args.p5_root / "union_pairs.npy", mmap_mode="r"),
        attrs=np.load(args.p5_root / "union_edge_features.npy", mmap_mode="r"),
        parent_train=parent_train,
        parent_cap=cap_all,
        output=output,
        chunk=args.chunk,
    )
    manifest = {
        "schema_version": 1,
        "stage": "P8 G-PATCH frozen training-only feature transform",
        "rotation": args.rotation,
        "train_folds": list(train_folds),
        "validation_fold": validation_fold,
        "development_test_fold": development_fold,
        "training_authoritative_nodes": int(len(train_parent)),
        "node": {
            "raw_columns": ["Degree", "Clustering", "Density", "Neigh Density",
                            "I_eig1", "I_eig2", "I_eig3"],
            "si_columns": list(SI_COLUMNS),
            "si_median_scope": "per cap, authoritative training-fold nodes only",
            "si_medians": medians,
            "boxcox_scope": "pooled authoritative training-fold nodes after cap SI",
            "boxcox_epsilon": 1e-6,
            "ntilde_scope": "rotation/cap frozen curve evaluated at nodes; zscore on training nodes",
            "ntilde_log_mean": ntilde_mean,
            "ntilde_log_std": ntilde_std,
            "transformed_path": str(transformed_path),
            "transformed_sha256": sha256(transformed_path),
            "power_transformer": str(scaler_path),
            "power_transformer_sha256": sha256(scaler_path),
        },
        "edge": edge,
        "inputs": {
            "assignment": str(args.assignment),
            "assignment_sha256": sha256(args.assignment),
            "p5_adapter": str(args.p5_root / "adapter_manifest.json"),
            "p5_adapter_sha256": sha256(args.p5_root / "adapter_manifest.json"),
            "selection": str(args.selection),
            "selection_sha256": sha256(args.selection),
        },
        "gates": {
            "training_only_node_fit": True,
            "training_internal_edge_fit": all(edge[str(cap)]["directed_fit_edges"] > 0 for cap in (0, 1)),
            "features_finite": bool(np.all(np.isfinite(np.load(transformed_path, mmap_mode="r")))),
            "eight_node_features": np.load(transformed_path, mmap_mode="r").shape[1] == 8,
        },
        "elapsed_seconds": time.time() - started,
    }
    manifest["pass"] = all(manifest["gates"].values())
    atomic_json(output / "feature_manifest.json", manifest)
    if not manifest["pass"]:
        raise RuntimeError(f"G-PATCH feature gates failed: {manifest['gates']}")
    (output / "G_PATCH_FEATURES_READY").write_text(
        f"rotation={args.rotation} training_nodes={len(train_parent)}\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
