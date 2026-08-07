#!/usr/bin/env python3
"""Fit global Bright+Faint graph-feature transforms using training folds only."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import time

import healpy as hp
import numpy as np
from sklearn.preprocessing import PowerTransformer

from workflows.abacus_tweb.p6_refit_fullcap_selection import (
    CAP_ID,
    build_cap_lookup,
    radius_to_redshift_grid,
)
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, fold_roles, sha256
from workflows.abacus_tweb.p8_prepare_graph_features import (
    SI_COLUMNS,
    edge_fit,
    transform_si,
)
from workflows.abacus_tweb.p8_refit_multitracer_selection import point_folds


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
P4 = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", default="bf_proxy_response_v1")
    parser.add_argument("--rotation", type=int, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--p4-root", type=Path, default=P4)
    parser.add_argument("--chunk", type=int, default=2_000_000)
    return parser.parse_args()


def curves_at_nodes(
    redshift: np.ndarray,
    cap: np.ndarray,
    tracer: np.ndarray,
    selection: dict,
    rotation: int,
) -> np.ndarray:
    result = np.empty(len(redshift), dtype=np.float64)
    bright_manifest = json.loads(
        Path(selection["tracers"]["BGS_BRIGHT"]["selection_manifest"]).read_text()
    )
    curves = {
        0: bright_manifest["rotations"][str(rotation)]["caps"],
        1: selection["tracers"]["BGS_FAINT"]["rotations"][str(rotation)]["caps"],
    }
    for tracer_id in (0, 1):
        for cap_id, cap_name in ((0, "SGC"), (1, "NGC")):
            selected = (tracer == tracer_id) & (cap == cap_id)
            curve = curves[tracer_id][cap_name]
            grid_z = np.asarray(curve["grid_z"], dtype=np.float64)
            ntilde = np.asarray(curve["ntilde"], dtype=np.float64)
            result[selected] = np.interp(
                np.clip(redshift[selected], grid_z[0], grid_z[-1]), grid_z, ntilde
            )
    return result


def node_angular_response(
    points: np.ndarray, tracer: np.ndarray, cap: np.ndarray, field_manifest: dict
) -> np.ndarray:
    response = np.ones(len(points), dtype=np.float32)
    faint = tracer == 1
    if not np.any(faint):
        return response
    xyz = np.asarray(points[faint, :3], dtype=np.float64)
    radius = np.linalg.norm(xyz, axis=1)
    nside = int(field_manifest["response"]["nside"])
    pixel = hp.vec2pix(
        nside, xyz[:, 0] / radius, xyz[:, 1] / radius, xyz[:, 2] / radius,
        nest=False,
    )
    faint_cap = cap[faint]
    values = np.empty(len(xyz), dtype=np.float32)
    for cap_id, cap_name in ((0, "SGC"), (1, "NGC")):
        path = (
            Path(field_manifest["components"][cap_name]["file"]).parent
            / f"{cap_name.lower()}_faint_response.npz"
        )
        table = np.load(path)
        selected = faint_cap == cap_id
        values[selected] = np.asarray(table["response"], dtype=np.float32)[pixel[selected]]
    response[faint] = values
    return response


def main() -> None:
    args = parse_args()
    started = time.time()
    product = args.product
    catalogue_manifest_path = args.root / "catalogues" / product / "manifest.json"
    field_manifest_path = args.root / "fields" / product / "manifest.json"
    selection_path = args.root / "selection" / product / "multitracer_selection_manifest.json"
    adapter_root = args.root / "graph" / product / "adapter"
    catalogue = json.loads(catalogue_manifest_path.read_text())
    fields = json.loads(field_manifest_path.read_text())
    selection = json.loads(selection_path.read_text())
    p4 = json.loads((args.p4_root / "spatial_manifest.json").read_text())
    rotations = json.loads((args.p4_root / "rotations.json").read_text())
    train_folds, validation_fold, development_fold = fold_roles(rotations, args.rotation)
    core_mpc = float(p4["unit_contract"]["core_mpc"])
    cores = np.load(args.p4_root / "cores.npz", mmap_mode="r")
    lookups = {
        cap_id: build_cap_lookup(cores, cap_id, core_mpc) for cap_id in (0, 1)
    }

    output = args.root / "models/g_patch_features" / product / f"rotation_{args.rotation}"
    output.mkdir(parents=True, exist_ok=True)
    points = np.load(catalogue["points"], mmap_mode="r")
    index = np.load(catalogue["index"])
    tracer = np.asarray(index["tracer_type"], dtype=np.uint8)
    context = np.asarray(index["context"], dtype=bool)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    raw = np.load(adapter_root / "node_features.npy", mmap_mode="r")
    if len(raw) != len(points) or len(raw) != len(tracer):
        raise RuntimeError("multitracer graph/node/index row mismatch")

    fold = np.full(len(points), -1, dtype=np.int8)
    for cap_id in (0, 1):
        selected = cap == cap_id
        fold[selected] = point_folds(
            np.asarray(points[selected, :3]),
            base_mpc=lookups[cap_id]["base_mpc"],
            core_mpc=core_mpc,
            fold_lookup=lookups[cap_id]["lookup"],
        )
    training = context & np.isin(fold, train_folds)
    train_id = np.flatnonzero(training)
    if len(train_id) == 0 or not set(np.unique(tracer[train_id])) == {0, 1}:
        raise RuntimeError("training-fold graph context does not contain both tracers")

    medians = {}
    for cap_id in (0, 1):
        selected = train_id[cap[train_id] == cap_id]
        medians[str(cap_id)] = {
            str(column): float(np.median(raw[selected, column])) for column in SI_COLUMNS
        }
    train_si = transform_si(
        np.asarray(raw[train_id], dtype=np.float64), cap[train_id], medians
    )
    if np.any(train_si + 1.0e-6 <= 0):
        raise RuntimeError("multitracer Box-Cox input is not positive")
    power = PowerTransformer(method="box-cox").fit(train_si + 1.0e-6)
    radius_grid, redshift_grid = radius_to_redshift_grid(0.10, 0.60)
    radius = np.linalg.norm(np.asarray(points[:, :3]), axis=1)
    redshift = np.interp(radius, radius_grid, redshift_grid)
    ntilde_train = curves_at_nodes(
        redshift[train_id], cap[train_id], tracer[train_id], selection, args.rotation
    )
    log_ntilde = np.log(np.maximum(ntilde_train, 1.0e-12))
    ntilde_mean = float(log_ntilde.mean())
    ntilde_std = float(log_ntilde.std())
    if not np.isfinite(ntilde_std) or ntilde_std <= 0:
        raise RuntimeError("invalid multitracer ntilde scaler")
    response = node_angular_response(points, tracer, cap, fields)

    transformed_path = output / "node_features_10d.npy"
    transformed = np.lib.format.open_memmap(
        transformed_path, mode="w+", dtype=np.float32, shape=(len(raw), 10)
    )
    for start in range(0, len(raw), args.chunk):
        stop = min(start + args.chunk, len(raw))
        transformed[start:stop, :7] = power.transform(
            transform_si(np.asarray(raw[start:stop]), cap[start:stop], medians) + 1.0e-6
        ).astype(np.float32)
        ntilde = curves_at_nodes(
            redshift[start:stop], cap[start:stop], tracer[start:stop],
            selection, args.rotation,
        )
        transformed[start:stop, 7] = (
            (np.log(np.maximum(ntilde, 1.0e-12)) - ntilde_mean) / ntilde_std
        ).astype(np.float32)
        transformed[start:stop, 8] = tracer[start:stop].astype(np.float32)
        transformed[start:stop, 9] = response[start:stop]
    transformed.flush()
    del transformed, train_si
    scaler_path = output / "node_power_transformer.pkl"
    with scaler_path.open("wb") as handle:
        pickle.dump(power, handle)

    edge = edge_fit(
        pairs=np.load(adapter_root / "union_pairs.npy", mmap_mode="r"),
        attrs=np.load(adapter_root / "union_edge_features.npy", mmap_mode="r"),
        parent_train=training,
        parent_cap=cap,
        output=output,
        chunk=args.chunk,
    )
    manifest = {
        "schema_version": "p8-multitracer-graph-features-v1",
        "product": product,
        "rotation": args.rotation,
        "train_folds": list(train_folds),
        "validation_fold": validation_fold,
        "development_test_fold": development_fold,
        "training_context_nodes": int(len(train_id)),
        "training_context_by_tracer": {
            "BGS_BRIGHT": int(np.count_nonzero(tracer[train_id] == 0)),
            "BGS_FAINT": int(np.count_nonzero(tracer[train_id] == 1)),
        },
        "node": {
            "raw_metric_columns": [
                "Degree", "Clustering", "Density", "Neigh Density",
                "I_eig1", "I_eig2", "I_eig3",
            ],
            "output_columns": [
                "metric_0_boxcox", "metric_1_boxcox", "metric_2_boxcox",
                "metric_3_boxcox", "metric_4_boxcox", "metric_5_boxcox",
                "metric_6_boxcox", "log_ntilde_tracer_zscore", "is_faint",
                "angular_response",
            ],
            "fit_scope": "all label-free context nodes in P4 training folds only",
            "si_medians": medians,
            "ntilde_log_mean": ntilde_mean,
            "ntilde_log_std": ntilde_std,
            "transformed_path": str(transformed_path),
            "transformed_sha256": sha256(transformed_path),
            "power_transformer": str(scaler_path),
            "power_transformer_sha256": sha256(scaler_path),
        },
        "edge": edge,
        "supervision_contract": "Faint is context only; feature fitting uses no labels",
        "inputs": {
            "adapter_manifest": str(adapter_root / "adapter_manifest.json"),
            "adapter_manifest_sha256": sha256(adapter_root / "adapter_manifest.json"),
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": sha256(selection_path),
            "catalogue_manifest": str(catalogue_manifest_path),
            "catalogue_manifest_sha256": sha256(catalogue_manifest_path),
        },
        "gates": {
            "training_only_fit": True,
            "both_tracers_in_training_context": set(np.unique(tracer[train_id])) == {0, 1},
            "ten_features": np.load(transformed_path, mmap_mode="r").shape[1] == 10,
            "features_finite": bool(
                np.all(np.isfinite(np.load(transformed_path, mmap_mode="r")))
            ),
            "response_bounded": bool(np.all((response >= 0) & (response <= 1))),
            "training_internal_edges": all(
                edge[str(cap_id)]["directed_fit_edges"] > 0 for cap_id in (0, 1)
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    manifest["pass"] = bool(all(manifest["gates"].values()))
    atomic_json(output / "feature_manifest.json", manifest)
    if not manifest["pass"]:
        raise RuntimeError(f"multitracer graph feature gates failed: {manifest['gates']}")
    (output / "MULTITRACER_G_PATCH_FEATURES_READY").write_text(
        f"rotation={args.rotation} training_context_nodes={len(train_id)}\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
