#!/usr/bin/env python3
"""Independent P4 readback, determinism, balance, and completion validator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest")
    ap.add_argument("--geometry", type=Path, default=root / "spatial_manifest.json")
    ap.add_argument("--graph-support", type=Path, default=root / "graph_support_manifest.json")
    ap.add_argument("--field-support", type=Path, default=root / "field_support_manifest.json")
    ap.add_argument("--rebuild", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest_determinism/"
        "spatial_manifest.json"))
    ap.add_argument("--out", type=Path, default=root / "p4_validation.json")
    ap.add_argument("--marker", type=Path, default=root / "PATCH_MANIFEST_COMPLETE")
    args = ap.parse_args()
    geometry = json.loads(args.geometry.read_text())
    graph = json.loads(args.graph_support.read_text())
    field = json.loads(args.field_support.read_text())
    rebuild = json.loads(args.rebuild.read_text())

    cores = np.load(geometry["artifacts"]["cores"])
    supers = np.load(geometry["artifacts"]["super_blocks"])
    active = np.load(geometry["artifacts"]["active_assignment"])
    context = np.load(geometry["artifacts"]["context_assignment"])
    eligible = np.asarray(active["supervised_eligible"], dtype=bool)
    folds = np.asarray(active["fold"], dtype=np.uint8)
    distance = np.asarray(active["distance_to_conservative_fold_boundary_mpc"],
                          dtype=np.float64)
    fold_distance = {
        str(fold_id): {
            "eligible_rows": int(np.sum(eligible & (folds == fold_id))),
            "distance_mpc_quantiles": [float(v) for v in np.quantile(
                distance[eligible & (folds == fold_id)], [0.0, 0.1, 0.5, 0.9, 1.0])],
        } for fold_id in range(5)
    }
    medians = np.asarray([fold_distance[str(i)]["distance_mpc_quantiles"][2]
                          for i in range(5)])
    super_counts = np.asarray([geometry["folds"][str(i)]["super_blocks"] for i in range(5)])

    semantic_determinism = {
        "artifact_sha256": geometry["artifact_sha256"] == rebuild["artifact_sha256"],
        "unit_contract": geometry["unit_contract"] == rebuild["unit_contract"],
        "counts": geometry["counts"] == rebuild["counts"],
        "folds": geometry["folds"] == rebuild["folds"],
        "fold_assignment": geometry.get("fold_assignment") == rebuild.get("fold_assignment"),
        "fold_balance": geometry["fold_balance"] == rebuild["fold_balance"],
        "periodic_image_audit": geometry["periodic_image_audit"] == rebuild["periodic_image_audit"],
        "rotations_sha256": geometry["rotations_sha256"] == rebuild["rotations_sha256"],
    }
    geometry_hashes = {
        key: sha256(Path(path)) for key, path in geometry["artifacts"].items()
    }
    graph_hashes = {key: sha256(Path(path)) for key, path in graph["artifacts"].items()}
    field_hashes = {key: sha256(Path(path)) for key, path in field["artifacts"].items()}

    gates = {
        "geometry_pass": bool(geometry["pass"]),
        "graph_support_pass": bool(graph["pass"]),
        "field_support_pass": bool(field["pass"]),
        "geometry_artifact_hashes_match": geometry_hashes == geometry["artifact_sha256"],
        "graph_artifact_hashes_match": graph_hashes == graph["artifact_sha256"],
        "field_artifact_hashes_match": field_hashes == field["artifact_sha256"],
        "deterministic_semantic_rebuild": all(semantic_determinism.values()),
        "core_ids_dense_unique": np.array_equal(
            np.asarray(cores["core_id"]), np.arange(len(cores["core_id"]))),
        "superblock_ids_dense_unique": np.array_equal(
            np.asarray(supers["superblock_id"]), np.arange(len(supers["superblock_id"]))),
        "active_parent_ids_unique": len(np.unique(active["parent_node_id"])) == len(active["parent_node_id"]),
        "context_parent_ids_unique": len(np.unique(context["parent_node_id"])) == len(context["parent_node_id"]),
        "eligible_count_matches_geometry": int(eligible.sum()) == geometry["counts"]["active_rows"],
        "fold_distance_medians_matched_below_25pct": float(medians.max() / medians.min()) < 1.25,
        "fold_occupied_superblock_ratio_below_1p25": float(super_counts.max() / super_counts.min()) < 1.25,
        "all_fold_cap_shell_cells_nonempty": all(
            count > 0 for fold in geometry["folds"].values()
            for values in fold["by_cap_shell"].values() for count in values),
        "fft_status_explicitly_reserved": geometry["support_status"]["fft"].startswith("reserved=0"),
    }
    payload = {
        "schema_version": 1, "stage": "P4 independent final validation",
        "geometry": str(args.geometry), "geometry_sha256": sha256(args.geometry),
        "graph_support": str(args.graph_support), "graph_support_sha256": sha256(args.graph_support),
        "field_support": str(args.field_support), "field_support_sha256": sha256(args.field_support),
        "determinism_rebuild": str(args.rebuild), "determinism_rebuild_sha256": sha256(args.rebuild),
        "unit_contract": geometry["unit_contract"],
        "counts": geometry["counts"], "folds": geometry["folds"],
        "fold_distance": fold_distance,
        "graph_support_summary": graph["support"],
        "field_support_summary": field["global"],
        "periodic_image_audit": geometry["periodic_image_audit"],
        "semantic_determinism": semantic_determinism,
        "gates": gates, "pass": all(gates.values()),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=bool) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, default=bool))
    if not payload["pass"]:
        raise RuntimeError(f"P4 final validation failed: {gates}")
    args.marker.write_text(
        f"stage=P4_COMPLETE\nvalidation_sha256={sha256(args.out)}\n"
        f"geometry_sha256={sha256(args.geometry)}\n"
        f"graph_support_sha256={sha256(args.graph_support)}\n"
        f"field_support_sha256={sha256(args.field_support)}\n")


if __name__ == "__main__":
    main()
