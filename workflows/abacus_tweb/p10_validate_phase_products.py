#!/usr/bin/env python3
"""Validate phase P2 products or the complete P1--P4 phase contract.

P2 validation streams Parquet metrics so finite values, row identity, endpoint
bounds, and disconnected caps are checked without loading the whole graph twice.
The phase validator separates exact physical invariants from statistical
cosmic-variance diagnostics and writes an atomic readiness marker only after all
P1--P4 completion gates pass.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, WORKFLOW_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from p10_phase_assets import DEFAULT_REGISTRY, load_registry, sha256_file  # noqa: E402


class ProductValidationError(RuntimeError):
    """A P2 or P1--P4 product failed its frozen contract."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        payload, indent=2, sort_keys=True,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ) + "\n")
    os.replace(temporary, path)


def phase_paths(registry: dict[str, Any], phase: str) -> dict[str, Path]:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    prefix = f"{phase}_bgs_bright_full_delaunay"
    density_prefix = f"AbacusSummit_base_c000_{phase}_z0.200_ngrid2048_ab10_tsc_counts"
    return {
        "root": root, "p1": root / "p1_canonical", "p2": root / "p2_graph",
        "p2_union": root / "p2_union", "p3": root / "p3_fields", "p4": root / "p4_patches",
        "graph": root / "p2_graph/GRAPH_COMPLETE.json",
        "graph_meta": root / f"p2_graph/{prefix}_metadata.json",
        "gnn_meta": root / f"p2_graph/{prefix}_cugraph_gnn_metadata.json",
        "union": root / "p2_union/p2b_union_manifest.json",
        "p2_complete": root / "p2_graph/P2_COMPLETE.json",
        "density": root / f"targets/density/{density_prefix}.manifest.json",
        "tweb": root / "targets/tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json",
    }


def validate_p2(registry: dict[str, Any], phase: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    paths = phase_paths(registry, phase)
    p1 = json.loads((paths["p1"] / "manifest.json").read_text())
    index = np.load(paths["p1"] / "canonical_index.npz")
    cap = np.asarray(index["cap"], dtype=np.uint8)
    context = np.asarray(index["context"], dtype=bool)
    graph = json.loads(paths["graph"].read_text())
    graph_meta = json.loads(paths["graph_meta"].read_text())
    gnn = json.loads(paths["gnn_meta"].read_text())
    union = json.loads(paths["union"].read_text())
    if graph["phase"] != phase or p1["phase"] != phase:
        raise ProductValidationError("P1/P2 phase mismatch")
    n = len(cap)
    if graph_meta["n_points"] != n or gnn["n_points"] != n:
        raise ProductValidationError("P2 node count differs from P1")
    node_path = Path(gnn["outputs"]["node_features"])
    edge_path = Path(gnn["outputs"]["edge_features"])
    arrays_path = Path(gnn["outputs"]["gnn_arrays_npz"])
    node_pf, edge_pf = pq.ParquetFile(node_path), pq.ParquetFile(edge_path)
    if node_pf.metadata.num_rows != n or edge_pf.metadata.num_rows != graph_meta["n_edges"]:
        raise ProductValidationError("metric Parquet row counts differ from graph")
    node_columns = ["Node ID", *gnn["node_feature_columns"]]
    expected_nodes = 0
    finite_nodes = True
    for batch in node_pf.iter_batches(batch_size=500_000, columns=node_columns):
        values = batch.to_pandas()
        ids = values["Node ID"].to_numpy(dtype=np.int64)
        if not np.array_equal(ids, np.arange(expected_nodes, expected_nodes + len(ids))):
            raise ProductValidationError("node metric row identity is not canonical P1 order")
        finite_nodes &= bool(np.isfinite(values[gnn["node_feature_columns"]].to_numpy()).all())
        expected_nodes += len(ids)
    finite_edges, cross_cap, endpoint_max, edge_rows = True, 0, -1, 0
    edge_columns = ["src", "dst", *gnn["edge_feature_columns"]]
    for batch in edge_pf.iter_batches(batch_size=1_000_000, columns=edge_columns):
        values = batch.to_pandas()
        src = values["src"].to_numpy(dtype=np.int64)
        dst = values["dst"].to_numpy(dtype=np.int64)
        if len(src):
            endpoint_max = max(endpoint_max, int(max(src.max(), dst.max())))
            cross_cap += int(np.count_nonzero(cap[src] != cap[dst]))
        finite_edges &= bool(np.isfinite(values[gnn["edge_feature_columns"]].to_numpy()).all())
        edge_rows += len(src)
    gates = {
        "p1_row_identity": expected_nodes == n,
        "node_metrics_finite": finite_nodes,
        "edge_metrics_finite": finite_edges,
        "edge_rows_match": edge_rows == graph_meta["n_edges"],
        "edge_endpoints_in_bounds": endpoint_max < n,
        "disconnected_caps": cross_cap == 0,
        "gnn_bundle_present": arrays_path.is_file() and arrays_path.stat().st_size > 0,
        "metric_columns_frozen": (
            gnn["node_feature_columns"] == ["Degree", "Clustering", "Density", "Neigh Density",
                                               "I_eig1", "I_eig2", "I_eig3"]
            and gnn["edge_feature_columns"] == ["edge_length", "x_dir", "y_dir", "z_dir",
                                                "density_contrast"]
        ),
        "union_radius_frozen": np.isclose(float(union["radius_mpc"]), 14.78),
        "union_parent_rows_match": union["counts"]["parent_nodes"] == n,
        "union_context_rows_match": union["counts"]["context_nodes"] == int(context.sum()),
        "union_cross_cap_zero": union["assembly_contract"]["cross_cap_pairs"] == 0,
    }
    report = {
        "schema_version": "p10-p2-complete-v1", "created_utc": utc_now(),
        "phase": phase, "catalogue_id": p1["catalogue_id"],
        "inputs": {"p1": str((paths["p1"] / "manifest.json").resolve()),
                   "graph": str(paths["graph"].resolve()),
                   "graph_metadata": str(paths["graph_meta"].resolve()),
                   "gnn_metadata": str(paths["gnn_meta"].resolve()),
                   "union": str(paths["union"].resolve())},
        "hashes": {"p1": sha256_file(paths["p1"] / "manifest.json"),
                   "graph": sha256_file(paths["graph"]),
                   "gnn_metadata": sha256_file(paths["gnn_meta"]),
                   "union": sha256_file(paths["union"]),
                   "gnn_arrays": sha256_file(arrays_path)},
        "counts": {"nodes": n, "delaunay_edges": edge_rows,
                   "union_context_pairs": union["counts"]["union_pairs_context"],
                   "cross_cap_edges": cross_cap},
        "gates": gates, "pass": all(gates.values()),
    }
    if not report["pass"]:
        raise ProductValidationError(f"P2 gates failed: {gates}")
    atomic_json(paths["p2_complete"], report)
    return report


def invariant_without_catalogue_id(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    payload.pop("catalogue_id", None)
    return payload


def validate_phase(registry: dict[str, Any], phase: str) -> dict[str, Any]:
    paths = phase_paths(registry, phase)
    cfg = registry["phases"][phase]
    required = {
        "p1_marker": paths["p1"] / "CATALOGUE_COMPLETE.json",
        "p1_manifest": paths["p1"] / "manifest.json",
        "p2": paths["p2_complete"],
        "p3_marker": paths["p3"] / "FIELD_COMPLETE",
        "p3_manifest": paths["p3"] / "field_manifest.json",
        "p4_marker": paths["p4"] / "PATCH_MANIFEST_COMPLETE",
        "p4_validation": paths["p4"] / "p4_validation.json",
        "schemas": paths["root"] / "contracts/SCHEMAS_COMPLETE.json",
    }
    blind = cfg["role"] == "sealed_blind"
    if not blind:
        required.update({"density": paths["density"], "tweb": paths["tweb"]})
    missing = {name: str(path) for name, path in required.items() if not path.is_file()}
    if missing:
        raise ProductValidationError(f"phase is incomplete: {missing}")
    p1 = json.loads(required["p1_marker"].read_text())
    p1_manifest = json.loads(required["p1_manifest"].read_text())
    p2 = json.loads(required["p2"].read_text())
    p3 = json.loads(required["p3_manifest"].read_text())
    p4 = json.loads(required["p4_validation"].read_text())
    schema_marker = json.loads(required["schemas"].read_text())
    phase_p3_schema = Path(schema_marker["records"]["p3"]["output"])
    phase_p4_schema = Path(schema_marker["records"]["p4"]["output"])
    base_p3 = Path(schema_marker["records"]["p3"]["base"])
    base_p4 = Path(schema_marker["records"]["p4"]["base"])
    exact_physics = {
        "target_contract_matches_registry": p1["target_contract"] == registry["target_contract"],
        "coordinate_units_mpc": p1["geometry"]["units"] == "Mpc",
        "coordinate_cosmology_planck18": p1["geometry"]["cosmology"] == "Astropy Planck18",
        "p3_schema_only_catalogue_id_differs": (
            invariant_without_catalogue_id(phase_p3_schema) == invariant_without_catalogue_id(base_p3)),
        "p4_schema_only_catalogue_id_differs": (
            invariant_without_catalogue_id(phase_p4_schema) == invariant_without_catalogue_id(base_p4)),
        "p2_pass": bool(p2["pass"]), "p3_pass": all(p3["gates"].values()),
        "p4_pass": bool(p4["pass"]),
        "catalogue_identity_consistent": len({p1_manifest["catalogue_id"], p2["catalogue_id"],
                                                p3["catalogue_id"]}) == 1,
    }
    blind_gates = {
        "truth_not_embedded_in_p1": not bool(p1_manifest["target_truth_present"]),
        "density_product_absent": not paths["density"].exists(),
        "tweb_product_absent": not paths["tweb"].exists(),
    }
    truth_gates: dict[str, bool] | None = None
    truth_diagnostics: dict[str, Any] | None = None
    if not blind:
        density = json.loads(paths["density"].read_text())
        tweb = json.loads(paths["tweb"].read_text())
        build = density["build"]
        truth_gates = {
            "truth_embedded_in_p1": bool(p1_manifest["target_truth_present"]),
            "density_phase_matches": density["phase"] == phase,
            "density_contract_matches_registry": density["target_contract"] == registry["target_contract"],
            "density_grid_frozen": (
                int(build["ngrid"]) == int(registry["target_contract"]["grid_size"])
                and np.isclose(float(build["boxsize_mpc_h"]),
                               float(registry["target_contract"]["box_size_mpc_h"]))
                and build["dtype"] == "float32"
            ),
            "density_uses_all_136_a_plus_b_slabs": int(build["processed_file_count"]) == 136,
            "density_count_conserved": float(build["relative_count_error"]) <= 2.0e-6,
            "tweb_phase_matches": tweb["phase"] == phase,
            "tweb_contract_matches_registry": tweb["target_contract"] == registry["target_contract"],
            "tweb_density_verified": bool(tweb["density"]["verified"]),
            "tweb_rank_layout_complete": (
                int(tweb["mpi_size"]) == 16
                and int(tweb["outputs"]["rank_count"]) == 16
                and tweb["outputs"]["x_coverage"] == [0, 2048]
                and bool(tweb["outputs"]["verified"])
            ),
        }
        truth_diagnostics = {
            "particle_count": int(build["particle_count"]),
            "deposited_count": float(build["deposited_count"]),
            "relative_count_error": float(build["relative_count_error"]),
            "density_wall_seconds": float(build["wall_seconds"]),
            "tweb_total_bytes": int(tweb["outputs"]["total_bytes"]),
        }
    ph000_p1 = json.loads(Path("/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/manifest.json").read_text())
    count_ratio = p1["counts"]["context"] / ph000_p1["counts"]["context"]
    graph_mean_degree = 2.0 * p2["counts"]["union_context_pairs"] / p1["counts"]["context"]
    diagnostics = {
        "reference_phase": "ph000", "context_count_ratio_to_ph000": count_ratio,
        "union_mean_degree": graph_mean_degree,
        "shell_active_counts": p1["counts"]["by_shell"],
        "interpretation": ("count/degree differences are cosmic-realization and observation-view "
                           "diagnostics, not expected to equal ph000 exactly"),
    }
    gates = {**exact_physics, "context_count_plausible_vs_ph000": 0.5 < count_ratio < 1.5,
             "union_mean_degree_positive": np.isfinite(graph_mean_degree) and graph_mean_degree > 0}
    if blind:
        gates.update({f"sealed_{key}": value for key, value in blind_gates.items()})
    else:
        gates.update(truth_gates or {})
    payload = {
        "schema_version": "p10-phase-input-complete-v1" if blind else "p10-phase-complete-v1",
        "created_utc": utc_now(), "phase": phase, "role": cfg["role"],
        "completion_scope": "blind P1-P4 inference inputs; no scored truth" if blind else "truth-bearing P1-P4",
        "artifacts": {name: str(path.resolve()) for name, path in required.items()},
        "artifact_sha256": {name: sha256_file(path) for name, path in required.items()},
        "exact_physics_gates": exact_physics, "blind_gates": blind_gates if blind else None,
        "truth_gates": truth_gates, "truth_diagnostics": truth_diagnostics,
        "statistical_diagnostics": diagnostics, "gates": gates, "pass": all(gates.values()),
    }
    if not payload["pass"]:
        raise ProductValidationError(f"phase gates failed: {gates}")
    marker = paths["root"] / ("BLIND_INPUT_COMPLETE.json" if blind else "PHASE_COMPLETE.json")
    atomic_json(marker, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--stage", choices=("p2", "phase"), required=True)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    if args.phase not in registry["phases"]:
        raise ProductValidationError(f"unregistered phase: {args.phase}")
    payload = validate_p2(registry, args.phase) if args.stage == "p2" else validate_phase(registry, args.phase)
    print(json.dumps(
        payload, indent=2, sort_keys=True,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ))


if __name__ == "__main__":
    try:
        main()
    except (ProductValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
