#!/usr/bin/env python3
"""Project full-graph cuGraph GNN arrays onto a wedge subgraph (CPU, no GPU).

Wedge graph topology (points, edges, tetrahedra, ``global_node_ids``) comes from
``subset_abacus_graph_wedge_for_sbi.py``. This script **does not** re-run cuGraph on
the induced wedge; it copies node metrics from the full-volume NPZ and edge metrics
from the full graph wherever a wedge edge matches the parent edge list orientation.

Outputs (same schema as ``abacus_graph_features_cugraph.py``):
  <wedge-prefix>_cugraph_gnn_arrays.npz   (x, edge_index, edge_attr)
  <wedge-prefix>_cugraph_gnn_metadata.json

Edge alignment
----------------
Parent wedge edges are an **induced subgraph** of the full graph: each wedge row
``(u_local, v_local)`` is the parent edge ``(g_u, g_v)`` reindexed with
``global_node_ids``. We lookup full ``edge_attr`` with key
``g_u * n_full + g_v`` (same directed pair as stored in full ``edge_index``).

If any wedge edge is missing from the full lookup (should not happen for a proper
B1 subset), the script fails by default. Optional ``--recompute-missing-edge-lengths``
fills only ``edge_length`` from wedge ``points_xyz`` and leaves other edge columns
as NaN for those rows (not recommended for production).

Node features are always ``x_full[global_node_ids]`` (global graph indices).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NODE_COLS = [
    "Degree",
    "Clustering",
    "Density",
    "Neigh Density",
    "I_eig1",
    "I_eig2",
    "I_eig3",
]
EDGE_COLS = ["edge_length", "x_dir", "y_dir", "z_dir", "density_contrast"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/graph_constructions"),
        help="Directory with graph + cuGraph artifacts.",
    )
    p.add_argument(
        "--full-prefix",
        type=str,
        required=True,
        help="Prefix of full graph (e.g. staged_mock_stage3_postcollision_full_rs7).",
    )
    p.add_argument(
        "--wedge-prefix",
        type=str,
        required=True,
        help="Prefix of wedge subgraph from subset_abacus_graph_wedge_for_sbi.py.",
    )
    p.add_argument(
        "--full-gnn-npz",
        type=Path,
        default=None,
        help="Override path to full *_cugraph_gnn_arrays.npz.",
    )
    p.add_argument(
        "--full-gnn-metadata",
        type=Path,
        default=None,
        help="Override path to full *_cugraph_gnn_metadata.json (for provenance only).",
    )
    p.add_argument(
        "--wedge-metadata",
        type=Path,
        default=None,
        help="Override path to wedge *_metadata.json (default: artifacts-dir/wedge-prefix).",
    )
    p.add_argument(
        "--edge-chunk",
        type=int,
        default=5_000_000,
        help="Chunk size when building full-graph edge keys (limits peak RAM).",
    )
    p.add_argument(
        "--recompute-missing-edge-lengths",
        action="store_true",
        help="If a wedge edge is absent from full edge_index, set edge_length from wedge xyz.",
    )
    p.add_argument(
        "--compare-existing-npz",
        type=Path,
        default=None,
        help="Optional existing wedge NPZ to print max|diff| vs new x/edge_attr (sanity).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and edge lookup coverage without writing outputs.",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _edge_keys(src: np.ndarray, dst: np.ndarray, n_nodes: int) -> np.ndarray:
    return src.astype(np.int64) * int(n_nodes) + dst.astype(np.int64)


def _build_sorted_edge_index(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    n_nodes: int,
    *,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_keys, order) where order permutes edge_attr rows."""
    n_edges = int(edge_index.shape[1])
    keys = np.empty(n_edges, dtype=np.int64)
    for start in range(0, n_edges, chunk):
        stop = min(start + chunk, n_edges)
        keys[start:stop] = _edge_keys(edge_index[0, start:stop], edge_index[1, start:stop], n_nodes)
    order = np.argsort(keys, kind="mergesort")
    return keys[order], order


def _lookup_edge_attr(
    keys_sorted: np.ndarray,
    order: np.ndarray,
    edge_attr_full: np.ndarray,
    wedge_edges: np.ndarray,
    global_ids: np.ndarray,
    n_full: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map wedge (E,2) local edges to full edge_attr rows; return (edge_attr, missing_mask)."""
    g_src = global_ids[wedge_edges[:, 0].astype(np.int64)]
    g_dst = global_ids[wedge_edges[:, 1].astype(np.int64)]
    qkeys = _edge_keys(g_src, g_dst, n_full)

    pos = np.searchsorted(keys_sorted, qkeys)
    found = np.zeros(qkeys.shape[0], dtype=bool)
    valid = pos < keys_sorted.size
    found[valid] = keys_sorted[pos[valid]] == qkeys[valid]

    out = np.empty((wedge_edges.shape[0], edge_attr_full.shape[1]), dtype=np.float32)
    missing = ~found
    if np.any(found):
        out[found] = edge_attr_full[order[pos[found]]]
    return out, missing


def _recompute_edge_lengths(points_xyz: np.ndarray, wedge_edges: np.ndarray) -> np.ndarray:
    diffs = points_xyz[wedge_edges[:, 1], :3] - points_xyz[wedge_edges[:, 0], :3]
    return np.linalg.norm(diffs, axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    art = args.artifacts_dir.expanduser().resolve()

    full_npz = (
        args.full_gnn_npz.expanduser().resolve()
        if args.full_gnn_npz
        else art / f"{args.full_prefix}_cugraph_gnn_arrays.npz"
    )
    full_gnn_meta = (
        args.full_gnn_metadata.expanduser().resolve()
        if args.full_gnn_metadata
        else art / f"{args.full_prefix}_cugraph_gnn_metadata.json"
    )
    wedge_meta_path = (
        args.wedge_metadata.expanduser().resolve()
        if args.wedge_metadata
        else art / f"{args.wedge_prefix}_metadata.json"
    )

    wedge_edges_path = art / f"{args.wedge_prefix}_edges_combined_idx.npy"
    global_ids_path = art / f"{args.wedge_prefix}_global_node_ids.npy"
    wedge_xyz_path = art / f"{args.wedge_prefix}_points_xyz.npy"

    for path in (full_npz, full_gnn_meta, wedge_meta_path, wedge_edges_path, global_ids_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    wedge_meta = _load_json(wedge_meta_path)
    n_wedge = int(wedge_meta["n_points"])
    n_wedge_edges = int(wedge_meta["n_edges"])

    global_ids = np.load(global_ids_path).astype(np.int64, copy=False)
    if global_ids.shape[0] != n_wedge:
        raise ValueError(
            f"global_node_ids length {global_ids.shape[0]:,} != wedge n_points {n_wedge:,}"
        )

    wedge_edges = np.load(wedge_edges_path).astype(np.int64, copy=False)
    if wedge_edges.shape != (n_wedge_edges, 2):
        raise ValueError(f"Expected wedge edges ({n_wedge_edges}, 2); got {wedge_edges.shape}")

    print(f"Loading full cuGraph NPZ: {full_npz}")
    with np.load(full_npz) as full_data:
        x_full = full_data["x"]
        edge_index_full = full_data["edge_index"]
        edge_attr_full = full_data["edge_attr"]

    n_full = int(x_full.shape[0])
    if global_ids.max(initial=0) >= n_full or global_ids.min(initial=0) < 0:
        raise ValueError(
            f"global_node_ids out of range [0, {n_full}): "
            f"min={int(global_ids.min())} max={int(global_ids.max())}"
        )

    print(f"Full graph: n_nodes={n_full:,}, n_edges={edge_index_full.shape[1]:,}")
    print(f"Wedge:      n_nodes={n_wedge:,}, n_edges={n_wedge_edges:,}")

    x_wedge = np.asarray(x_full[global_ids], dtype=np.float32)
    if x_wedge.shape != (n_wedge, x_full.shape[1]):
        raise RuntimeError(f"Unexpected wedge x shape: {x_wedge.shape}")

    print("Building sorted full-graph edge key index...")
    keys_sorted, order = _build_sorted_edge_index(
        edge_index_full, edge_attr_full, n_full, chunk=int(args.edge_chunk)
    )
    del edge_index_full  # free before wedge lookup

    print("Looking up wedge edge attributes from full graph...")
    edge_attr_wedge, missing = _lookup_edge_attr(
        keys_sorted,
        order,
        edge_attr_full,
        wedge_edges,
        global_ids,
        n_full,
    )
    n_missing = int(np.count_nonzero(missing))
    if n_missing:
        msg = f"{n_missing:,} / {n_wedge_edges:,} wedge edges not found in full edge_index"
        if args.recompute_missing_edge_lengths:
            if not wedge_xyz_path.exists():
                raise FileNotFoundError(
                    f"--recompute-missing-edge-lengths requires {wedge_xyz_path}"
                )
            xyz = np.load(wedge_xyz_path).astype(np.float64, copy=False)
            edge_attr_wedge[missing, 0] = _recompute_edge_lengths(xyz, wedge_edges[missing])
            print(f"WARN: {msg}; filled edge_length only for missing rows.")
        else:
            raise RuntimeError(
                f"{msg}. Wedge topology may not be an induced subgraph of the full graph, "
                "or edge orientation differs. Pass --recompute-missing-edge-lengths only "
                "for debugging."
            )
    else:
        print("All wedge edges matched full-graph edge_index (directed keys).")

    edge_index_wedge = wedge_edges.T.astype(np.int64, copy=False)

    if args.compare_existing_npz is not None:
        cmp_path = args.compare_existing_npz.expanduser().resolve()
        with np.load(cmp_path) as old:
            dx = np.max(np.abs(old["x"].astype(np.float64) - x_wedge.astype(np.float64)))
            de = np.max(np.abs(old["edge_attr"].astype(np.float64) - edge_attr_wedge.astype(np.float64)))
            print(f"Compare {cmp_path.name}: max|Δx|={dx:.6g}, max|Δedge_attr|={de:.6g}")

    out_npz = art / f"{args.wedge_prefix}_cugraph_gnn_arrays.npz"
    out_meta = art / f"{args.wedge_prefix}_cugraph_gnn_metadata.json"

    if args.dry_run:
        print("Dry run OK — no files written.")
        return

    art.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        x=x_wedge,
        edge_index=edge_index_wedge,
        edge_attr=edge_attr_wedge,
    )

    meta_payload = {
        "input_metadata_path": str(wedge_meta_path),
        "input_prefix": args.wedge_prefix,
        "input_mode": wedge_meta.get("mode"),
        "input_alpha_sq": wedge_meta.get("alpha_sq"),
        "n_points": n_wedge,
        "n_edges": n_wedge_edges,
        "n_tetrahedra": int(wedge_meta.get("n_tetrahedra", 0)),
        "node_feature_columns": NODE_COLS,
        "edge_feature_columns": EDGE_COLS,
        "metrics_source": "subset_from_full_cugraph",
        "full_cugraph_npz": str(full_npz),
        "full_cugraph_metadata_path": str(full_gnn_meta),
        "full_graph_prefix": args.full_prefix,
        "wedge_global_node_ids": str(global_ids_path),
        "edge_attr_policy": (
            "lookup_full_directed_edge_attr"
            if n_missing == 0
            else "lookup_with_edge_length_fallback_for_missing"
        ),
        "outputs": {
            "gnn_arrays_npz": str(out_npz),
        },
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, sort_keys=True)

    print(f"Wrote {out_npz}")
    print(f"Wrote {out_meta}")
    print("Next: build_abacus_sbi_cache.py --gnn-metadata-path", out_meta)


if __name__ == "__main__":
    main()
