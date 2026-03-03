#!/usr/bin/env python3
"""Compute Abacus graph features with cuGraph for GNN training.

This script mirrors the feature intent of `network_stats_delaunay2`:
- Node features: Degree, Clustering, Density, Neigh Density, I_eig1/2/3
- Edge features: edge_length, unit direction vector, density_contrast

Outputs include parquet tables and GNN-ready arrays.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Allow canonical workflow scripts to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.resource_requirements import require_gpu_slurm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cuGraph-based node/edge features for Abacus graphs and export "
            "GNN-ready training artifacts."
        )
    )
    parser.add_argument(
        "--points-path",
        default=None,
        help=(
            "Optional explicit points path override. "
            "By default, points are resolved from metadata.json files map."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help=(
            "Path to graph metadata JSON manifest. Defaults to "
            "<artifacts-dir>/<prefix>_metadata.json."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        default="/pscratch/sd/d/dkololgi/abacus/graph_constructions",
        help="Directory containing graph artifact npy files.",
    )
    parser.add_argument(
        "--prefix",
        default="abacus_alpha",
        help="Artifact prefix (e.g. abacus_alpha or abacus_delaunay).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to --artifacts-dir.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Defaults to <prefix>_cugraph.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2_000_000,
        help="Batch size for vectorized edge-length computation.",
    )
    return parser.parse_args()


def _resolve_input_files(
    points_path: str | None,
    metadata_path: str | None,
    artifacts_dir: str,
    prefix: str,
) -> tuple[dict, Path, Path, Path, Path, Path]:
    base = Path(artifacts_dir)
    manifest = Path(metadata_path) if metadata_path else (base / f"{prefix}_metadata.json")
    if not manifest.exists():
        raise FileNotFoundError(
            f"Metadata manifest not found: {manifest}. "
            "Run build_abacus_graph.py to generate canonical graph artifacts."
        )

    with manifest.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    files = metadata.get("files", {})
    points_file = Path(points_path) if points_path else (base / files.get("points", f"{prefix}_points.npy"))
    edges_file = base / files.get("edges", f"{prefix}_edges_combined_idx.npy")
    tetra_file = base / files.get("tetrahedra_idx", f"{prefix}_tetrahedra_idx.npy")
    volume_file = base / files.get("tetrahedra_volumes", f"{prefix}_tetrahedra_volumes.npy")

    for file_path in (points_file, edges_file, tetra_file, volume_file):
        if not file_path.exists():
            raise FileNotFoundError(f"Required graph artifact missing: {file_path}")

    return metadata, manifest, points_file, edges_file, tetra_file, volume_file


def load_artifacts(
    points_path: str | None,
    metadata_path: str | None,
    artifacts_dir: str,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, Path]:
    metadata, manifest, points_file, edges_file, tetra_file, volume_file = _resolve_input_files(
        points_path=points_path,
        metadata_path=metadata_path,
        artifacts_dir=artifacts_dir,
        prefix=prefix,
    )

    points = np.load(points_file).astype(np.float64)
    edges = np.load(edges_file).astype(np.int64)
    tetrahedra = np.load(tetra_file).astype(np.int64)
    volumes = np.load(volume_file).astype(np.float64)

    expected_n_points = metadata.get("n_points")
    if expected_n_points is not None and int(expected_n_points) != int(points.shape[0]):
        raise ValueError(
            f"Metadata n_points={expected_n_points} does not match points rows={points.shape[0]} "
            f"from {points_file}"
        )

    expected_n_edges = metadata.get("n_edges")
    if expected_n_edges is not None and int(expected_n_edges) != int(edges.shape[0]):
        raise ValueError(
            f"Metadata n_edges={expected_n_edges} does not match edges rows={edges.shape[0]} "
            f"from {edges_file}"
        )

    expected_n_tets = metadata.get("n_tetrahedra")
    if expected_n_tets is not None and int(expected_n_tets) != int(tetrahedra.shape[0]):
        raise ValueError(
            f"Metadata n_tetrahedra={expected_n_tets} does not match tetrahedra rows={tetrahedra.shape[0]} "
            f"from {tetra_file}"
        )

    return points, edges, tetrahedra, volumes, metadata, manifest


def edge_lengths(points: np.ndarray, edges: np.ndarray, batch_size: int) -> np.ndarray:
    out = np.empty(len(edges), dtype=np.float64)
    for i in range(0, len(edges), batch_size):
        batch = edges[i : i + batch_size]
        diffs = points[batch[:, 0], :3] - points[batch[:, 1], :3]
        out[i : i + len(batch)] = np.linalg.norm(diffs, axis=1)
    return out


def weighted_degree_from_edges(n_nodes: int, edges: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    deg = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(deg, edges[:, 0], lengths)
    np.add.at(deg, edges[:, 1], lengths)
    return deg


def unweighted_degree_from_edges(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    deg = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    return deg


def cugraph_local_clustering_from_triangles(
    n_nodes: int,
    edges: np.ndarray,
    degree_u: np.ndarray,
) -> np.ndarray:
    import cudf
    import cugraph

    df = cudf.DataFrame(
        {
            "src": edges[:, 0].astype(np.int32),
            "dst": edges[:, 1].astype(np.int32),
        }
    )
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_edgelist(df, source="src", destination="dst", renumber=False)

    # Local clustering coefficient from triangles:
    # C_i = 2*T_i / (k_i * (k_i - 1)) for k_i >= 2, else 0
    tc = cugraph.triangle_count(graph)
    tri_counts = tc["counts"].to_numpy()
    tri_vertices = tc["vertex"].to_numpy()
    out = np.zeros(n_nodes, dtype=np.float64)
    out[tri_vertices] = tri_counts.astype(np.float64)

    k = degree_u.astype(np.float64)
    denom = k * (k - 1.0)
    valid = denom > 0
    out[valid] = (2.0 * out[valid]) / denom[valid]
    out[~valid] = 0.0
    return out


def tetra_density(n_nodes: int, tetrahedra: np.ndarray, volumes: np.ndarray, weighted_degree: np.ndarray) -> np.ndarray:
    if tetrahedra.size == 0:
        return np.zeros(n_nodes, dtype=np.float64)

    node_indices = tetrahedra.ravel()
    repeated_vol = np.repeat(volumes, 4)

    node_tetra_count = np.bincount(node_indices, minlength=n_nodes).astype(np.float64)
    node_tetra_vol = np.bincount(node_indices, weights=repeated_vol, minlength=n_nodes).astype(np.float64)

    deg_safe = np.where(weighted_degree > 0, weighted_degree, 1.0)
    dens = np.zeros(n_nodes, dtype=np.float64)
    mask = node_tetra_vol > 0
    dens[mask] = (node_tetra_count[mask] / node_tetra_vol[mask]) / deg_safe[mask]
    return dens


def neighbor_density_mean(n_nodes: int, edges: np.ndarray, density: np.ndarray) -> np.ndarray:
    neigh_sum = np.zeros(n_nodes, dtype=np.float64)
    neigh_cnt = np.zeros(n_nodes, dtype=np.int64)

    np.add.at(neigh_sum, edges[:, 0], density[edges[:, 1]])
    np.add.at(neigh_sum, edges[:, 1], density[edges[:, 0]])
    np.add.at(neigh_cnt, edges[:, 0], 1)
    np.add.at(neigh_cnt, edges[:, 1], 1)

    out = np.zeros(n_nodes, dtype=np.float64)
    mask = neigh_cnt > 0
    out[mask] = neigh_sum[mask] / neigh_cnt[mask]
    return out


def inertia_eigs(points: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = len(points)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.int8)
    adj = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

    eig = np.zeros((n_nodes, 3), dtype=np.float64)
    indptr = adj.indptr
    indices = adj.indices

    for node in range(n_nodes):
        nbr = indices[indptr[node] : indptr[node + 1]]
        if len(nbr) < 3:
            continue
        nbr_pos = points[nbr, :3]
        center = nbr_pos.mean(axis=0)
        rel = nbr_pos - center
        cov = (rel.T @ rel) / len(nbr)
        vals = np.linalg.eigvalsh(cov)
        vals[vals < 0] = 0.0
        eig[node] = vals
        if node > 0 and node % 1_000_000 == 0:
            print(f"  inertia processed {node:,} nodes")

    return eig[:, 0], eig[:, 1], eig[:, 2]


def edge_feature_table(points: np.ndarray, edges: np.ndarray, lengths: np.ndarray, density: np.ndarray) -> pd.DataFrame:
    vec = points[edges[:, 1], :3] - points[edges[:, 0], :3]
    denom = np.where(lengths > 0, lengths, 1.0)
    unit = vec / denom[:, None]

    src_d = density[edges[:, 0]]
    dst_d = density[edges[:, 1]]
    density_contrast_along_edge = np.zeros_like(lengths, dtype=np.float64)
    mask = src_d > 0
    density_contrast_along_edge[mask] = dst_d[mask] / src_d[mask]

    return pd.DataFrame(
        {
            "src": edges[:, 0],
            "dst": edges[:, 1],
            "edge_length": lengths,
            "x_dir": unit[:, 0],
            "y_dir": unit[:, 1],
            "z_dir": unit[:, 2],
            # Keep both names so this matches Network_stats intent while
            # staying backward compatible with earlier exports.
            "density_contrast_along_edge": density_contrast_along_edge,
            "density_contrast": density_contrast_along_edge,
        }
    )


def save_gnn_arrays(
    out_npz: Path,
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    node_cols = ["Degree", "Clustering", "Density", "Neigh Density", "I_eig1", "I_eig2", "I_eig3"]
    edge_cols = ["edge_length", "x_dir", "y_dir", "z_dir", "density_contrast"]

    x = node_df[["Degree", "Clustering", "Density", "Neigh Density", "I_eig1", "I_eig2", "I_eig3"]].to_numpy(
        dtype=np.float32
    )
    edge_index = edge_df[["src", "dst"]].to_numpy(dtype=np.int64).T
    edge_attr = edge_df[edge_cols].to_numpy(dtype=np.float32)
    np.savez_compressed(out_npz, x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Optional torch export for PyG convenience.
    try:
        import torch

        pt_path = out_npz.with_suffix(".pt")
        torch.save(
            {
                "x": torch.from_numpy(x),
                "edge_index": torch.from_numpy(edge_index),
                "edge_attr": torch.from_numpy(edge_attr),
            },
            pt_path,
        )
        print(f"Saved PyTorch tensor bundle: {pt_path}")
    except Exception:
        pass

    return node_cols, edge_cols


def main() -> None:
    args = parse_args()
    require_gpu_slurm("abacus_graph_features_cugraph.py", min_gpus=1)

    points, edges, tetrahedra, volumes, metadata, manifest = load_artifacts(
        points_path=args.points_path,
        metadata_path=args.metadata_path,
        artifacts_dir=args.artifacts_dir,
        prefix=args.prefix,
    )
    n_nodes = points.shape[0]
    print(f"Loaded points={points.shape}, edges={edges.shape}, tetrahedra={tetrahedra.shape}")
    print(f"Using metadata manifest: {manifest}")

    lengths = edge_lengths(points, edges, args.batch_size)
    degree_w = weighted_degree_from_edges(n_nodes, edges, lengths)
    degree_u = unweighted_degree_from_edges(n_nodes, edges)
    clustering = cugraph_local_clustering_from_triangles(n_nodes, edges, degree_u)
    density = tetra_density(n_nodes, tetrahedra, volumes, degree_w)
    neigh_density = neighbor_density_mean(n_nodes, edges, density)
    i1, i2, i3 = inertia_eigs(points, edges)

    node_df = pd.DataFrame(
        {
            "Node ID": np.arange(n_nodes, dtype=np.int64),
            "Degree": degree_w,
            "Clustering": clustering,
            "Density": density,
            "Neigh Density": neigh_density,
            "I_eig1": i1,
            "I_eig2": i2,
            "I_eig3": i3,
        }
    )
    edge_df = edge_feature_table(points, edges, lengths, density)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = args.output_prefix or f"{args.prefix}_cugraph"

    node_path = out_dir / f"{out_prefix}_node_features.parquet"
    edge_path = out_dir / f"{out_prefix}_edge_features.parquet"
    gnn_npz = out_dir / f"{out_prefix}_gnn_arrays.npz"
    gnn_meta = out_dir / f"{out_prefix}_gnn_metadata.json"

    node_df.to_parquet(node_path, index=False)
    edge_df.to_parquet(edge_path, index=False)
    node_cols, edge_cols = save_gnn_arrays(gnn_npz, node_df, edge_df)

    out_meta_payload = {
        "input_metadata_path": str(manifest),
        "input_prefix": metadata.get("prefix"),
        "input_mode": metadata.get("mode"),
        "input_alpha_sq": metadata.get("alpha_sq"),
        "n_points": int(n_nodes),
        "n_edges": int(edges.shape[0]),
        "n_tetrahedra": int(tetrahedra.shape[0]),
        "node_feature_columns": node_cols,
        "edge_feature_columns": edge_cols,
        "outputs": {
            "node_features": str(node_path),
            "edge_features": str(edge_path),
            "gnn_arrays_npz": str(gnn_npz),
        },
    }
    with gnn_meta.open("w", encoding="utf-8") as f:
        json.dump(out_meta_payload, f, indent=2, sort_keys=True)

    print(f"Saved node features: {node_path}")
    print(f"Saved edge features: {edge_path}")
    print(f"Saved GNN arrays: {gnn_npz}")
    print(f"Saved GNN metadata: {gnn_meta}")
    print("Done.")


if __name__ == "__main__":
    main()
