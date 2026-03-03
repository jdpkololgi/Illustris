#!/usr/bin/env python3
"""Compute Abacus graph metrics aligned with gcn_paper Network_stats methods.

This script computes and saves three feature tables using the same metric set used by:
- network_stats_delaunay
- network_stats_delaunay2
- network_stats_alpha
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import networkit as nk
import numpy as np
import pandas as pd

# Allow canonical workflow scripts to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.resource_requirements import require_cpu_mpi_slurm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Abacus graph metrics matching Network_stats (alpha/delaunay/delaunay2) "
            "from precomputed graph artifacts."
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
        default="abacus_delaunay",
        help="Artifact prefix (e.g. abacus_delaunay or abacus_alpha).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix for pkl tables. Defaults to <prefix>_graph_features.",
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


def load_inputs(
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
    edges = np.load(edges_file)
    tetrahedra = np.load(tetra_file)
    volumes = np.load(volume_file)

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


def edges_to_networkit(points: np.ndarray, edges: np.ndarray) -> nk.Graph:
    n_nodes = int(points.shape[0])
    graph = nk.Graph(n=n_nodes, weighted=True, directed=False)
    batch_size = 500_000

    print(f"Building Networkit graph with {n_nodes:,} nodes and {len(edges):,} edges...")
    for i in range(0, len(edges), batch_size):
        batch = edges[i : i + batch_size]
        diffs = points[batch[:, 0], :3] - points[batch[:, 1], :3]
        dists = np.linalg.norm(diffs, axis=1)
        for (u, v), w in zip(batch, dists):
            graph.addEdge(int(u), int(v), float(w))
        if i == 0 or ((i // batch_size) + 1) % 10 == 0:
            print(f"  processed {i + len(batch):,}/{len(edges):,} edges")
    return graph


def weighted_degree(graph: nk.Graph) -> np.ndarray:
    n_nodes = graph.numberOfNodes()
    return np.fromiter((graph.weightedDegree(v) for v in graph.iterNodes()), dtype=np.float64, count=n_nodes)


def weighted_clustering(graph: nk.Graph) -> np.ndarray:
    cc = nk.centrality.LocalClusteringCoefficient(graph, turbo=True)
    cc.run()
    return np.array(cc.scores(), dtype=np.float64)


def edge_length_stats(graph: nk.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = graph.numberOfNodes()
    min_lengths = np.zeros(n_nodes, dtype=np.float64)
    max_lengths = np.zeros(n_nodes, dtype=np.float64)
    mean_lengths = np.zeros(n_nodes, dtype=np.float64)

    print("Computing edge-length stats...")
    for node in range(n_nodes):
        if graph.degree(node) > 0:
            neighbors = list(graph.iterNeighbors(node))
            w = np.array([graph.weight(node, nbr) for nbr in neighbors], dtype=np.float64)
            min_lengths[node] = w.min()
            max_lengths[node] = w.max()
            mean_lengths[node] = w.mean()
        if node > 0 and node % 1_000_000 == 0:
            print(f"  processed {node:,} nodes")
    return mean_lengths, min_lengths, max_lengths


def tetra_density(
    tetrahedra: np.ndarray,
    volumes: np.ndarray,
    weighted_deg: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    print("Computing tetrahedral density (Network_stats-compatible)...")
    if tetrahedra.size == 0:
        return np.zeros(n_nodes, dtype=np.float64)

    all_node_indices = tetrahedra.ravel()
    repeated_volumes = np.repeat(volumes, 4)
    node_tetra_count = np.bincount(all_node_indices, minlength=n_nodes).astype(np.float64)
    node_tetra_volume = np.bincount(all_node_indices, weights=repeated_volumes, minlength=n_nodes).astype(np.float64)

    deg_safe = np.where(weighted_deg > 0, weighted_deg, 1.0)
    density = np.zeros(n_nodes, dtype=np.float64)
    mask = node_tetra_volume > 0
    density[mask] = (node_tetra_count[mask] / node_tetra_volume[mask]) / deg_safe[mask]
    return density


def neighbor_density_mean(graph: nk.Graph, density: np.ndarray) -> np.ndarray:
    print("Computing neighbour tetrahedral density (mean)...")
    out = np.zeros(graph.numberOfNodes(), dtype=np.float64)
    for node in graph.iterNodes():
        neighbors = list(graph.iterNeighbors(node))
        if neighbors:
            out[node] = float(np.mean(density[neighbors]))
        if node > 0 and node % 1_000_000 == 0:
            print(f"  processed {node:,} nodes")
    return out


def inertia_eigenvalues(graph: nk.Graph, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("Computing inertia eigenvalues...")
    n_nodes = graph.numberOfNodes()
    eig = np.zeros((n_nodes, 3), dtype=np.float64)
    for node in graph.iterNodes():
        neighbors = list(graph.iterNeighbors(node))
        if len(neighbors) < 3:
            continue
        nbr_pos = points[neighbors, :3]
        center = nbr_pos.mean(axis=0)
        rel = nbr_pos - center
        cov = (rel.T @ rel) / len(neighbors)
        vals = np.linalg.eigvalsh(cov)
        vals[vals < 0] = 0.0
        eig[node] = vals
        if node > 0 and node % 1_000_000 == 0:
            print(f"  processed {node:,} nodes")
    return eig[:, 0], eig[:, 1], eig[:, 2]


def main() -> None:
    args = parse_args()
    require_cpu_mpi_slurm("abacus_graph_features.py", min_tasks=1)

    points, edges, tetrahedra, volumes, metadata, manifest = load_inputs(
        points_path=args.points_path,
        metadata_path=args.metadata_path,
        artifacts_dir=args.artifacts_dir,
        prefix=args.prefix,
    )
    n_nodes = int(points.shape[0])
    print(f"Loaded points={points.shape}, edges={edges.shape}, tetrahedra={tetrahedra.shape}, volumes={volumes.shape}")
    print(f"Using metadata manifest: {manifest}")

    graph = edges_to_networkit(points, edges)
    degree = weighted_degree(graph)
    clustering = weighted_clustering(graph)
    mean_el, min_el, max_el = edge_length_stats(graph)
    density = tetra_density(tetrahedra, volumes, degree, n_nodes=n_nodes)
    neigh_density = neighbor_density_mean(graph, density)
    i1, i2, i3 = inertia_eigenvalues(graph, points)

    # Matches network_stats_delaunay / network_stats_alpha output schema.
    df_delaunay_alpha = pd.DataFrame(
        {
            "Degree": degree,
            "Mean E.L.": mean_el,
            "Min E.L.": min_el,
            "Max E.L.": max_el,
            "Clustering": clustering,
            "Density": density,
            "Neigh Density": neigh_density,
            "I_eig1": i1,
            "I_eig2": i2,
            "I_eig3": i3,
        }
    )

    # Matches network_stats_delaunay2 output schema.
    df_delaunay2 = pd.DataFrame(
        {
            "Degree": degree,
            "Clustering": clustering,
            "Density": density,
            "Neigh Density": neigh_density,
            "I_eig1": i1,
            "I_eig2": i2,
            "I_eig3": i3,
        }
    )

    out_prefix = args.output_prefix or f"{args.prefix}_graph_features"
    out_dir = Path(args.artifacts_dir)
    out_main = out_dir / f"{out_prefix}.pkl"
    out_delaunay2 = out_dir / f"{out_prefix}_delaunay2.pkl"
    out_meta = out_dir / f"{out_prefix}_metrics_metadata.json"

    df_delaunay_alpha.to_pickle(out_main)
    df_delaunay2.to_pickle(out_delaunay2)
    out_meta_payload = {
        "input_metadata_path": str(manifest),
        "input_prefix": metadata.get("prefix"),
        "input_mode": metadata.get("mode"),
        "input_alpha_sq": metadata.get("alpha_sq"),
        "n_points": n_nodes,
        "n_edges": int(edges.shape[0]),
        "n_tetrahedra": int(tetrahedra.shape[0]),
        "node_feature_columns_delaunay_alpha": list(df_delaunay_alpha.columns),
        "node_feature_columns_delaunay2": list(df_delaunay2.columns),
        "outputs": {
            "delaunay_alpha_table": str(out_main),
            "delaunay2_table": str(out_delaunay2),
        },
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(out_meta_payload, f, indent=2, sort_keys=True)

    print(f"Saved: {out_main}")
    print(f"Saved: {out_delaunay2}")
    print(f"Saved: {out_meta}")
    print("Done.")


if __name__ == "__main__":
    main()