"""Reusable graph-construction utilities for galaxy workflows."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean


def stack_xyz_points(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return Nx3 cartesian point array from x/y/z arrays."""
    return np.vstack([x, y, z]).T


def _add_nodes(graph: nx.Graph, points: np.ndarray, offset: int = 0) -> None:
    for i, point in enumerate(points):
        idx = i + offset
        graph.add_node(idx, pos=point)


def _add_edges_from_simplices(
    graph: nx.Graph,
    simplices: np.ndarray,
    points: np.ndarray,
    *,
    offset: int = 0,
) -> None:
    for simplex in simplices:
        for j in range(4):
            for k in range(j + 1, 4):
                u_local = int(simplex[j])
                v_local = int(simplex[k])
                u = u_local + offset
                v = v_local + offset
                graph.add_edge(
                    u,
                    v,
                    length=euclidean(points[u_local], points[v_local]),
                )


def build_delaunay_graph(points: np.ndarray) -> tuple[nx.Graph, Delaunay]:
    """Build a weighted undirected graph from a 3D Delaunay triangulation."""
    tri = Delaunay(points)
    graph = nx.Graph()
    _add_nodes(graph, points)
    _add_edges_from_simplices(graph, tri.simplices, points)
    return graph, tri


def build_delaunay_graph_split(
    points_north: np.ndarray,
    points_south: np.ndarray,
) -> tuple[nx.Graph, Delaunay, Delaunay]:
    """
    Build a graph from separate north/south triangulations.

    South node indices are offset by len(points_north).
    """
    tri_north = Delaunay(points_north)
    tri_south = Delaunay(points_south)

    graph = nx.Graph()
    _add_nodes(graph, points_north, offset=0)
    offset = len(points_north)
    _add_nodes(graph, points_south, offset=offset)

    _add_edges_from_simplices(graph, tri_north.simplices, points_north, offset=0)
    _add_edges_from_simplices(graph, tri_south.simplices, points_south, offset=offset)
    return graph, tri_north, tri_south


def build_alpha_complex_graph(
    points: np.ndarray,
    *,
    alpha_sq: float,
) -> tuple[nx.Graph, Any]:
    """Build a graph and simplex tree from a Gudhi alpha complex."""
    try:
        import gudhi
    except ImportError as exc:
        raise ImportError(
            "Gudhi is required for alpha-complex graph construction. "
            "Install `gudhi` in the active environment."
        ) from exc

    alpha_complex = gudhi.AlphaComplex(points=points)
    simplex_tree = alpha_complex.create_simplex_tree()

    graph = nx.Graph()
    _add_nodes(graph, points)

    for simplex, filtration in simplex_tree.get_filtration():
        if len(simplex) == 2 and filtration <= alpha_sq:
            u, v = simplex
            graph.add_edge(
                int(u),
                int(v),
                length=euclidean(points[u], points[v]),
                alpha_filtration=filtration,
            )

    return graph, simplex_tree


def build_alpha_complex_graph_split(
    points_north: np.ndarray,
    points_south: np.ndarray,
    *,
    alpha_sq: float,
) -> tuple[nx.Graph, Any, Any]:
    """
    Build split-hemisphere alpha-complex graph and simplex trees.

    South node indices are offset by len(points_north).
    """
    try:
        import gudhi
    except ImportError as exc:
        raise ImportError(
            "Gudhi is required for alpha-complex graph construction. "
            "Install `gudhi` in the active environment."
        ) from exc

    alpha_complex_n = gudhi.AlphaComplex(points=points_north)
    simplex_tree_n = alpha_complex_n.create_simplex_tree()

    alpha_complex_s = gudhi.AlphaComplex(points=points_south)
    simplex_tree_s = alpha_complex_s.create_simplex_tree()

    graph = nx.Graph()
    _add_nodes(graph, points_north, offset=0)
    offset = len(points_north)
    _add_nodes(graph, points_south, offset=offset)

    for simplex, filtration in simplex_tree_n.get_filtration():
        if len(simplex) == 2 and filtration <= alpha_sq:
            u, v = simplex
            graph.add_edge(
                int(u),
                int(v),
                length=euclidean(points_north[u], points_north[v]),
                alpha_filtration=filtration,
            )

    for simplex, filtration in simplex_tree_s.get_filtration():
        if len(simplex) == 2 and filtration <= alpha_sq:
            u, v = simplex
            graph.add_edge(
                int(u + offset),
                int(v + offset),
                length=euclidean(points_south[u], points_south[v]),
                alpha_filtration=filtration,
            )

    return graph, simplex_tree_n, simplex_tree_s
