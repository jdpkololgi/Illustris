"""
Load TNG300 (sub)halo positions in the *same node ordering* used by the cached graph.

Rationale
---------
The cached `GraphsTuple` was built from a Delaunay triangulation of the positions.
It does not store node xyz positions, but we can re-load them via the same code path
used originally (Utilities.cat via Network_stats.network).

We include a lightweight validation against cached edge direction vectors to catch
any accidental misalignment early (wrong mass cut, different ordering, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import jraph


@dataclass(frozen=True)
class TngPositionConfig:
    masscut: float = 1e9
    from_DESI: bool = False


def load_tng_positions_via_groupcat(masscut: float = 1e9, from_DESI: bool = False) -> np.ndarray:
    """
    Load positions using the same `Network_stats.network` -> `Utilities.cat` path.

    Returns:
        pos: [N, 3] float array (Mpc), where N is expected to match cached graph nodes.
    """
    # Import inside the function to keep this module lightweight unless used.
    from Network_stats import network  # type: ignore

    if from_DESI:
        raise ValueError("This helper is for TNG (from_DESI=False) Phase A.")

    cat_obj = network(masscut=masscut, from_DESI=False)

    # Utilities.cat.readcat sets posx/posy/posz for the *masscut-filtered* sample.
    pos = np.vstack([cat_obj.posx, cat_obj.posy, cat_obj.posz]).T.astype(np.float64)
    return pos


def validate_edge_directions(
    pos: np.ndarray,
    graph: jraph.GraphsTuple,
    n_checks: int = 2048,
    seed: int = 0,
    atol: float = 5e-3,
) -> None:
    """
    Validate that positions align with cached graph edge direction vectors.

    The cached edges (v2) are structured as:
      [scaled_log_length, x_dir, y_dir, z_dir, scaled_log_density_contrast]

    Direction components are *not* scaled in the cache pipeline, so we can compare them
    directly to (pos[v] - pos[u]) / ||pos[v] - pos[u]|| for random edges.
    """
    pos = np.asarray(pos, dtype=np.float64)
    senders = np.asarray(graph.senders, dtype=np.int64)
    receivers = np.asarray(graph.receivers, dtype=np.int64)
    edges = np.asarray(graph.edges)

    if pos.shape[0] != int(graph.nodes.shape[0]):
        raise ValueError(f"pos has N={pos.shape[0]} but graph has N={int(graph.nodes.shape[0])}")
    if edges.ndim != 2 or edges.shape[1] < 4:
        raise ValueError(f"Expected edge feature dim >= 4, got edges.shape={edges.shape}")

    rng = np.random.default_rng(seed)
    E = senders.shape[0]
    idx = rng.integers(0, E, size=min(n_checks, E), dtype=np.int64)

    u = senders[idx]
    v = receivers[idx]
    d = pos[v] - pos[u]
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    unit = d / norm

    cached = edges[idx, 1:4]  # x_dir,y_dir,z_dir

    max_abs = float(np.max(np.abs(unit - cached)))
    mean_abs = float(np.mean(np.abs(unit - cached)))

    if not np.isfinite(max_abs):
        raise ValueError("Direction validation produced non-finite values.")
    if max_abs > atol:
        raise ValueError(
            "Position<->graph index mismatch likely: "
            f"direction vectors differ (max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, atol={atol})"
        )



