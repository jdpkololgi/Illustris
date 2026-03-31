"""METIS-style graph partitioning helpers for Abacus SBI caches.

Builds an *undirected simple* graph from (senders, receivers), runs METIS
(via pymetis when available), and exposes iterators that yield core node ID
chunks grouped by partition — suitable for `build_abacus_partition_batches.py`.

Dependencies:
  - scipy (CSR construction; already typical alongside sklearn in this repo)
  - pymetis + libmetis (optional at runtime; use --metis-partition-npy to skip)

For very large graphs, run METIS elsewhere and pass ``--metis-partition-npy``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


def undirected_csr_from_edges(
    senders: np.ndarray,
    receivers: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return METIS-style CSR (xadj, adjncy) for an undirected simple graph."""
    s = np.asarray(senders, dtype=np.int64).ravel()
    r = np.asarray(receivers, dtype=np.int64).ravel()
    if s.shape[0] != r.shape[0]:
        raise ValueError("senders and receivers must have the same length.")
    m = (s != r) & (s >= 0) & (r >= 0) & (s < n_nodes) & (r < n_nodes)
    s, r = s[m], r[m]
    if s.size == 0:
        xadj = np.zeros(n_nodes + 1, dtype=np.int32)
        return xadj, np.zeros(0, dtype=np.int32)

    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "undirected_csr_from_edges requires scipy. Install scipy or provide "
            "--metis-partition-npy from an external partitioner."
        ) from exc

    row = np.concatenate([s, r])
    col = np.concatenate([r, s])
    agg = sparse.coo_matrix(
        (np.ones(row.shape[0], dtype=np.int8), (row, col)),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    if agg.nnz > 0:
        agg.data.fill(1)
    return agg.indptr.astype(np.int32, copy=False), agg.indices.astype(np.int32, copy=False)


def run_pymetis(
    xadj: np.ndarray,
    adjncy: np.ndarray,
    nparts: int,
) -> np.ndarray:
    """Run METIS; return part id per node, shape (n_nodes,), int32."""
    if nparts < 1:
        raise ValueError("nparts must be >= 1.")
    try:
        import pymetis
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pymetis is not installed. On Perlmutter/NERSC, install pymetis in your "
            "environment, or partition offline and pass --metis-partition-npy "
            f"(uint32/int64 vector of length n_nodes with values in 0..nparts-1). Original: {exc}"
        ) from exc

    n_nodes = int(xadj.shape[0] - 1)
    if nparts > n_nodes:
        nparts = n_nodes

    xadj_list = xadj.tolist()
    adjncy_list = adjncy.tolist()
    _cuts, parts = pymetis.part_graph(nparts, xadj=xadj_list, adjncy=adjncy_list)
    out = np.asarray(parts, dtype=np.int32)
    if out.shape[0] != n_nodes:
        raise RuntimeError(f"METIS returned {out.shape[0]} parts for n_nodes={n_nodes}.")
    return out


def load_or_compute_partition_vector(
    senders: np.ndarray,
    receivers: np.ndarray,
    n_nodes: int,
    nparts: int,
    *,
    partition_npy: Path | None,
) -> tuple[np.ndarray, int]:
    """Load ``part[node]`` from disk or compute with pymetis.

    Returns ``(part_per_node, effective_nparts)`` where labels satisfy
    ``0 <= part < effective_nparts``.
    """
    if partition_npy is not None:
        p = Path(partition_npy).expanduser().resolve()
        vec = np.load(p)
        vec = np.asarray(vec).astype(np.int32, copy=False).ravel()
        if vec.shape[0] != n_nodes:
            raise ValueError(
                f"--metis-partition-npy length {vec.shape[0]} != n_nodes {n_nodes}."
            )
        if vec.size and int(vec.min()) < 0:
            raise ValueError("Partition labels must be non-negative.")
        need = int(vec.max()) + 1 if vec.size else 1
        if nparts > 0 and nparts < need:
            raise ValueError(
                f"--metis-nparts={nparts} is too small: labels require at least {need} parts."
            )
        effective = max(nparts, need) if nparts > 0 else need
        return vec, effective

    xadj, adjncy = undirected_csr_from_edges(senders, receivers, n_nodes)
    print(
        f"METIS CSR: n_nodes={n_nodes:,}, nnz={adjncy.shape[0]:,}, nparts={nparts:,}",
        flush=True,
    )
    out = run_pymetis(xadj, adjncy, nparts)
    return out, nparts


def save_partition_vector(path: Path, part_per_node: np.ndarray) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, part_per_node.astype(np.int32, copy=False))


def iter_metis_core_chunks(
    split_node_ids: np.ndarray,
    part_per_node: np.ndarray,
    max_core_nodes: int,
) -> Iterator[np.ndarray]:
    """Yield sorted core ID arrays: METIS-part groups, chunked by ``max_core_nodes``.

    Nodes in ``split_node_ids`` are grouped by global METIS part id (edge-cut
    locality). Within each group, IDs are sorted; groups larger than
    ``max_core_nodes`` are split into consecutive chunks (still inside the same
    METIS part, so cut edges to *other* parts stay at chunk boundaries only
    when a part is subdivided — internal chunk boundaries may add redundant
    halo but remain graph-local).
    """
    ids = np.asarray(split_node_ids, dtype=np.int64).ravel()
    if ids.size == 0:
        return
    parts = part_per_node[ids]
    order = np.lexsort((ids, parts))
    sorted_ids = ids[order]
    sorted_parts = parts[order]
    boundaries = np.nonzero(sorted_parts[1:] != sorted_parts[:-1])[0] + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), boundaries.astype(np.int64)))
    ends = np.concatenate((boundaries.astype(np.int64), np.array([sorted_ids.size], dtype=np.int64)))

    max_core = int(max_core_nodes) if max_core_nodes > 0 else sorted_ids.size
    for start, end in zip(starts, ends):
        block = sorted_ids[start:end]
        if block.size == 0:
            continue
        for i in range(0, block.size, max_core):
            chunk = block[i : i + max_core]
            yield np.sort(chunk)


def default_metis_nparts(n_nodes: int, core_partition_size: int) -> int:
    """Heuristic initial part count: ~one METIS part per intended core batch."""
    c = max(1, int(core_partition_size))
    return max(1, (int(n_nodes) + c - 1) // c)


def manifest_metis_options(args: Any) -> dict[str, Any]:
    """Subset of argparse namespace to store in partition_manifest.json."""
    return {
        "metis_nparts_requested": int(getattr(args, "metis_nparts", 0)),
        "metis_partition_npy": getattr(args, "metis_partition_npy", None),
        "metis_save_partition_npy": getattr(args, "metis_save_partition_npy", None),
    }
