"""Build an SBI-compatible cache from Abacus graph-feature artifacts.

This script converts the outputs referenced by
`abacus_alpha_cugraph_gnn_metadata.json` (NPZ node/edge arrays) plus T-Web
targets from the source FITS catalog into the cache schema consumed by:
`workflows/sbi/jraph_sbi_flowjax.py`.

Output pickle keys:
  - graph: jraph.GraphsTuple
  - regression_targets: jnp.ndarray [N, 3]
  - masks: tuple(train_mask, val_mask, test_mask) as jnp.bool arrays
  - target_scaler: sklearn StandardScaler fitted on train split only
  - eigenvalues_raw: np.ndarray [N, 3] float64
  - stats: dict (only for transformed-target mode)
  - classification_labels: optional CWEB labels (when available)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Iterable

import fitsio
import jax.numpy as jnp
import jraph
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Allow workflow script to resolve repo-root modules after reorganization.
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.eigenvalue_transformations import eigenvalues_to_increments


def _resolve_col(table, candidates: Iterable[str]) -> str:
    names_upper = {name.upper(): name for name in table.dtype.names}
    for candidate in candidates:
        resolved = names_upper.get(candidate.upper())
        if resolved is not None:
            return resolved
    raise KeyError(
        f"None of the candidate columns {list(candidates)} found. "
        f"Available columns include: {table.dtype.names[:20]}..."
    )


def _apply_optional_y1y5_filter(table: np.ndarray) -> np.ndarray:
    names_upper = {name.upper(): name for name in table.dtype.names}
    in_y1 = names_upper.get("IN_Y1")
    in_y5 = names_upper.get("IN_Y5")
    if in_y1 is None or in_y5 is None:
        return np.ones(len(table), dtype=bool)
    return (table[in_y1] == 1) | (table[in_y5] == 1)


def _load_targets_from_source_catalog(
    source_path: Path,
    expected_n: int,
    *,
    apply_y1y5_filter: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    table = fitsio.read(str(source_path))
    mask = np.ones(len(table), dtype=bool)
    if apply_y1y5_filter:
        mask &= _apply_optional_y1y5_filter(table)

    l1_col = _resolve_col(table, ("LAMBDA1", "L1", "EIG1", "LAM1", "LAMBDA_1"))
    l2_col = _resolve_col(table, ("LAMBDA2", "L2", "EIG2", "LAM2", "LAMBDA_2"))
    l3_col = _resolve_col(table, ("LAMBDA3", "L3", "EIG3", "LAM3", "LAMBDA_3"))

    eig = np.stack(
        [table[l1_col][mask], table[l2_col][mask], table[l3_col][mask]],
        axis=-1,
    ).astype(np.float64)

    cweb = None
    try:
        cweb_col = _resolve_col(table, ("CWEB", "TARGET", "LABEL"))
        cweb = np.asarray(table[cweb_col][mask], dtype=np.int32)
    except KeyError:
        pass

    if eig.shape[0] != expected_n:
        raise ValueError(
            "Target row count mismatch after filtering. "
            f"Expected {expected_n:,} rows from graph arrays but got {eig.shape[0]:,}. "
            "Try toggling --apply-y1y5-filter / --no-apply-y1y5-filter based on how the graph was built."
        )
    return eig, cweb


def _make_splits(
    n_nodes: int,
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    stratify_labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.")

    all_idx = np.arange(n_nodes, dtype=np.int64)
    strat_base = stratify_labels if stratify_labels is not None else None

    # First split: train vs remainder
    train_idx, rem_idx = train_test_split(
        all_idx,
        test_size=(1.0 - train_frac),
        random_state=seed,
        stratify=strat_base,
    )

    # Second split: val vs test from remainder
    rem_strat = stratify_labels[rem_idx] if stratify_labels is not None else None
    val_over_rem = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        rem_idx,
        train_size=val_over_rem,
        random_state=seed,
        stratify=rem_strat,
    )

    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_idx, val_idx, test_idx, train_mask, val_mask, test_mask


def _build_graph_from_npz(
    npz_path: Path,
    *,
    make_bidirectional: bool,
    scale_edge_length_density: bool,
) -> jraph.GraphsTuple:
    with np.load(npz_path) as data:
        x = data["x"].astype(np.float32)
        edge_index = data["edge_index"].astype(np.int64)  # [2, E]
        edge_attr = data["edge_attr"].astype(np.float32)  # [E, 5]

    senders = edge_index[0]
    receivers = edge_index[1]

    if make_bidirectional:
        rev_edge_attr = edge_attr.copy()
        rev_edge_attr[:, 1] *= -1.0
        rev_edge_attr[:, 2] *= -1.0
        rev_edge_attr[:, 3] *= -1.0
        rev_edge_attr[:, 4] = 1.0 / np.maximum(rev_edge_attr[:, 4], 1e-6)

        orig_senders = senders
        orig_receivers = receivers
        senders = np.concatenate([orig_senders, orig_receivers], axis=0)
        receivers = np.concatenate([orig_receivers, orig_senders], axis=0)
        edge_attr = np.concatenate([edge_attr, rev_edge_attr], axis=0)

    if scale_edge_length_density:
        edge_attr = edge_attr.copy()
        edge_attr[:, 0] = np.log(np.maximum(edge_attr[:, 0], 1e-6))
        edge_attr[:, 4] = np.log(np.maximum(edge_attr[:, 4], 1e-6))
        scaler_edge = StandardScaler()
        edge_attr[:, [0, 4]] = scaler_edge.fit_transform(edge_attr[:, [0, 4]])

    n_nodes = x.shape[0]
    n_edges = senders.shape[0]
    return jraph.GraphsTuple(
        nodes=jnp.array(x, dtype=jnp.float32),
        edges=jnp.array(edge_attr, dtype=jnp.float32),
        senders=jnp.array(senders, dtype=jnp.int32),
        receivers=jnp.array(receivers, dtype=jnp.int32),
        n_node=jnp.array([n_nodes], dtype=jnp.int32),
        n_edge=jnp.array([n_edges], dtype=jnp.int32),
        globals=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gnn-metadata-path",
        required=True,
        help="Path to <prefix>_cugraph_gnn_metadata.json",
    )
    parser.add_argument(
        "--output-cache-path",
        required=True,
        help="Where to write SBI cache pickle (.pkl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test splitting.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Training split fraction.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.21,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.09,
        help="Test split fraction.",
    )
    parser.add_argument(
        "--no-transformed-eig",
        action="store_true",
        help="Use raw eigenvalues as targets instead of transformed increments.",
    )
    parser.add_argument(
        "--apply-y1y5-filter",
        action="store_true",
        default=True,
        help="Apply IN_Y1/IN_Y5 filter when loading targets from source catalog (default: true).",
    )
    parser.add_argument(
        "--no-apply-y1y5-filter",
        dest="apply_y1y5_filter",
        action="store_false",
        help="Disable IN_Y1/IN_Y5 filtering.",
    )
    parser.add_argument(
        "--no-bidirectional-edges",
        action="store_true",
        help="Keep edges as stored in NPZ instead of duplicating reverse direction.",
    )
    parser.add_argument(
        "--no-edge-v2-scaling",
        action="store_true",
        help="Disable log+standard scaling on edge length/density_contrast.",
    )
    parser.add_argument(
        "--allow-login-node",
        action="store_true",
        help=(
            "Allow execution outside a Slurm allocation. "
            "Intended only for tiny smoke tests; full Abacus conversion should run on compute nodes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Full Abacus conversion is heavy I/O + memory. Require compute allocation by default.
    if "SLURM_JOB_ID" not in os.environ and not args.allow_login_node:
        raise RuntimeError(
            "build_abacus_sbi_cache.py should run in a Slurm compute allocation "
            "(CPU or GPU node). Re-run via salloc/sbatch, or pass --allow-login-node "
            "only for very small smoke tests."
        )

    meta_path = Path(args.gnn_metadata_path).expanduser().resolve()
    out_cache = Path(args.output_cache_path).expanduser().resolve()
    out_cache.parent.mkdir(parents=True, exist_ok=True)

    with meta_path.open("r", encoding="utf-8") as f:
        gnn_meta = json.load(f)

    npz_path = Path(gnn_meta["outputs"]["gnn_arrays_npz"]).expanduser().resolve()
    input_meta_path = Path(gnn_meta["input_metadata_path"]).expanduser().resolve()

    with input_meta_path.open("r", encoding="utf-8") as f:
        graph_meta = json.load(f)
    source_catalog = Path(graph_meta["source_path"]).expanduser().resolve()

    print(f"Loading graph arrays from: {npz_path}")
    graph = _build_graph_from_npz(
        npz_path,
        make_bidirectional=not args.no_bidirectional_edges,
        scale_edge_length_density=not args.no_edge_v2_scaling,
    )
    n_nodes = int(graph.n_node[0])
    n_edges = int(graph.n_edge[0])
    print(f"Graph ready: nodes={n_nodes:,}, edges={n_edges:,}")

    print(f"Loading targets from source catalog: {source_catalog}")
    eigenvalues_raw, cweb = _load_targets_from_source_catalog(
        source_catalog,
        n_nodes,
        apply_y1y5_filter=args.apply_y1y5_filter,
    )

    train_idx, val_idx, test_idx, train_mask, val_mask, test_mask = _make_splits(
        n_nodes,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        stratify_labels=cweb,
    )

    print(
        "Split sizes: "
        f"train={train_mask.sum():,}, val={val_mask.sum():,}, test={test_mask.sum():,}"
    )

    target_scaler = StandardScaler()
    use_transformed_eig = not args.no_transformed_eig

    stats = None
    if use_transformed_eig:
        transformed = np.array(eigenvalues_to_increments(jnp.array(eigenvalues_raw)))
        target_scaler.fit(transformed[train_idx])
        transformed_scaled = target_scaler.transform(transformed)
        regression_targets = jnp.array(transformed_scaled, dtype=jnp.float64)

        scaled_min = np.min(transformed_scaled[train_idx], axis=0)
        scaled_max = np.max(transformed_scaled[train_idx], axis=0)
        stats = {
            "v1_min_scaled": float(scaled_min[0]),
            "v1_max_scaled": float(scaled_max[0]),
            "target_min": scaled_min.tolist(),
            "target_max": scaled_max.tolist(),
            "scaler_mean": target_scaler.mean_.tolist(),
            "scaler_std": target_scaler.scale_.tolist(),
        }
    else:
        target_scaler.fit(eigenvalues_raw[train_idx])
        scaled = target_scaler.transform(eigenvalues_raw)
        regression_targets = jnp.array(scaled, dtype=jnp.float32)

    payload = {
        "graph": graph,
        "regression_targets": regression_targets,
        "target_scaler": target_scaler,
        "eigenvalues_raw": eigenvalues_raw.astype(np.float64),
        "masks": (jnp.array(train_mask), jnp.array(val_mask), jnp.array(test_mask)),
        "stats": stats,
    }
    if cweb is not None:
        payload["classification_labels"] = jnp.array(cweb, dtype=jnp.int32)

    with out_cache.open("wb") as f:
        pickle.dump(payload, f)

    mode_name = "transformed (v1, Δλ2, Δλ3)" if use_transformed_eig else "raw scaled (λ1, λ2, λ3)"
    print(f"Target mode: {mode_name}")
    print(f"Wrote SBI cache: {out_cache}")


if __name__ == "__main__":
    main()
