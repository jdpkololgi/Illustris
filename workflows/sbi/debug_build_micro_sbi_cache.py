#!/usr/bin/env python3
"""Build a micro SBI cache from a single partition NPZ for A/B debugging.

This creates a cache artifact compatible with `jraph_sbi_flowjax.py` from:
- one micro partition NPZ (`x`, `edge_index`, `edge_attr`, `targets`)
- a source SBI cache (for target_scaler and optional metadata)

The output cache includes:
- graph (jraph.GraphsTuple)
- regression_targets
- masks (train/val/test)  [all-true by default for strict overfit diagnostics]
- target_scaler
- eigenvalues_raw
- stats (if present in source cache)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import jax.numpy as jnp
import jraph
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.eigenvalue_transformations import samples_to_raw_eigenvalues


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partition-npz", required=True, help="Path to micro partition NPZ.")
    p.add_argument("--source-cache-path", required=True, help="Path to source SBI cache (.pkl).")
    p.add_argument("--output-cache-path", required=True, help="Path for output micro SBI cache (.pkl).")
    p.add_argument(
        "--mask-mode",
        choices=("all_true", "split_80_10_10"),
        default="all_true",
        help="Mask generation mode. all_true is best for strict overfit A/B tests.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed used when mask-mode=split_80_10_10.")
    p.add_argument(
        "--use-transformed-eig",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether targets are transformed increments (v1, d2, d3).",
    )
    return p.parse_args()


def _make_masks(n: int, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "all_true":
        m = np.ones(n, dtype=bool)
        return m.copy(), m.copy(), m.copy()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    train[train_idx] = True
    val[val_idx] = True
    test[test_idx] = True
    return train, val, test


def main() -> None:
    args = parse_args()
    part_path = Path(args.partition_npz).expanduser().resolve()
    src_cache = Path(args.source_cache_path).expanduser().resolve()
    out_cache = Path(args.output_cache_path).expanduser().resolve()
    out_cache.parent.mkdir(parents=True, exist_ok=True)

    with part_path.open("rb"):
        pass
    with src_cache.open("rb"):
        pass

    with np.load(part_path) as d:
        x = np.asarray(d["x"], dtype=np.float32)
        edge_index = np.asarray(d["edge_index"], dtype=np.int32)
        edge_attr = np.asarray(d["edge_attr"], dtype=np.float32)
        targets = np.asarray(d["targets"], dtype=np.float32)

    with src_cache.open("rb") as f:
        src = pickle.load(f)

    target_scaler = src["target_scaler"]
    stats = src.get("stats")
    classification_labels = src.get("classification_labels")

    graph = jraph.GraphsTuple(
        nodes=jnp.asarray(x, dtype=jnp.float32),
        edges=jnp.asarray(edge_attr, dtype=jnp.float32),
        senders=jnp.asarray(edge_index[0], dtype=jnp.int32),
        receivers=jnp.asarray(edge_index[1], dtype=jnp.int32),
        n_node=jnp.asarray([x.shape[0]], dtype=jnp.int32),
        n_edge=jnp.asarray([edge_index.shape[1]], dtype=jnp.int32),
        globals=None,
    )

    train_mask, val_mask, test_mask = _make_masks(x.shape[0], args.mask_mode, args.seed)
    eigenvalues_raw = samples_to_raw_eigenvalues(targets, target_scaler, args.use_transformed_eig).astype(np.float64)

    payload = {
        "graph": graph,
        "regression_targets": jnp.asarray(targets, dtype=jnp.float32),
        "masks": (jnp.asarray(train_mask), jnp.asarray(val_mask), jnp.asarray(test_mask)),
        "target_scaler": target_scaler,
        "eigenvalues_raw": eigenvalues_raw,
        "stats": stats,
    }

    if classification_labels is not None:
        cls = np.asarray(classification_labels)
        if cls.shape[0] >= x.shape[0]:
            payload["classification_labels"] = jnp.asarray(cls[: x.shape[0]], dtype=jnp.int32)

    with out_cache.open("wb") as f:
        pickle.dump(payload, f)

    print("=" * 72)
    print("Micro SBI cache created")
    print("=" * 72)
    print(f"Partition NPZ: {part_path}")
    print(f"Source cache : {src_cache}")
    print(f"Output cache : {out_cache}")
    print(f"Nodes/edges  : {x.shape[0]} / {edge_index.shape[1]}")
    print(f"Mask mode    : {args.mask_mode}")
    print(f"use_transformed_eig: {args.use_transformed_eig}")
    print("=" * 72)


if __name__ == "__main__":
    main()

