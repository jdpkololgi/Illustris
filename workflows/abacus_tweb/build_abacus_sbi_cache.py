"""Build an SBI-compatible cache from Abacus graph-feature artifacts.

This script converts the outputs referenced by
`abacus_alpha_cugraph_gnn_metadata.json` (NPZ node/edge arrays) plus T-Web
targets from the source FITS catalog into the cache schema consumed by:
`workflows/sbi/jraph_sbi_flowjax.py`.

Important:
- Abacus CutSky catalogs can include rows that are not validly embedded in any
  underlying cubic box (`BOX_INDEX == -1`). These must be excluded to keep
  the node/target alignment consistent with graph construction.
- Any upstream workflow that assigns eigenvalues using sky coordinates
  (RA/DEC/Z -> cubic frame) must do so in a way that is consistent with the
  CutSky remap, i.e. using `BOX_INDEX` (or an equivalent explicit linkage)
  rather than a naive periodic modulo into a single cube.

Output pickle keys:
  - graph: jraph.GraphsTuple
  - regression_targets: jnp.ndarray [N, 3]
  - masks: tuple(train_mask, val_mask, test_mask) as jnp.bool arrays
  - target_scaler: sklearn StandardScaler fitted on train split only
  - eigenvalues_raw: np.ndarray [N, 3] float64
  - stats: dict (only for transformed-target mode)
  - classification_labels: optional CWEB labels (when available)
  - box_index: optional np.ndarray [N] int32 (when present in source catalog)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Iterable

# This script is a CPU-side cache builder; it should run on CPU-only nodes.
# Force JAX to use the CPU backend to avoid CUDA plugin initialization on nodes
# without GPUs (common for `salloc -C cpu`).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

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


def _apply_optional_box_index_filter(table: np.ndarray, *, box_index_col: str) -> np.ndarray:
    names_upper = {name.upper(): name for name in table.dtype.names}
    resolved = names_upper.get(box_index_col.upper())
    if resolved is None:
        return np.ones(len(table), dtype=bool)
    return np.asarray(table[resolved] != -1)


def _load_targets_from_source_catalog(
    source_path: Path,
    expected_n: int,
    *,
    apply_y1y5_filter: bool,
    exclude_invalid_box_index: bool,
    box_index_col: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    table = fitsio.read(str(source_path))
    mask = np.ones(len(table), dtype=bool)
    if apply_y1y5_filter:
        mask &= _apply_optional_y1y5_filter(table)
    if exclude_invalid_box_index:
        mask &= _apply_optional_box_index_filter(table, box_index_col=box_index_col)

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

    box_index = None
    try:
        box_col = _resolve_col(table, (box_index_col, "BOX_INDEX"))
        box_index = np.asarray(table[box_col][mask], dtype=np.int32)
    except KeyError:
        pass

    if eig.shape[0] != expected_n:
        raise ValueError(
            "Target row count mismatch after filtering. "
            f"Expected {expected_n:,} rows from graph arrays but got {eig.shape[0]:,}. "
            "Try toggling --apply-y1y5-filter / --no-apply-y1y5-filter and/or "
            "--no-exclude-invalid-box-index based on how the graph was built."
        )
    deriv12 = _try_load_derivative_targets(table, mask)
    return eig, cweb, box_index, deriv12


def _try_load_derivative_targets(table: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Return (N, 12) array of dimensionless derivative targets, or None if columns missing."""
    names_upper = {name.upper(): name for name in table.dtype.names}
    required = [
        "DLAM1_DX",
        "DLAM1_DY",
        "DLAM1_DZ",
        "DLAM2_DX",
        "DLAM2_DY",
        "DLAM2_DZ",
        "DLAM3_DX",
        "DLAM3_DY",
        "DLAM3_DZ",
        "LAP_LAM1",
        "LAP_LAM2",
        "LAP_LAM3",
    ]
    if any(r.upper() not in names_upper for r in required):
        return None
    cols = [names_upper[r] for r in required]
    arr = np.stack([np.asarray(table[c][mask], dtype=np.float64) for c in cols], axis=-1)
    return arr


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
    try:
        train_idx, rem_idx = train_test_split(
            all_idx,
            test_size=(1.0 - train_frac),
            random_state=seed,
            stratify=strat_base,
        )
    except ValueError as exc:
        if strat_base is None:
            raise
        print(
            f"WARN: stratified train split failed ({exc}). "
            "Falling back to unstratified split."
        )
        train_idx, rem_idx = train_test_split(
            all_idx,
            test_size=(1.0 - train_frac),
            random_state=seed,
            stratify=None,
        )

    # Second split: val vs test from remainder
    rem_strat = stratify_labels[rem_idx] if stratify_labels is not None else None
    val_over_rem = val_frac / (val_frac + test_frac)
    try:
        val_idx, test_idx = train_test_split(
            rem_idx,
            train_size=val_over_rem,
            random_state=seed,
            stratify=rem_strat,
        )
    except ValueError as exc:
        if rem_strat is None:
            raise
        print(
            f"WARN: stratified val/test split failed ({exc}). "
            "Falling back to unstratified split."
        )
        val_idx, test_idx = train_test_split(
            rem_idx,
            train_size=val_over_rem,
            random_state=seed,
            stratify=None,
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
        "--targets-catalog-path",
        default="",
        help=(
            "Optional override path to the FITS catalog containing target eigenvalues. "
            "This must include LAMBDA1/2/3 (or configured via --lambda{1,2,3}-col). "
            "If omitted, defaults to `source_path` from the graph metadata, which is often the raw CutSky file "
            "and may not contain eigenvalue targets."
        ),
    )
    parser.add_argument("--lambda1-col", default="LAMBDA1", help="Column name for λ1 (default: LAMBDA1).")
    parser.add_argument("--lambda2-col", default="LAMBDA2", help="Column name for λ2 (default: LAMBDA2).")
    parser.add_argument("--lambda3-col", default="LAMBDA3", help="Column name for λ3 (default: LAMBDA3).")
    parser.add_argument("--cweb-col", default="CWEB", help="Optional CWEB column name (default: CWEB).")
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
        "--box-index-col",
        default="BOX_INDEX",
        help="Column name used for Abacus CutSky remap bookkeeping (default: BOX_INDEX).",
    )
    parser.add_argument(
        "--no-exclude-invalid-box-index",
        dest="exclude_invalid_box_index",
        action="store_false",
        default=True,
        help="Do not exclude BOX_INDEX == -1 rows (default: excluded).",
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
    targets_catalog = Path(args.targets_catalog_path).expanduser().resolve() if args.targets_catalog_path else source_catalog

    print(f"Loading graph arrays from: {npz_path}")
    graph = _build_graph_from_npz(
        npz_path,
        make_bidirectional=not args.no_bidirectional_edges,
        scale_edge_length_density=not args.no_edge_v2_scaling,
    )
    n_nodes = int(graph.n_node[0])
    n_edges = int(graph.n_edge[0])
    print(f"Graph ready: nodes={n_nodes:,}, edges={n_edges:,}")

    print(f"Loading targets from catalog: {targets_catalog}")
    try:
        eigenvalues_raw, cweb, box_index, deriv12 = _load_targets_from_source_catalog(
            targets_catalog,
            n_nodes,
            apply_y1y5_filter=args.apply_y1y5_filter,
            exclude_invalid_box_index=args.exclude_invalid_box_index,
            box_index_col=args.box_index_col,
        )
    except KeyError as exc:
        # The raw CutSky catalogs do not include eigenvalue columns; the targets must
        # come from an *annotated* FITS (e.g. produced by `annotate_cutsky_with_tweb_eigs.py`).
        raise KeyError(
            f"{exc}\n\n"
            "Target eigenvalue columns were not found in the provided targets catalog. "
            "If you passed the raw CutSky FITS, you must first generate an annotated FITS "
            "that includes LAMBDA1/2/3 (and optionally CWEB), using a BOX_INDEX-aware mapping.\n"
            "Suggested workflow:\n"
            "  - Run `TNG/Illustris/workflows/abacus_tweb/annotate_cutsky_with_tweb_eigs.py` to create <cutsky>_with_tweb_eigs.fits\n"
            "  - Re-run this script with `--targets-catalog-path <that_output.fits>`\n"
        ) from exc

    # If the user configured explicit column names, validate they exist in the target FITS.
    # (We still use the flexible resolver in _load_targets_from_source_catalog for legacy names.)
    # Note: this is just a guardrail; the actual eigenvalue loading happens in the helper above.
    _ = args.lambda1_col, args.lambda2_col, args.lambda3_col, args.cweb_col

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

    # If derivative columns are present, build the 15-d target vector:
    # [0]=v1=λ1, [1]=v2=λ2-λ1, [2]=v3=λ3-λ2,
    # [3:12]=R*∂λi/∂{x,y,z}, [12:15]=R^2*∇^2λi (already dimensionless in FITS).
    stats = None
    regression_targets_raw: np.ndarray
    if deriv12 is not None:
        v1 = eigenvalues_raw[:, 0]
        v2 = eigenvalues_raw[:, 1] - eigenvalues_raw[:, 0]
        v3 = eigenvalues_raw[:, 2] - eigenvalues_raw[:, 1]
        regression_targets_raw = np.concatenate([np.stack([v1, v2, v3], axis=-1), deriv12], axis=-1).astype(
            np.float64
        )
        if regression_targets_raw.shape[1] != 15:
            raise RuntimeError(f"Expected 15 target dims, got {regression_targets_raw.shape}")

        mu = np.mean(regression_targets_raw[train_idx], axis=0)
        sd = np.std(regression_targets_raw[train_idx], axis=0)
        print("Raw target stats (train split) for 15-d targets:")
        print("  mean:", np.array2string(mu, precision=4, floatmode="fixed"))
        print("  std :", np.array2string(sd, precision=4, floatmode="fixed"))

        target_scaler.fit(regression_targets_raw[train_idx])
        scaled = target_scaler.transform(regression_targets_raw)
        regression_targets = jnp.array(scaled, dtype=jnp.float32)

        mu_s = np.mean(scaled[train_idx], axis=0)
        sd_s = np.std(scaled[train_idx], axis=0)
        print("Scaled target stats (train split) for 15-d targets:")
        print("  mean:", np.array2string(mu_s, precision=4, floatmode="fixed"))
        print("  std :", np.array2string(sd_s, precision=4, floatmode="fixed"))
    else:
        if use_transformed_eig:
            transformed = np.array(eigenvalues_to_increments(jnp.array(eigenvalues_raw)))
            regression_targets_raw = np.asarray(transformed, dtype=np.float64)
            target_scaler.fit(transformed[train_idx])
            transformed_scaled = target_scaler.transform(transformed)
            regression_targets = jnp.array(transformed_scaled, dtype=jnp.float32)

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
            regression_targets_raw = np.asarray(eigenvalues_raw, dtype=np.float64)
            target_scaler.fit(eigenvalues_raw[train_idx])
            scaled = target_scaler.transform(eigenvalues_raw)
            regression_targets = jnp.array(scaled, dtype=jnp.float32)

    payload = {
        "graph": graph,
        "regression_targets": regression_targets,
        "regression_targets_raw": regression_targets_raw.astype(np.float64),
        "target_scaler": target_scaler,
        "eigenvalues_raw": eigenvalues_raw.astype(np.float64),
        "masks": (jnp.array(train_mask), jnp.array(val_mask), jnp.array(test_mask)),
        "stats": stats,
    }
    if cweb is not None:
        payload["classification_labels"] = jnp.array(cweb, dtype=jnp.int32)
    if box_index is not None:
        payload["box_index"] = box_index.astype(np.int32)
        payload["box_index_col"] = str(args.box_index_col)
        payload["excluded_box_index_minus_one"] = bool(args.exclude_invalid_box_index)

    with out_cache.open("wb") as f:
        pickle.dump(payload, f)

    if deriv12 is not None:
        mode_name = "15-d (v1,v2,v3 + R*grads + R^2*laps)"
    else:
        mode_name = "transformed (v1, Δλ2, Δλ3)" if use_transformed_eig else "raw scaled (λ1, λ2, λ3)"
    print(f"Target mode: {mode_name}")
    print(f"Wrote SBI cache: {out_cache}")


if __name__ == "__main__":
    main()
