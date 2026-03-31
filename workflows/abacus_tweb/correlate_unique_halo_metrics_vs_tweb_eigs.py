#!/usr/bin/env python3
"""Pearson correlation between unique-host-halo graph metrics and T-Web eigenvalues.

Graph features from ``abacus_graph_features_cugraph.py`` are **one row per unique
host halo**, in the same order as the point cloud used to build the graph.

Eigenvalues are **not** read from the annotated FITS. They are sampled **directly**
from slabwise T-Web grids (``abacus_cactus_tweb_rank*.npz``) at each halo's
box-frame position, matching the assignment used when annotating mocks (grid
index from ``x_com`` / periodic box).

Inputs:
- Node feature parquet (cuGraph export).
- ``points.npy`` with the same halo order as the graph (typically
  ``*_points.npy`` written by ``build_abacus_graph.py`` for the same run).
- ``--tweb-dir`` containing ``abacus_cactus_tweb_rank*.npz`` files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_tweb_helpers():
    """Load slab helpers from validate_cutsky_eigs_boxindex_vs_halo_xcom (same grid logic)."""
    mod_path = Path(__file__).resolve().parent / "validate_cutsky_eigs_boxindex_vs_halo_xcom.py"
    name = "abacus_tweb_validate_slabs"
    spec = importlib.util.spec_from_file_location(name, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load slab helpers from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    # Must register before exec_module so dataclasses (and similar) resolve __module__ correctly.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.discover_slabs, mod.build_slab_maps, mod.assign_eigs_from_slabs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--node-parquet",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
            "host_halos_unique_alpha_cugraph_node_features.parquet"
        ),
        help="Parquet from abacus_graph_features_cugraph.py (one row per node).",
    )
    p.add_argument(
        "--points-npy",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
            "host_halos_unique_alpha_points.npy"
        ),
        help=(
            "Box-frame halo positions [N,3] or [N,>=3], same row order as the alpha graph / "
            "node parquet (from build_abacus_graph output or the original export points)."
        ),
    )
    p.add_argument(
        "--tweb-dir",
        type=Path,
        default=None,
        help=(
            "Directory with abacus_cactus_tweb_rank*.npz. "
            "Default: TNG_ABACUS_TWEB_OUTPUT_DIR or shared.config_paths.ABACUS_TWEB_OUTPUT_DIR."
        ),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to save Pearson matrix CSV.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to save summary JSON.",
    )
    return p.parse_args()


def _default_tweb_dir() -> Path:
    try:
        from shared.config_paths import ABACUS_TWEB_OUTPUT_DIR

        return Path(ABACUS_TWEB_OUTPUT_DIR)
    except Exception:
        return Path("/pscratch/sd/d/dkololgi/AbacusSummit_densities/tweb_rank_outputs")


def _pearson_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x: (n, p), y: (n, q) -> (p, q) Pearson r."""
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    num = xc.T @ yc
    xnorm = np.sqrt(np.sum(xc * xc, axis=0, keepdims=True)).T
    ynorm = np.sqrt(np.sum(yc * yc, axis=0, keepdims=True))
    return num / np.maximum(xnorm * ynorm, 1e-12)


def main() -> None:
    args = parse_args()
    discover_slabs, build_slab_maps, assign_eigs_from_slabs = _load_tweb_helpers()

    tweb_dir = args.tweb_dir if args.tweb_dir is not None else _default_tweb_dir()
    if not tweb_dir.is_dir():
        raise FileNotFoundError(f"T-Web directory not found: {tweb_dir}")

    feature_cols = [
        "Degree",
        "Clustering",
        "Density",
        "Neigh Density",
        "I_eig1",
        "I_eig2",
        "I_eig3",
    ]

    print(f"Loading node features: {args.node_parquet}")
    node_df = pd.read_parquet(args.node_parquet, columns=feature_cols)

    print(f"Loading halo positions: {args.points_npy}")
    pts = np.load(args.points_npy)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected points shape (N, >=3), got {pts.shape}")
    xyz = np.ascontiguousarray(pts[:, :3], dtype=np.float64)

    n = len(node_df)
    if xyz.shape[0] != n:
        raise ValueError(
            f"Row count mismatch: node features {n:,} vs points {xyz.shape[0]:,}. "
            "Use the points file from the same build_abacus_graph run as the graph."
        )

    print(f"Discovering T-Web slabs in: {tweb_dir}")
    slabs = discover_slabs(tweb_dir)
    ix_to_slab, slab_xstart, ngrid, boxsize = build_slab_maps(slabs)
    print(f"  ngrid={ngrid}, boxsize={boxsize}, n_slabs={len(slabs)}")

    print("Assigning eigenvalues from slab grids (this may take a few minutes)...")
    y = assign_eigs_from_slabs(
        xyz,
        slabs=slabs,
        ix_to_slab=ix_to_slab,
        slab_xstart=slab_xstart,
        ngrid=ngrid,
        boxsize=boxsize,
    ).astype(np.float64)

    x = node_df.to_numpy(dtype=np.float64)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if not ok.all():
        n_bad = int((~ok).sum())
        print(f"Dropping {n_bad:,} rows with non-finite features or eigenvalues.")
    x = x[ok]
    y = y[ok]
    print(f"Halos used for Pearson: {x.shape[0]:,} / {n:,}")

    pearson = _pearson_matrix(x, y)
    targ_idx = ["lambda1", "lambda2", "lambda3"]
    df = pd.DataFrame(pearson, index=feature_cols, columns=targ_idx)
    print("\nPearson r (T-Web slabs sampled at halo x_com, same order as graph nodes):")
    print(df.to_string(float_format=lambda v: f"{v:8.5f}"))

    summary = {
        "tweb_dir": str(tweb_dir.resolve()),
        "points_npy": str(args.points_npy.resolve()),
        "node_parquet": str(args.node_parquet.resolve()),
        "ngrid": int(ngrid),
        "boxsize": float(boxsize),
        "n_slabs": int(len(slabs)),
        "n_halos_in_graph": int(n),
        "n_halos_correlated": int(x.shape[0]),
    }

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv)
        print(f"\nSaved CSV: {args.out_csv}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pearson": df.to_dict(), "stats": summary}
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"Saved JSON: {args.out_json}")


if __name__ == "__main__":
    main()
