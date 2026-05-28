#!/usr/bin/env python3
"""Validate eigenvalues: annotated FITS vs direct T-Web slab lookup (unique halos).

This script compares two label paths for the **same unique host halos**:

Path A (FITS):
  - Read the annotated CutSky FITS (contains LAMBDA1/2/3).
  - Filter rows using (IN_Y1|IN_Y5) & (R_MAG_APP < 19.5) & (BOX_INDEX != -1).
  - Map each (FILE_NUM, BOX_INDEX) halo key to the corresponding eigenvalues.

Path B (slabs):
  - Take the same unique halo positions (box-frame x_com) from points.npy.
  - Sample (LAMBDA1/2/3) directly from slabwise T-Web grids
    (abacus_cactus_tweb_rank*.npz) at those coordinates.

It then reports Pearson r, MAE, and RMSE between Path A and Path B eigenvalues
across halos where both are available.

Why this helps:
  - If FITS labels match slab sampling for the same halos, then the annotation
    / mapping pipeline is consistent, and weak feature↔label correlations are
    unlikely to be caused by a label-coordinate bug.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.abacus_cutsky_selection import R_MAG_APP_BRIGHT_LT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cutsky-fits", type=Path, required=True, help="Annotated FITS with LAMBDA1/2/3.")
    p.add_argument(
        "--keys-npy",
        type=Path,
        required=True,
        help="keys.npy [N,2] = FILE_NUM, BOX_INDEX in node order (from export_cutsky_unique_host_halo_points.py).",
    )
    p.add_argument(
        "--points-npy",
        type=Path,
        required=True,
        help="points.npy [N,3] (box-frame halo x_com) in the same node order as keys.npy.",
    )
    p.add_argument(
        "--tweb-dir",
        type=Path,
        required=True,
        help="Directory containing abacus_cactus_tweb_rank*.npz slab files.",
    )
    p.add_argument("--chunk-size", type=int, default=2_000_000, help="FITS rows per chunk.")
    p.add_argument(
        "--max-halos",
        type=int,
        default=None,
        help="Optional cap on number of halos used (random subsample after alignment).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --max-halos subsample.")
    p.add_argument("--out-json", type=Path, default=None, help="Optional path to write summary JSON.")
    return p.parse_args()


def _resolve_col(names: list[str], candidates: tuple[str, ...]) -> str:
    up = {n.upper(): n for n in names}
    for c in candidates:
        k = c.upper()
        if k in up:
            return up[k]
    raise KeyError(f"None of {candidates} in columns (sample: {names[:30]})")


def _build_sorted_key_index(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_pairs, sorted_node_ids) where sorted_pairs is structured array."""
    file_num = keys[:, 0].astype(np.int32, copy=False)
    box_idx = keys[:, 1].astype(np.int64, copy=False)
    pairs = np.empty(keys.shape[0], dtype=[("file_num", np.int32), ("box_index", np.int64)])
    pairs["file_num"] = file_num
    pairs["box_index"] = box_idx
    order = np.argsort(pairs, kind="mergesort")
    return pairs[order], order.astype(np.int64)


def _search_pairs(sorted_pairs: np.ndarray, query_pairs: np.ndarray) -> np.ndarray:
    """Return indices into sorted_pairs for each query (or -1 if not found)."""
    idx = np.searchsorted(sorted_pairs, query_pairs)
    out = idx.astype(np.int64, copy=False)
    bad = (out < 0) | (out >= sorted_pairs.shape[0])
    out[bad] = -1
    ok = ~bad
    if np.any(ok):
        match = (sorted_pairs[out[ok]] == query_pairs[ok])
        tmp = out[ok]
        tmp[~match] = -1
        out[ok] = tmp
    return out


def load_fits_eigs_aligned_to_nodes(
    *,
    fits_path: Path,
    keys: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, dict]:
    """Return y_fits [N,3] with NaNs where missing, plus stats."""
    import fitsio

    sorted_pairs, order = _build_sorted_key_index(keys)
    n_nodes = keys.shape[0]
    y = np.full((n_nodes, 3), np.nan, dtype=np.float32)

    f = fitsio.FITS(str(fits_path), "r")
    hdu = f[1]
    nrows = hdu.get_nrows()
    colnames = hdu.get_colnames()

    c_file = _resolve_col(colnames, ("FILE_NUM",))
    c_box = _resolve_col(colnames, ("BOX_INDEX",))
    c_y1 = _resolve_col(colnames, ("IN_Y1",))
    c_y5 = _resolve_col(colnames, ("IN_Y5",))
    c_rmag = _resolve_col(colnames, ("R_MAG_APP",))
    c_l1 = _resolve_col(colnames, ("LAMBDA1", "L1", "EIG1", "LAM1", "LAMBDA_1"))
    c_l2 = _resolve_col(colnames, ("LAMBDA2", "L2", "EIG2", "LAM2", "LAMBDA_2"))
    c_l3 = _resolve_col(colnames, ("LAMBDA3", "L3", "EIG3", "LAM3", "LAMBDA_3"))

    col_list = [c_file, c_box, c_y1, c_y5, c_rmag, c_l1, c_l2, c_l3]

    filtered_rows = 0
    matched_rows = 0
    duplicate_rows = 0
    mismatch_rows = 0
    max_abs_diff = 0.0

    for start in range(0, nrows, chunk_size):
        stop = min(start + chunk_size, nrows)
        chunk = hdu[start:stop][col_list]
        fn = np.asarray(chunk[c_file], dtype=np.int32)
        bi = np.asarray(chunk[c_box], dtype=np.int64)
        y1 = np.asarray(chunk[c_y1]) == 1
        y5 = np.asarray(chunk[c_y5]) == 1
        rmag = np.asarray(chunk[c_rmag], dtype=np.float64)
        m = (y1 | y5) & (rmag < float(R_MAG_APP_BRIGHT_LT)) & (bi != -1)
        if not np.any(m):
            continue
        filtered_rows += int(np.sum(m))

        fn = fn[m]
        bi = bi[m]
        l1 = np.asarray(chunk[c_l1], dtype=np.float32)[m]
        l2 = np.asarray(chunk[c_l2], dtype=np.float32)[m]
        l3 = np.asarray(chunk[c_l3], dtype=np.float32)[m]
        eig = np.stack([l1, l2, l3], axis=1)

        qp = np.empty(fn.shape[0], dtype=sorted_pairs.dtype)
        qp["file_num"] = fn
        qp["box_index"] = bi
        pos_in_sorted = _search_pairs(sorted_pairs, qp)
        hit = pos_in_sorted >= 0
        if not np.any(hit):
            continue

        matched_rows += int(np.sum(hit))
        node_ids = order[pos_in_sorted[hit]]
        eig_hit = eig[hit]

        existing = y[node_ids]
        empty = ~np.isfinite(existing).all(axis=1)
        if np.any(empty):
            y[node_ids[empty]] = eig_hit[empty]

        filled = ~empty
        if np.any(filled):
            duplicate_rows += int(np.sum(filled))
            diff = np.abs(existing[filled] - eig_hit[filled])
            dmax = float(np.max(diff)) if diff.size else 0.0
            if dmax > max_abs_diff:
                max_abs_diff = dmax
            mismatch = np.any(diff > 1e-5, axis=1)
            mismatch_rows += int(np.sum(mismatch))

    f.close()

    stats = {
        "fits_total_rows": int(nrows),
        "filtered_galaxy_rows": int(filtered_rows),
        "matched_filtered_rows_to_unique_halos": int(matched_rows),
        "duplicate_filtered_rows_same_halo": int(duplicate_rows),
        "mismatch_rows_gt_1e-5": int(mismatch_rows),
        "max_abs_diff_across_duplicate_rows": float(max_abs_diff),
        "unique_halos_with_fits_eigs": int(np.isfinite(y).all(axis=1).sum()),
        "n_unique_halos": int(n_nodes),
    }
    return y, stats


@dataclass(frozen=True)
class SlabMeta:
    slab_id: int
    path: Path
    x_start: int
    x_end: int
    ngrid: int
    boxsize: float


def discover_slabs(tweb_dir: Path) -> list[SlabMeta]:
    files = sorted(tweb_dir.glob("abacus_cactus_tweb_rank*.npz"))
    if not files:
        raise FileNotFoundError(f"No slab files found in {tweb_dir} (abacus_cactus_tweb_rank*.npz)")

    slabs: list[SlabMeta] = []
    for path in files:
        with np.load(path) as d:
            slabs.append(
                SlabMeta(
                    slab_id=-1,
                    path=path,
                    x_start=int(d["x_start"]),
                    x_end=int(d["x_end"]),
                    ngrid=int(d["ngrid"]),
                    boxsize=float(d["boxsize"]),
                )
            )

    slabs = sorted(slabs, key=lambda s: s.x_start)
    # Renumber slab_id by x_start ordering
    out: list[SlabMeta] = []
    for i, s in enumerate(slabs):
        out.append(
            SlabMeta(
                slab_id=i,
                path=s.path,
                x_start=s.x_start,
                x_end=s.x_end,
                ngrid=s.ngrid,
                boxsize=s.boxsize,
            )
        )
    # Basic consistency checks
    ngrid_set = {s.ngrid for s in out}
    box_set = {s.boxsize for s in out}
    if len(ngrid_set) != 1 or len(box_set) != 1:
        raise ValueError("Inconsistent ngrid/boxsize across T-Web slab files.")
    return out


def build_slab_maps(slabs: list[SlabMeta]) -> tuple[np.ndarray, np.ndarray, int, float]:
    ngrid = slabs[0].ngrid
    boxsize = slabs[0].boxsize
    ix_to_slab = np.full(ngrid, -1, dtype=np.int16)
    slab_xstart = np.full(len(slabs), -1, dtype=np.int32)
    expected = 0
    for s in slabs:
        if s.x_start != expected:
            raise ValueError(f"Slab coverage gap/overlap near x={expected}, got x_start={s.x_start}")
        ix_to_slab[s.x_start : s.x_end] = s.slab_id
        slab_xstart[s.slab_id] = s.x_start
        expected = s.x_end
    if expected != ngrid or np.any(ix_to_slab < 0):
        raise ValueError("Slab coverage does not fully cover ix.")
    return ix_to_slab, slab_xstart, ngrid, boxsize


def positions_to_indices(xyz: np.ndarray, *, ngrid: int, boxsize: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = boxsize / float(ngrid)
    ix = np.floor(xyz[:, 0] / cell).astype(np.int32)
    iy = np.floor(xyz[:, 1] / cell).astype(np.int32)
    iz = np.floor(xyz[:, 2] / cell).astype(np.int32)
    np.clip(ix, 0, ngrid - 1, out=ix)
    np.clip(iy, 0, ngrid - 1, out=iy)
    np.clip(iz, 0, ngrid - 1, out=iz)
    return ix, iy, iz


def assign_eigs_from_slabs(
    xyz: np.ndarray,
    *,
    slabs: list[SlabMeta],
    ix_to_slab: np.ndarray,
    slab_xstart: np.ndarray,
    ngrid: int,
    boxsize: float,
) -> np.ndarray:
    """Return eigs [N,3] by loading each slab once and indexing."""
    ix, iy, iz = positions_to_indices(xyz, ngrid=ngrid, boxsize=boxsize)
    slab_ids = ix_to_slab[ix]
    local_ix = ix - slab_xstart[slab_ids]

    out = np.empty((xyz.shape[0], 3), dtype=np.float32)
    for slab in slabs:
        m = slab_ids == slab.slab_id
        if not np.any(m):
            continue
        rows = np.nonzero(m)[0]
        li = local_ix[rows].astype(np.int64)
        yj = iy[rows].astype(np.int64)
        zk = iz[rows].astype(np.int64)
        with np.load(slab.path) as d:
            eig = d["eig_vals"]  # [3, nx_local, ngrid, ngrid]
            out[rows, 0] = eig[0, li, yj, zk]
            out[rows, 1] = eig[1, li, yj, zk]
            out[rows, 2] = eig[2, li, yj, zk]
    return out


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def main() -> None:
    args = parse_args()
    keys = np.load(args.keys_npy)
    pts = np.load(args.points_npy)
    if keys.ndim != 2 or keys.shape[1] != 2:
        raise ValueError(f"Expected keys shape (N,2), got {keys.shape}")
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected points shape (N,>=3), got {pts.shape}")
    if keys.shape[0] != pts.shape[0]:
        raise ValueError(f"keys rows {keys.shape[0]:,} != points rows {pts.shape[0]:,}")

    xyz = np.ascontiguousarray(pts[:, :3], dtype=np.float64)

    print(f"Loading FITS eigenvalues and aligning to nodes: {args.cutsky_fits}")
    y_fits, stats_fits = load_fits_eigs_aligned_to_nodes(
        fits_path=args.cutsky_fits,
        keys=keys,
        chunk_size=args.chunk_size,
    )
    for k, v in stats_fits.items():
        print(f"  {k}: {v}")

    print(f"Discovering slabs: {args.tweb_dir}")
    slabs = discover_slabs(args.tweb_dir)
    ix_to_slab, slab_xstart, ngrid, boxsize = build_slab_maps(slabs)
    print(f"  ngrid={ngrid}, boxsize={boxsize}, n_slabs={len(slabs)}")

    print("Sampling eigenvalues from slabs at halo positions (can be slow)...")
    y_slabs = assign_eigs_from_slabs(
        xyz,
        slabs=slabs,
        ix_to_slab=ix_to_slab,
        slab_xstart=slab_xstart,
        ngrid=ngrid,
        boxsize=boxsize,
    )

    ok = np.isfinite(y_fits).all(axis=1) & np.isfinite(y_slabs).all(axis=1)
    n_ok = int(np.sum(ok))
    print(f"Halos with both FITS and slab eigs: {n_ok:,} / {keys.shape[0]:,}")

    yf = y_fits[ok].astype(np.float64)
    ys = y_slabs[ok].astype(np.float64)

    if args.max_halos is not None and n_ok > int(args.max_halos):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(n_ok, size=int(args.max_halos), replace=False)
        yf = yf[idx]
        ys = ys[idx]
        print(f"Subsampled halos for metrics: {yf.shape[0]:,}")

    pearson = [_pearsonr(yf[:, i], ys[:, i]) for i in range(3)]
    mae = np.mean(np.abs(yf - ys), axis=0)
    rmse = np.sqrt(np.mean((yf - ys) ** 2, axis=0))

    print("\nFITS vs slabs (same halos):")
    for i, name in enumerate(["LAMBDA1", "LAMBDA2", "LAMBDA3"]):
        print(
            f"  {name}: pearson_r={pearson[i]:.6f} | "
            f"MAE={mae[i]:.6e} | RMSE={rmse[i]:.6e}"
        )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fits_stats": stats_fits,
            "tweb": {
                "tweb_dir": str(args.tweb_dir.resolve()),
                "ngrid": int(ngrid),
                "boxsize": float(boxsize),
                "n_slabs": int(len(slabs)),
            },
            "alignment": {
                "n_halos_total": int(keys.shape[0]),
                "n_halos_both": int(n_ok),
                "subsample_used": int(yf.shape[0]),
            },
            "metrics": {
                "pearson_r": {"lambda1": pearson[0], "lambda2": pearson[1], "lambda3": pearson[2]},
                "mae": {"lambda1": float(mae[0]), "lambda2": float(mae[1]), "lambda3": float(mae[2])},
                "rmse": {"lambda1": float(rmse[0]), "lambda2": float(rmse[1]), "lambda3": float(rmse[2])},
            },
        }
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON: {args.out_json}")


if __name__ == "__main__":
    main()

