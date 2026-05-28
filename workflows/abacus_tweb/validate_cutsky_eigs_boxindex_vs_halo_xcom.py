#!/usr/bin/env python3
"""Validate CutSky T-Web eigenvalue assignment: BOX_INDEX vs halo x_com lookup.

Given a CutSky mock FITS that already contains (CWEB, LAMBDA1/2/3) assigned via
BOX_INDEX, this script:

1) Samples rows after applying typical survey filters:
   - (IN_Y1 == 1) | (IN_Y5 == 1)
   - R_MAG_APP < 19.5 (DESI BGS bright)
   - BOX_INDEX != -1
2) Uses (FILE_NUM, HALO_INDEX) to fetch the host halo's box-frame x_com from
   Abacus CompaSO halo_info files.
3) Assigns eigenvalues to those host-halo positions via slabwise T-Web outputs
   (abacus_cactus_tweb_rank*.npz).
4) Reports agreement statistics and Pearson correlations between the two
   eigenvalue assignments.

This isolates whether poor downstream correlations are driven by sky/lightcone
coordinate inversion/modulo or by later graph/feature steps.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.abacus_cutsky_selection import cutsky_desi_bgs_mock_mask


@dataclass(frozen=True)
class SlabMeta:
    slab_id: int
    path: Path
    x_start: int
    x_end: int
    ngrid: int
    boxsize: float


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


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
    ngrid_set = {s.ngrid for s in slabs}
    box_set = {s.boxsize for s in slabs}
    if len(ngrid_set) != 1 or len(box_set) != 1:
        raise ValueError("Inconsistent ngrid/boxsize across T-Web slab files.")

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


def sample_filtered_rows(
    *,
    fits_path: Path,
    sample_size: int,
    seed: int,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Reservoir-sample filtered rows without loading full FITS."""
    import fitsio

    rng = np.random.default_rng(seed)
    f = fitsio.FITS(str(fits_path), "r")
    hdu = f[1]
    nrows = hdu.get_nrows()

    cols = [
        "IN_Y1",
        "IN_Y5",
        "R_MAG_APP",
        "BOX_INDEX",
        "FILE_NUM",
        "HALO_INDEX",
        "LAMBDA1",
        "LAMBDA2",
        "LAMBDA3",
    ]
    have = set(hdu.get_colnames())
    missing = [c for c in cols if c not in have]
    if missing:
        raise KeyError(f"Missing required columns in {fits_path}: {missing}")

    keep = {c: [] for c in cols}
    seen = 0

    for start in range(0, nrows, chunk_size):
        stop = min(start + chunk_size, nrows)
        chunk = hdu[start:stop][cols]
        mask = cutsky_desi_bgs_mock_mask(chunk)
        box = np.asarray(chunk["BOX_INDEX"], dtype=np.int64)
        mask &= box != -1
        if not np.any(mask):
            continue
        idx = np.nonzero(mask)[0]
        # reservoir sampling on the filtered stream
        for j in idx:
            seen += 1
            if len(keep["BOX_INDEX"]) < sample_size:
                for c in cols:
                    keep[c].append(chunk[c][j])
            else:
                r = int(rng.integers(0, seen))
                if r < sample_size:
                    for c in cols:
                        keep[c][r] = chunk[c][j]

    f.close()
    out = {}
    for c in cols:
        out[c] = np.asarray(keep[c])
    return out


def load_halo_positions_xcom(
    *,
    halo_info_dir: Path,
    file_nums: np.ndarray,
    halo_indices: np.ndarray,
) -> np.ndarray:
    """Load x_com for each (FILE_NUM, HALO_INDEX) pair.

    Notes:
    - We load one halo_info_<FILE_NUM>.asdf per group and index into halos['x_com'].
    - Uses abacusnbody's CompaSOHaloCatalog.
    """
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    file_nums = np.asarray(file_nums, dtype=np.int32)
    halo_indices = np.asarray(halo_indices, dtype=np.int64)
    if file_nums.shape != halo_indices.shape:
        raise ValueError("file_nums and halo_indices must have same shape.")

    out = np.empty((file_nums.shape[0], 3), dtype=np.float32)
    order = np.argsort(file_nums, kind="mergesort")
    file_sorted = file_nums[order]
    idx_sorted = halo_indices[order]

    unique_files, starts = np.unique(file_sorted, return_index=True)
    starts = list(starts) + [len(file_sorted)]

    for uf, s0, s1 in zip(unique_files, starts[:-1], starts[1:]):
        fp = halo_info_dir / f"halo_info_{int(uf):03d}.asdf"
        if not fp.exists():
            raise FileNotFoundError(f"Missing halo_info file: {fp}")
        # Request only the field we need when supported.
        try:
            cat = CompaSOHaloCatalog(str(fp), cleaned=False, fields=["x_com"])
        except TypeError:
            cat = CompaSOHaloCatalog(str(fp), cleaned=False)
        halos = cat.halos
        xs = halos["x_com"][idx_sorted[s0:s1]].astype(np.float32, copy=False)
        out[order[s0:s1]] = xs
        # help GC
        del halos, cat, xs

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cutsky-with-eigs",
        required=True,
        help="Input CutSky FITS with BOX_INDEX-assigned CWEB/LAMBDA* columns.",
    )
    p.add_argument(
        "--tweb-dir",
        required=True,
        help="Directory containing abacus_cactus_tweb_rank*.npz files (slab outputs).",
    )
    p.add_argument(
        "--halo-info-dir",
        required=True,
        help="Directory containing halo_info_*.asdf (CompaSO) files.",
    )
    p.add_argument("--sample-size", type=int, default=500_000, help="Number of filtered rows to sample.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=1_000_000, help="Rows per FITS chunk when sampling.")
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for reporting exact-equality rate between eigenvalues.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cutsky = Path(args.cutsky_with_eigs).expanduser().resolve()
    tweb_dir = Path(args.tweb_dir).expanduser().resolve()
    halo_dir = Path(args.halo_info_dir).expanduser().resolve()

    print(f"Sampling rows from: {cutsky}")
    s = sample_filtered_rows(
        fits_path=cutsky,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
    )
    n = s["BOX_INDEX"].shape[0]
    if n == 0:
        raise RuntimeError("No rows sampled (filters too strict or empty input).")
    print(f"Sampled {n:,} filtered rows.")

    eig_box = np.stack([s["LAMBDA1"], s["LAMBDA2"], s["LAMBDA3"]], axis=-1).astype(np.float32)
    file_num = s["FILE_NUM"].astype(np.int32)
    halo_idx = s["HALO_INDEX"].astype(np.int64)

    print("Loading host halo x_com positions via (FILE_NUM, HALO_INDEX)...")
    xyz = load_halo_positions_xcom(halo_info_dir=halo_dir, file_nums=file_num, halo_indices=halo_idx)
    if not np.all(np.isfinite(xyz)):
        raise RuntimeError("Non-finite values found in loaded x_com positions.")

    print("Loading T-Web slabs and assigning eigenvalues at x_com...")
    slabs = discover_slabs(tweb_dir)
    ix_to_slab, slab_xstart, ngrid, boxsize = build_slab_maps(slabs)
    eig_xcom = assign_eigs_from_slabs(
        xyz,
        slabs=slabs,
        ix_to_slab=ix_to_slab,
        slab_xstart=slab_xstart,
        ngrid=ngrid,
        boxsize=boxsize,
    )

    # Stats
    diff = (eig_box - eig_xcom).astype(np.float64)
    mae = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff * diff, axis=0))
    eq = np.mean(np.abs(diff) <= float(args.tolerance), axis=0)

    print("\n=== BOX_INDEX-eigs vs x_com-eigs (same sampled rows) ===")
    for i, name in enumerate(["LAMBDA1", "LAMBDA2", "LAMBDA3"]):
        r = _pearsonr(eig_box[:, i], eig_xcom[:, i])
        print(
            f"{name}: pearson_r={r:.6f} | MAE={mae[i]:.6e} | RMSE={rmse[i]:.6e} | "
            f"frac(|Δ|<=tol)={eq[i]:.6f}"
        )

    # Optional: quick sanity on x range
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    print(f"\nx_com range (Mpc/h): x=[{mins[0]:.3f},{maxs[0]:.3f}] y=[{mins[1]:.3f},{maxs[1]:.3f}] z=[{mins[2]:.3f},{maxs[2]:.3f}]")
    print(f"T-Web grid: ngrid={ngrid}, boxsize={boxsize}")


if __name__ == "__main__":
    main()

