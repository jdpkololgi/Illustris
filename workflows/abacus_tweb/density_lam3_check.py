#!/usr/bin/env python3
"""Batched neighbor counts (10 Mpc/h) vs slab λ3 for unique halo positions."""
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--tweb-dir", type=Path, required=True)
    p.add_argument("--box", type=float, default=2000.0)
    p.add_argument("--r", type=float, default=10.0)
    p.add_argument("--batch", type=int, default=200_000)
    return p.parse_args()

def _wrap_periodic_box(coords: np.ndarray, boxsize: float) -> np.ndarray:
    """Map to [0, boxsize); scipy periodic cKDTree requires coords < boxsize (strict)."""
    c = np.mod(coords.astype(np.float64), boxsize)
    # Edge cases: mod can yield boxsize from FP noise; clip just below boxsize.
    eps = max(1e-9, float(np.finfo(np.float64).eps * boxsize * 1e3))
    np.clip(c, 0.0, boxsize - eps, out=c)
    return c


def main():
    args = parse_args()

    slab_files = sorted(args.tweb_dir.glob("abacus_cactus_tweb_rank*.npz"))
    if not slab_files:
        raise SystemExit(f"No slabs in {args.tweb_dir}")

    with np.load(slab_files[0]) as d0:
        ngrid = int(d0["ngrid"])
        boxsize = float(d0["boxsize"])
    # Use slab boxsize for wrapping + KD-tree (must match).
    args.box = boxsize

    coords = np.load(args.points)[:, :3].astype(np.float64)
    coords = _wrap_periodic_box(coords, args.box).astype(np.float32)
    # Ensure strict [0, boxsize) after float32 (avoid rounding up to boxsize).
    boxf = np.float32(args.box)
    np.clip(coords, 0.0, np.nextafter(boxf, np.float32(0.0)), out=coords)

    ix_to_slab = np.full(ngrid, -1, dtype=np.int16)
    slab_xstart = np.full(len(slab_files), -1, dtype=np.int32)
    slabs = []
    for sid, fp in enumerate(slab_files):
        with np.load(fp) as d:
            xs, xe = int(d["x_start"]), int(d["x_end"])
        slabs.append((sid, fp, xs, xe))
        ix_to_slab[xs:xe] = sid
        slab_xstart[sid] = xs

    cell = args.box / float(ngrid)
    ix = np.floor(coords[:, 0] / cell).astype(np.int32)
    iy = np.floor(coords[:, 1] / cell).astype(np.int32)
    iz = np.floor(coords[:, 2] / cell).astype(np.int32)
    np.clip(ix, 0, ngrid - 1, out=ix)
    np.clip(iy, 0, ngrid - 1, out=iy)
    np.clip(iz, 0, ngrid - 1, out=iz)
    slab_ids = ix_to_slab[ix]
    local_ix = ix - slab_xstart[slab_ids]

    lam3 = np.empty(coords.shape[0], dtype=np.float32)
    for sid, fp, *_ in slabs:
        m = slab_ids == sid
        if not np.any(m):
            continue
        rows = np.nonzero(m)[0]
        li = local_ix[rows].astype(np.int64)
        yj = iy[rows].astype(np.int64)
        zk = iz[rows].astype(np.int64)
        with np.load(fp) as d:
            ev = d["eig_vals"]
            lam3[rows] = ev[0, li, yj, zk]

    print("Building KD-tree...")
    tree = cKDTree(coords, boxsize=args.box)

    n = coords.shape[0]
    counts = np.empty(n, dtype=np.int32)
    for i in range(0, n, args.batch):
        j = min(i + args.batch, n)
        counts[i:j] = tree.query_ball_point(coords[i:j], r=args.r, return_length=True).astype(np.int32) - 1
        if i == 0 or (i // args.batch) % 10 == 0:
            print(f"  counts {j:,}/{n:,}")

    c = counts.astype(np.float64)
    l = lam3.astype(np.float64)
    c -= c.mean()
    l -= l.mean()
    r = float((c @ l) / np.sqrt((c @ c) * (l @ l) + 1e-30))
    print(f"N={n:,}  R={args.r} Mpc/h  mean_count={counts.mean():.3f}  pearson_r(counts, lambda3)={r:.6f}")

if __name__ == "__main__":
    main()