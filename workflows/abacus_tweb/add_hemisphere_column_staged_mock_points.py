#!/usr/bin/env python3
"""Add Galactic north_flag (column 4) to an existing (N,3) points_xyz export.

Replays the same stage-3 FITS filters and dedupe order as
``export_staged_mock_stage3_full_points.py``, then checks ``n_points`` against
``<prefix>_points_export.json`` before appending the hemisphere column.

Use when a full FITS re-read for xyz is unnecessary but RA/Dec are needed only
for ``north_flag`` — still requires scanning the FITS once for sky coords.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from export_staged_mock_stage3_full_points import (
    DEFAULT_OUT_DIR,
    DEFAULT_PREFIX,
    DEFAULT_STAGE3_FITS,
    galactic_north_flag,
    _iter_fits_chunks,
    _resolve_col,
)


def _collect_ra_dec_z(stage3: Path, chunk_size: int, redshift_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    import fitsio

    with fitsio.FITS(str(stage3)) as ff:
        cols = [str(c) for c in ff[1].get_colnames()]
        n_total = int(ff[1].get_nrows())

    ra_c, dec_c = _resolve_col(cols, "RA"), _resolve_col(cols, "DEC")
    z_c = _resolve_col(cols, redshift_col)
    fn_c, hi_c, bi_c = (_resolve_col(cols, c) for c in ("FILE_NUM", "HALO_INDEX", "BOX_INDEX"))
    coll_c = _resolve_col(cols, "COLLISION")
    read_cols = list(dict.fromkeys([ra_c, dec_c, z_c, fn_c, hi_c, bi_c, coll_c]))

    ra_buf: list[np.ndarray] = []
    dec_buf: list[np.ndarray] = []
    z_buf: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    n_collision_pass = 0
    n_dedup_skip = 0

    for _start, tab in _iter_fits_chunks(stage3, read_cols, chunk_size):
        m = tab[coll_c] == 0
        if not np.any(m):
            continue
        n_collision_pass += int(np.count_nonzero(m))
        ra = tab[ra_c][m].astype(np.float64, copy=False)
        dec = tab[dec_c][m].astype(np.float64, copy=False)
        zz = tab[z_c][m].astype(np.float64, copy=False)
        fn = tab[fn_c][m].astype(np.int64, copy=False)
        hi = tab[hi_c][m].astype(np.int64, copy=False)
        bi = tab[bi_c][m].astype(np.int64, copy=False)

        keep = np.ones(fn.size, dtype=bool)
        for i in range(fn.size):
            key = (int(fn[i]), int(hi[i]), int(bi[i]))
            if key in seen:
                keep[i] = False
                n_dedup_skip += 1
                continue
            seen.add(key)
        if not np.any(keep):
            continue
        ra_buf.append(ra[keep])
        dec_buf.append(dec[keep])
        z_buf.append(zz[keep])

    stats = {
        "n_rows_total": n_total,
        "n_rows_collision_eq0": n_collision_pass,
        "n_dedup_skip": n_dedup_skip,
        "n_points": int(sum(a.size for a in ra_buf)),
    }
    ra_all = np.concatenate(ra_buf) if ra_buf else np.empty(0, dtype=np.float64)
    dec_all = np.concatenate(dec_buf) if dec_buf else np.empty(0, dtype=np.float64)
    z_all = np.concatenate(z_buf) if z_buf else np.empty(0, dtype=np.float64)
    return ra_all, dec_all, z_all, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points-xyz", type=Path, required=True, help="Existing (N,3) float64 .npy")
    p.add_argument("--meta-json", type=Path, default=None, help="Export metadata JSON for row-count check")
    p.add_argument("--out-points", type=Path, default=None, help="Output (N,4); default: xyz path with _xyz -> _points")
    p.add_argument("--stage3-fits", type=Path, default=Path(DEFAULT_STAGE3_FITS))
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--redshift-col", default="Z")
    p.add_argument("--out-prefix", type=str, default=DEFAULT_PREFIX)
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    xyz_path = args.points_xyz.expanduser().resolve()
    xyz = np.load(xyz_path)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected (N,3) at {xyz_path}, got {xyz.shape}")

    meta_path = args.meta_json
    if meta_path is None:
        meta_path = xyz_path.parent / f"{args.out_prefix}_points_export.json"
    meta_path = Path(meta_path).expanduser().resolve()
    expected_n = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_n = int(meta.get("n_points", meta.get("n_points", -1)))
        if expected_n < 0:
            expected_n = None

    ra, dec, z, stats = _collect_ra_dec_z(
        args.stage3_fits.expanduser().resolve(), args.chunk_size, args.redshift_col
    )
    if ra.shape[0] != xyz.shape[0]:
        raise ValueError(
            f"Row count mismatch: xyz N={xyz.shape[0]:,} vs FITS replay N={ra.shape[0]:,}"
        )
    if expected_n is not None and expected_n != xyz.shape[0]:
        raise ValueError(
            f"Metadata n_points={expected_n:,} != xyz N={xyz.shape[0]:,} ({meta_path})"
        )

    north_flag = galactic_north_flag(ra, dec, z)
    points = np.column_stack((xyz.astype(np.float64, copy=False), north_flag.astype(np.float64)))

    if args.out_points is not None:
        out_path = args.out_points.expanduser().resolve()
    else:
        stem = xyz_path.name.replace("_points_xyz.npy", "_points.npy")
        out_path = xyz_path.parent / stem
        if out_path == xyz_path:
            out_path = args.out_dir.expanduser().resolve() / f"{args.out_prefix}_points.npy"

    np.save(out_path, points)
    print(
        f"Wrote {points.shape[0]:,} points (N,4) -> {out_path} "
        f"(north={int(north_flag.sum()):,}, south={int((north_flag == 0).sum()):,})"
    )
    print(f"FITS replay stats: {stats}")
    if meta_path.is_file():
        print(f"Verified against metadata: {meta_path}")


if __name__ == "__main__":
    main()
