#!/usr/bin/env python3
"""Export comoving Cartesian points for full stage-3 post-collision mock.

Reads ``datcomb_brightwdup.fits``, keeps ``COLLISION == 0`` and ``BOX_INDEX != -1``,
deduplicates on ``(FILE_NUM, HALO_INDEX, BOX_INDEX)`` (first row in FITS order wins),
and writes:

- ``<out-prefix>_points.npy``: (N, 4) float64 ``x, y, z, north_flag``
- ``<out-prefix>_points_xyz.npy``: (N, 3) float64 xyz only (optional convenience)

``north_flag`` is 1 when Galactic ``b > 0``, else 0 — same convention as
``build_abacus_graph._load_points_from_catalog`` for hemisphere-split Delaunay.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import astropy.units as u
import fitsio
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo

DEFAULT_STAGE3_FITS = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/fba0/"
    "datcomb_brightwdup.fits"
)
DEFAULT_OUT_DIR = "/pscratch/sd/d/dkololgi/abacus/graph_constructions"
DEFAULT_PREFIX = "staged_mock_stage3_postcollision_full_rs7"

TRIPLE_COLS = ("FILE_NUM", "HALO_INDEX", "BOX_INDEX")
POS_COLS = ("RA", "DEC", "Z")


def _resolve_col(colnames: list[str], name: str) -> str:
    m = {c.upper(): c for c in colnames}
    resolved = m.get(name.upper())
    if resolved is None:
        raise KeyError(f"Column {name!r} not in FITS (have {len(colnames)} cols).")
    return resolved


def sky_to_xyz_mpc(ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(cosmo.comoving_distance(z).value, dtype=np.float64)
    ra = np.deg2rad(ra_deg.astype(np.float64, copy=False))
    dec = np.deg2rad(dec_deg.astype(np.float64, copy=False))
    x = r * np.cos(dec) * np.cos(ra)
    y = r * np.cos(dec) * np.sin(ra)
    zc = r * np.sin(dec)
    return x, y, zc


def galactic_north_flag(ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Galactic b > 0 flag (int8), matching build_abacus_graph catalog mode."""
    comoving_distance = cosmo.comoving_distance(z)
    sky_icrs = SkyCoord(
        ra=ra_deg,
        dec=dec_deg,
        unit=(u.deg, u.deg),
        distance=comoving_distance,
        frame="icrs",
    )
    return (sky_icrs.galactic.b.deg > 0).astype(np.int8)


def _iter_fits_chunks(path: Path, columns: list[str], chunk_size: int):
    with fitsio.FITS(str(path)) as ff:
        hdu = ff[1]
        n = int(hdu.get_nrows())
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield start, hdu[columns][start:end]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage3-fits", type=Path, default=Path(DEFAULT_STAGE3_FITS))
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    p.add_argument("--out-prefix", type=str, default=DEFAULT_PREFIX)
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--redshift-col", default="Z")
    p.add_argument(
        "--skip-xyz",
        action="store_true",
        help="Do not write the xyz-only sidecar (_points_xyz.npy).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stage3 = args.stage3_fits.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with fitsio.FITS(str(stage3)) as ff:
        cols = [str(c) for c in ff[1].get_colnames()]
        n_total = int(ff[1].get_nrows())

    ra_c, dec_c = _resolve_col(cols, "RA"), _resolve_col(cols, "DEC")
    z_c = _resolve_col(cols, args.redshift_col)
    fn_c, hi_c, bi_c = (_resolve_col(cols, c) for c in TRIPLE_COLS)
    coll_c = _resolve_col(cols, "COLLISION")
    read_cols = list(dict.fromkeys([ra_c, dec_c, z_c, fn_c, hi_c, bi_c, coll_c]))

    ra_buf: list[np.ndarray] = []
    dec_buf: list[np.ndarray] = []
    z_buf: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()

    n_collision_pass = 0
    n_box_invalid_skip = 0
    n_dedup_skip = 0

    for _start, tab in _iter_fits_chunks(stage3, read_cols, args.chunk_size):
        m = tab[coll_c] == 0
        if not np.any(m):
            continue
        n_collision_pass += int(np.count_nonzero(m))

        m &= tab[bi_c] != -1
        if not np.any(m):
            n_box_invalid_skip += int(np.count_nonzero(tab[bi_c] == -1))
            continue
        n_box_invalid_skip += int(np.count_nonzero((tab[coll_c] == 0) & (tab[bi_c] == -1)))

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

    ra_all = np.concatenate(ra_buf) if ra_buf else np.empty(0, dtype=np.float64)
    dec_all = np.concatenate(dec_buf) if dec_buf else np.empty(0, dtype=np.float64)
    z_all = np.concatenate(z_buf) if z_buf else np.empty(0, dtype=np.float64)

    x, y, zc = sky_to_xyz_mpc(ra_all, dec_all, z_all)
    north_flag = galactic_north_flag(ra_all, dec_all, z_all)
    points = np.column_stack((x, y, zc, north_flag.astype(np.float64)))

    prefix = args.out_prefix
    points_path = out_dir / f"{prefix}_points.npy"
    points_xyz_path = out_dir / f"{prefix}_points_xyz.npy"
    meta_path = out_dir / f"{prefix}_points_export.json"
    np.save(points_path, points)
    if not args.skip_xyz:
        np.save(points_xyz_path, points[:, :3])

    meta = {
        "stage3_fits": str(stage3),
        "n_rows_total": n_total,
        "n_rows_collision_eq0": n_collision_pass,
        "n_rows_box_index_invalid_skipped": n_box_invalid_skip,
        "n_dedup_skip": n_dedup_skip,
        "n_points": int(points.shape[0]),
        "north_flag_sum": int(north_flag.sum()),
        "south_flag_sum": int((north_flag == 0).sum()),
        "redshift_col": args.redshift_col,
        "filters": {
            "COLLISION": "== 0",
            "BOX_INDEX": "!= -1",
            "dedupe_halo_triple": True,
        },
        "cosmology": "Planck18",
        "coord_convention": "sky_to_xyz_mpc + galactic b>0 north_flag (build_abacus_graph)",
        "points_path": str(points_path),
        "points_xyz_path": str(points_xyz_path) if not args.skip_xyz else None,
        "points_shape": list(points.shape),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {points.shape[0]:,} points (N,4) -> {points_path}")
    if not args.skip_xyz:
        print(f"Wrote xyz sidecar -> {points_xyz_path}")
    print(f"Metadata -> {meta_path}")
    print(
        f"Rows: total={n_total:,} collision==0 passes={n_collision_pass:,} "
        f"box_invalid_skip={n_box_invalid_skip:,} dedup_skip={n_dedup_skip:,} "
        f"unique={points.shape[0]:,} "
        f"north={north_flag.sum():,} south={(north_flag == 0).sum():,}"
    )


if __name__ == "__main__":
    main()
