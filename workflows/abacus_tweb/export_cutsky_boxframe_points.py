#!/usr/bin/env python3
"""Export comoving box-frame halo positions for CutSky mocks (Gudhi graph input).

Rows match the same FITS read order and optional IN_Y1|IN_Y5 filter used by
`build_abacus_graph.py` when loading from catalog (so node index i aligns with
catalog row i after masking).

Outputs a float64 array of shape (N, 3) suitable for::

    build_abacus_graph.py --catalog-path '' --points-path <this.npy> --no-split-hemispheres

Requires abacusnbody (same as annotate_cutsky_with_tweb_eigs.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import fitsio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config_paths import ABACUS_BASE, CUTSKY_Z0200_PATH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=str, default=CUTSKY_Z0200_PATH)
    p.add_argument(
        "--halo-info-dir",
        type=str,
        default=f"{ABACUS_BASE}/halos/z0.200/halo_info",
        help="Directory with halo_info_XXX.asdf",
    )
    p.add_argument(
        "--halo-pos-field",
        choices=("x_com", "x_L2com"),
        default="x_com",
    )
    p.add_argument(
        "--apply-y1y5-filter",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-apply-y1y5-filter",
        dest="apply_y1y5_filter",
        action="store_false",
    )
    p.add_argument("--out-npy", type=str, required=True, help="Output path for (N,3) float64 .npy")
    p.add_argument(
        "--meta-json",
        type=str,
        default=None,
        help="Optional path to write small JSON sidecar (row counts, paths).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any galaxy cannot be linked to a halo position.",
    )
    return p.parse_args()


def _y1y5_mask(table: np.ndarray) -> np.ndarray:
    names_upper = {n.upper(): n for n in table.dtype.names}
    in_y1 = names_upper.get("IN_Y1")
    in_y5 = names_upper.get("IN_Y5")
    if in_y1 is None and in_y5 is None:
        return np.ones(len(table), dtype=bool)
    mask = np.zeros(len(table), dtype=bool)
    if in_y1 is not None:
        mask |= table[in_y1] == 1
    if in_y5 is not None:
        mask |= table[in_y5] == 1
    return mask


def _col(names: dict[str, str], *candidates: str) -> str:
    for c in candidates:
        if c.upper() in names:
            return names[c.upper()]
    raise KeyError(f"None of {candidates} in table; sample: {list(names)[:20]}")


def main() -> None:
    args = parse_args()
    t0 = time.time()
    try:
        from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    except Exception as e:
        raise RuntimeError("Install abacusnbody (abacusutils) in the active environment.") from e

    cat_path = Path(args.catalog).expanduser().resolve()
    if not cat_path.exists():
        raise FileNotFoundError(cat_path)

    print(f"Reading {cat_path} …")
    table = fitsio.read(str(cat_path))
    names_upper = {n.upper(): n for n in table.dtype.names}
    fn_col = _col(names_upper, "FILE_NUM")
    hi_col = _col(names_upper, "HALO_INDEX")

    mask = _y1y5_mask(table)
    if not args.apply_y1y5_filter:
        mask = np.ones(len(table), dtype=bool)
    idx_all = np.nonzero(mask)[0]
    file_num = np.asarray(table[fn_col][idx_all], dtype=np.int32)
    halo_index = np.asarray(table[hi_col][idx_all], dtype=np.int64)

    n = len(idx_all)
    xyz = np.full((n, 3), np.nan, dtype=np.float64)
    halo_dir = Path(args.halo_info_dir)

    for fn in np.unique(file_num):
        sel = file_num == fn
        if not np.any(sel):
            continue
        hp = halo_dir / f"halo_info_{int(fn):03d}.asdf"
        if not hp.exists():
            raise FileNotFoundError(f"Missing halo_info file: {hp}")

        try:
            cat = CompaSOHaloCatalog(
                str(hp),
                fields=[args.halo_pos_field],
                subsamples=False,
                convert_units=True,
                verbose=False,
            )
        except (TypeError, ValueError):
            try:
                cat = CompaSOHaloCatalog(
                    str(hp),
                    fields=[args.halo_pos_field],
                    cleaned=True,
                    convert_units=True,
                    verbose=False,
                )
            except (TypeError, ValueError):
                cat = CompaSOHaloCatalog(str(hp), cleaned=True)

        arr = np.asarray(cat.halos[args.halo_pos_field], dtype=np.float64)
        nh = arr.shape[0]
        hidx = halo_index[sel]
        ok = (hidx >= 0) & (hidx < nh)
        rows = np.where(sel)[0]
        if np.any(ok):
            xyz[rows[ok]] = arr[hidx[ok]]
        bad_local = ~ok
        if np.any(bad_local) and args.strict:
            raise RuntimeError(
                f"strict: file_num={fn} has {bad_local.sum()} rows with invalid HALO_INDEX "
                f"(nhalo={nh})."
            )

    n_bad = int(np.sum(~np.isfinite(xyz).all(axis=1)))
    if n_bad and args.strict:
        raise RuntimeError(f"strict: {n_bad} rows have no valid xyz after linkage.")
    if n_bad:
        print(f"WARNING: {n_bad} / {n:,} rows lack valid halo xyz; they remain NaN.")

    out_npy = Path(args.out_npy).expanduser().resolve()
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, xyz.astype(np.float64))
    print(f"Saved {out_npy} shape={xyz.shape} dtype=float64 ({time.time()-t0:.1f}s)")

    if args.meta_json:
        meta = {
            "catalog": str(cat_path),
            "halo_info_dir": str(halo_dir),
            "halo_pos_field": args.halo_pos_field,
            "apply_y1y5_filter": bool(args.apply_y1y5_filter),
            "n_rows_output": int(n),
            "n_nan_rows": n_bad,
            "fits_index_min": int(idx_all.min()) if n else None,
            "fits_index_max": int(idx_all.max()) if n else None,
            "out_npy": str(out_npy),
        }
        mp = Path(args.meta_json).expanduser().resolve()
        mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta {mp}")


if __name__ == "__main__":
    main()
