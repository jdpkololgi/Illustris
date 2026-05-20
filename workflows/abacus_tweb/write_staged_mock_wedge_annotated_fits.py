#!/usr/bin/env python3
"""Write stage-3 mock wedge FITS: datcomb columns + T-Web eigenvalues from truth NPZ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitsio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.build_staged_mock_wedge_truth_npz import (  # noqa: E402
    EIG_THR,
    WEDGE_DEFAULTS,
    _Stage3Mask,
    _iter_fits_chunks,
    _resolve_col,
    wedge_mask,
    write_wedge_targets_fits,
)


def _sky_lut(npz_path: Path) -> dict[tuple[float, float, float], tuple[float, float, float, int]]:
    d = np.load(npz_path)
    lut: dict[tuple[float, float, float], tuple[float, float, float, int]] = {}
    for i in range(int(d["ra"].size)):
        key = (float(d["ra"][i]), float(d["dec"][i]), float(d["z"][i]))
        lut[key] = (float(d["lambda1"][i]), float(d["lambda2"][i]), float(d["lambda3"][i]), int(d["cls"][i]))
    return lut


def extract_annotated_datcomb(
    datcomb: Path,
    lut: dict[tuple[float, float, float], tuple[float, float, float, int]],
    *,
    wedge: dict[str, float],
    z_col: str,
    chunk_size: int,
) -> np.ndarray:
    mask_fn = _Stage3Mask()
    with fitsio.FITS(str(datcomb)) as ff:
        cols = [str(c) for c in ff[1].get_colnames()]
        base_dtype = None  # set from first chunk
    ra_c, dec_c = _resolve_col(cols, "RA"), _resolve_col(cols, "DEC")
    z_c = _resolve_col(cols, z_col)
    fn_c, hi_c, bi_c = _resolve_col(cols, "FILE_NUM"), _resolve_col(cols, "HALO_INDEX"), _resolve_col(cols, "BOX_INDEX")
    read_cols = list(cols)
    seen: set[tuple[int, int, int]] = set()
    parts: list[np.ndarray] = []

    for _start, tab in _iter_fits_chunks(datcomb, read_cols, chunk_size):
        m = wedge_mask(tab[ra_c], tab[dec_c], tab[z_c], **wedge)
        m &= mask_fn(tab, cols)
        if not np.any(m):
            continue
        sub = tab[m]
        if base_dtype is None:
            base_dtype = sub.dtype
        keep = []
        fn = sub[fn_c].astype(np.int64, copy=False)
        hi = sub[hi_c].astype(np.int64, copy=False)
        bi = sub[bi_c].astype(np.int64, copy=False)
        for i in range(fn.size):
            tkey = (int(fn[i]), int(hi[i]), int(bi[i]))
            if tkey in seen:
                continue
            sky = (float(sub[ra_c][i]), float(sub[dec_c][i]), float(sub[z_c][i]))
            if sky not in lut:
                continue
            seen.add(tkey)
            keep.append(i)
        if keep:
            parts.append(sub[np.asarray(keep, dtype=np.int64)])

    if not parts:
        raise RuntimeError("No datcomb rows matched NPZ sky keys in wedge/stage3 subset.")
    out = np.concatenate(parts)
    new_dtype = base_dtype.descr + [
        ("LAMBDA1", "f4"),
        ("LAMBDA2", "f4"),
        ("LAMBDA3", "f4"),
        ("CWEB", "i1"),
    ]
    table = np.empty(out.size, dtype=new_dtype)
    for name in out.dtype.names:
        table[name] = out[name]
    for i in range(out.size):
        sky = (float(out[ra_c][i]), float(out[dec_c][i]), float(out[z_c][i]))
        l1, l2, l3, cw = lut[sky]
        table["LAMBDA1"][i] = l1
        table["LAMBDA2"][i] = l2
        table["LAMBDA3"][i] = l3
        table["CWEB"][i] = cw
    return table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--truth-npz",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/"
            "staged_mock_wedge_stage3_postcollision_rs7.npz"
        ),
    )
    p.add_argument(
        "--datcomb",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/fba0/datcomb_brightwdup.fits"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/"
            "staged_mock_wedge_stage3_postcollision_rs7_annotated.fits"
        ),
    )
    p.add_argument(
        "--wedge-targets",
        type=Path,
        default=None,
        help="Optional wedge_targets.fits (adds FILE_NUM/HALO_INDEX/BOX_INDEX).",
    )
    p.add_argument("--chunk-size", type=int, default=2_000_000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"Exists (use --overwrite): {args.output}")
        return 0
    wedge = dict(WEDGE_DEFAULTS)
    lut = _sky_lut(args.truth_npz)
    print(f"NPZ sky LUT size: {len(lut):,}", flush=True)
    table = extract_annotated_datcomb(
        args.datcomb,
        lut,
        wedge=wedge,
        z_col="Z",
        chunk_size=args.chunk_size,
    )
    if table.size != len(lut):
        print(
            f"WARNING: wrote n={table.size:,} but NPZ n={len(lut):,}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(args.output), table, clobber=True)
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "truth_npz": str(args.truth_npz),
                "datcomb": str(args.datcomb),
                "n_rows": int(table.size),
                "wedge": wedge,
                "eig_threshold": EIG_THR,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output} (n={table.size:,}), manifest {manifest}", flush=True)

    if args.wedge_targets is not None:
        names = list(table.dtype.names)
        arrays = {
            "ra": table[_resolve_col(names, "RA")].astype(np.float64),
            "dec": table[_resolve_col(names, "DEC")].astype(np.float64),
            "z": table[_resolve_col(names, "Z")].astype(np.float64),
            "lambda1": table["LAMBDA1"].astype(np.float32),
            "lambda2": table["LAMBDA2"].astype(np.float32),
            "lambda3": table["LAMBDA3"].astype(np.float32),
            "file_num": table["FILE_NUM"].astype(np.int32),
            "halo_index": table["HALO_INDEX"].astype(np.int32),
            "box_index": table["BOX_INDEX"].astype(np.int32),
        }
        write_wedge_targets_fits(args.wedge_targets, arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
