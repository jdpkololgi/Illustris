#!/usr/bin/env python3
"""Write filtered stage-3 science FITS from datcomb_brightwdup.fits.

Filters (FITS row order):
  - COLLISION == 0
  - BOX_INDEX != -1
  - dedupe on (FILE_NUM, HALO_INDEX, BOX_INDEX), first row wins
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import fitsio
import numpy as np

DEFAULT_INPUT = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/fba0/"
    "datcomb_brightwdup.fits"
)
DEFAULT_OUTPUT = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/fba0/"
    "stage3_postcollision_dedup_science.fits"
)
DEFAULT_SCIENCE_COLS = (
    "FILE_NUM",
    "HALO_INDEX",
    "BOX_INDEX",
    "RA",
    "DEC",
    "Z",
    "COLLISION",
    "Z_COSMO",
    "RSDZ",
    "TARGETID",
)
TRIPLE_COLS = ("FILE_NUM", "HALO_INDEX", "BOX_INDEX")


def _resolve_col(colnames: list[str], name: str) -> str:
    m = {c.upper(): c for c in colnames}
    resolved = m.get(name.upper())
    if resolved is None:
        raise KeyError(f"Column {name!r} not in FITS (have {len(colnames)} cols).")
    return resolved


def _iter_fits_chunks(path: Path, columns: list[str], chunk_size: int):
    with fitsio.FITS(str(path)) as ff:
        hdu = ff[1]
        n = int(hdu.get_nrows())
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield start, hdu[columns][start:end]


def _resolve_output_columns(colnames: list[str], requested: list[str] | None, all_columns: bool) -> list[str]:
    if all_columns:
        return list(colnames)
    want = list(requested or DEFAULT_SCIENCE_COLS)
    return [_resolve_col(colnames, c) for c in want]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    p.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help=f"Output columns (default: {', '.join(DEFAULT_SCIENCE_COLS)}).",
    )
    p.add_argument(
        "--all-columns",
        action="store_true",
        help="Write all columns from the input datcomb table.",
    )
    p.add_argument(
        "--meta-json",
        type=Path,
        default=None,
        help="Optional path for export metadata JSON (default: <output>.json).",
    )
    p.add_argument("--no-meta-json", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with fitsio.FITS(str(inp)) as ff:
        colnames = [str(c) for c in ff[1].get_colnames()]
        n_total = int(ff[1].get_nrows())

    out_cols = _resolve_output_columns(colnames, args.columns, args.all_columns)
    fn_c, hi_c, bi_c = (_resolve_col(colnames, c) for c in TRIPLE_COLS)
    coll_c = _resolve_col(colnames, "COLLISION")
    read_cols = list(dict.fromkeys(colnames if args.all_columns else out_cols + [fn_c, hi_c, bi_c, coll_c]))

    seen: set[tuple[int, int, int]] = set()
    parts: list[np.ndarray] = []
    n_collision_pass = 0
    n_box_valid_pass = 0
    n_dedup_skip = 0
    n_box_invalid_skip = 0

    for _start, tab in _iter_fits_chunks(inp, read_cols, args.chunk_size):
        m = tab[coll_c] == 0
        if not np.any(m):
            continue
        n_collision_pass += int(np.count_nonzero(m))

        sub = tab[m]
        bi = sub[bi_c]
        box_ok = bi != -1
        if not np.any(box_ok):
            n_box_invalid_skip += int(sub.size)
            continue
        n_box_invalid_skip += int(np.count_nonzero(~box_ok))
        sub = sub[box_ok]
        n_box_valid_pass += int(sub.size)

        fn = sub[fn_c].astype(np.int64, copy=False)
        hi = sub[hi_c].astype(np.int64, copy=False)
        bi_keep = sub[bi_c].astype(np.int64, copy=False)

        keep_idx = []
        for i in range(fn.size):
            key = (int(fn[i]), int(hi[i]), int(bi_keep[i]))
            if key in seen:
                n_dedup_skip += 1
                continue
            seen.add(key)
            keep_idx.append(i)

        if not keep_idx:
            continue
        idx = np.asarray(keep_idx, dtype=np.int64)
        chunk = sub[idx]
        parts.append(chunk if args.all_columns else chunk[out_cols])

    if not parts:
        raise RuntimeError(f"No rows passed filters for {inp}")

    table = np.concatenate(parts)
    fitsio.write(str(out), table, clobber=True)

    meta_path = args.meta_json
    if meta_path is None and not args.no_meta_json:
        meta_path = out.with_suffix(out.suffix + ".json")

    meta = {
        "input_fits": str(inp),
        "output_fits": str(out),
        "n_rows_total": n_total,
        "n_rows_collision_eq0": n_collision_pass,
        "n_rows_box_index_valid": n_box_valid_pass,
        "n_rows_box_index_invalid_skipped": n_box_invalid_skip,
        "n_dedup_skip": n_dedup_skip,
        "n_rows_out": int(table.size),
        "output_columns": out_cols,
        "filters": {
            "COLLISION": "== 0",
            "BOX_INDEX": "!= -1",
            "dedupe_halo_triple": True,
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    if meta_path is not None:
        meta_path = meta_path.expanduser().resolve()
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {table.size:,} rows -> {out}")
    print(f"Columns ({len(out_cols)}): {out_cols}")
    print(
        f"Filter stats: total={n_total:,} collision==0={n_collision_pass:,} "
        f"box_valid={n_box_valid_pass:,} box_invalid_skip={n_box_invalid_skip:,} "
        f"dedup_skip={n_dedup_skip:,} out={table.size:,}"
    )
    if meta_path is not None:
        print(f"Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
