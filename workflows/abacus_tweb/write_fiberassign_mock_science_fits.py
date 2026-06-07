#!/usr/bin/env python3
"""Prepare a fiberassign ``mock_bgs_maglim.fits`` catalog for graph build / training.

Two intended use paths (do **not** triple-dedupe like legacy stage-3 science export):

**Graph / inference (DESI parity)**
  - Keep all mag-lim rows (default).
  - One graph node per row; stable key = ``TARGETID``.
  - Build with ``build_abacus_graph.py --no-apply-y1y5-filter --no-exclude-invalid-box-index``.

**Supervised mock training**
  - Same full row list for the graph (fiber duplicates stay as separate nodes).
  - Drop ``BOX_INDEX == -1`` only when building *labels* / train masks (see
    ``join_cutsky_eigs_to_fiberassign_catalog.py``), not necessarily when building
    the inference graph.
  - Eigenvalues come from annotated **CutSky** via triple join, not from re-running
    ``annotate_cutsky_with_tweb_eigs.py`` on this mock.

This script only copies/filters the input FITS (optional ``ZWARN`` cap). It never
dedupes on ``(FILE_NUM, HALO_INDEX, BOX_INDEX)``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import fitsio
import numpy as np

DEFAULT_INPUT = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/"
    "path1_fiberassign_20260604_083322/mock_bgs_maglim.fits"
)
DEFAULT_OUTPUT = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/"
    "path1_fiberassign_20260604_083322/mock_bgs_maglim_graph_ready.fits"
)
DEFAULT_SCIENCE_COLS = (
    "TARGETID",
    "RA",
    "DEC",
    "Z",
    "FILE_NUM",
    "HALO_INDEX",
    "BOX_INDEX",
    "BGS_TARGET",
    "ZWARN",
    "DELTACHI2",
    "SPECTYPE",
)


def _resolve_col(colnames: list[str], name: str) -> str | None:
    m = {c.upper(): c for c in colnames}
    return m.get(name.upper())


def _require_col(colnames: list[str], name: str) -> str:
    resolved = _resolve_col(colnames, name)
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
    return [_require_col(colnames, c) for c in want]


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
    p.add_argument("--all-columns", action="store_true", help="Write all input columns.")
    p.add_argument(
        "--zwarn-max",
        type=int,
        default=0,
        help="Keep rows with ZWARN <= this value (default: 0). Use -1 to disable.",
    )
    p.add_argument(
        "--copy-only",
        action="store_true",
        help="Copy input to output unchanged (ignore filters/column subset).",
    )
    p.add_argument("--meta-json", type=Path, default=None)
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

    _require_col(colnames, "TARGETID")
    for required in ("FILE_NUM", "HALO_INDEX", "BOX_INDEX", "RA", "DEC", "Z"):
        _require_col(colnames, required)

    n_zwarn_skip = 0
    if args.copy_only:
        shutil.copy2(inp, out)
        n_out = n_total
        out_cols = colnames
        filters: dict[str, object] = {"copy_only": True}
    else:
        out_cols = _resolve_output_columns(colnames, args.columns, args.all_columns)
        zwarn_c = _resolve_col(colnames, "ZWARN")
        read_cols = list(dict.fromkeys(colnames if args.all_columns else out_cols + ([zwarn_c] if zwarn_c else [])))

        parts: list[np.ndarray] = []
        apply_zwarn = zwarn_c is not None and int(args.zwarn_max) >= 0

        for _start, tab in _iter_fits_chunks(inp, read_cols, args.chunk_size):
            if apply_zwarn:
                m = tab[zwarn_c] <= int(args.zwarn_max)
                n_zwarn_skip += int(np.count_nonzero(~m))
                if not np.any(m):
                    continue
                tab = tab[m]
            parts.append(tab if args.all_columns else tab[out_cols])

        if not parts:
            raise RuntimeError(f"No rows passed filters for {inp}")
        table = np.concatenate(parts)
        fitsio.write(str(out), table, clobber=True)
        n_out = int(table.size)
        filters = {}
        if apply_zwarn:
            filters["ZWARN"] = f"<= {int(args.zwarn_max)}"

    meta_path = args.meta_json
    if meta_path is None and not args.no_meta_json:
        meta_path = out.with_suffix(out.suffix + ".json")

    meta = {
        "input_fits": str(inp),
        "output_fits": str(out),
        "n_rows_total": n_total,
        "n_rows_out": n_out,
        "output_columns": out_cols,
        "node_key": "TARGETID",
        "dedupe_halo_triple": False,
        "filters": filters,
        "graph_build_hint": (
            "build_abacus_graph.py --catalog-path <output> "
            "--no-apply-y1y5-filter --no-exclude-invalid-box-index"
        ),
        "training_labels_hint": "join_cutsky_eigs_to_fiberassign_catalog.py (triple join from annotated CutSky)",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    if meta_path is not None:
        meta_path.expanduser().resolve().write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {n_out:,} rows -> {out}")
    print(f"node_key=TARGETID dedupe_halo_triple=False")
    if not args.copy_only:
        print(f"zwarn_skip={n_zwarn_skip:,}")
    if meta_path is not None:
        print(f"Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
