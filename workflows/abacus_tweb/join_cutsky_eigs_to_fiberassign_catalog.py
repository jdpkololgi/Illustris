#!/usr/bin/env python3
"""Join T-Web eigenvalues from annotated CutSky onto a fiberassign catalog (per row).

Graph nodes are fiberassign rows (``TARGETID``); labels use halo triple linkage:

  (FILE_NUM, HALO_INDEX, BOX_INDEX) -> LAMBDA1/2/3 (+ optional 15-d derivatives)

The CutSky annotated FITS is scanned once to build a triple lookup table. When
the same triple appears multiple times in CutSky, the **first** occurrence wins
(lookup dedupe only — graph rows are **not** deduped). Every fiberassign row with
the same triple receives the same eigenvalues.

Rows with ``BOX_INDEX == -1`` are written with NaN eigenvalues and
``HAS_LABEL=0`` (inference-only unless you override).

Output row order matches ``build_abacus_graph.py`` catalog filtering when
``--graph-metadata`` is supplied (same base mask / row order as ``n_points``).
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

from shared.abacus_cutsky_selection import cutsky_desi_bgs_mock_mask

TRIPLE_COLS = ("FILE_NUM", "HALO_INDEX", "BOX_INDEX")
EIG_COLS = ("LAMBDA1", "LAMBDA2", "LAMBDA3")
DERIV_COLS = (
    "DLAM1_DX", "DLAM1_DY", "DLAM1_DZ",
    "DLAM2_DX", "DLAM2_DY", "DLAM2_DZ",
    "DLAM3_DX", "DLAM3_DY", "DLAM3_DZ",
    "LAP_LAM1", "LAP_LAM2", "LAP_LAM3",
)
DEFAULT_ANNOTATED_CUTSKY = (
    "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_20260427_rsmooth_7/"
    "cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs7_ngrid2048_thr0p2_15d.fits"
)
DEFAULT_FIBERASSIGN = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/"
    "path1_fiberassign_20260604_083322/mock_bgs_maglim.fits"
)


def _resolve_col(colnames: list[str], name: str) -> str:
    m = {c.upper(): c for c in colnames}
    r = m.get(name.upper())
    if r is None:
        raise KeyError(f"Column {name!r} not in FITS.")
    return r


def _iter_fits_chunks(path: Path, columns: list[str], chunk_size: int):
    with fitsio.FITS(str(path)) as ff:
        hdu = ff[1]
        n = int(hdu.get_nrows())
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield start, hdu[columns][start:end]


def _base_mask_fiberassign(table: np.ndarray, graph_meta: dict | None) -> np.ndarray:
    """Reproduce build_abacus_graph catalog mask for fiberassign mocks."""
    n = len(table)
    mask = np.ones(n, dtype=bool)
    filters = (graph_meta or {}).get("catalog_filters") or {}
    if bool(filters.get("apply_y1y5_filter", False)):
        mask &= cutsky_desi_bgs_mock_mask(table)
    if bool(filters.get("exclude_invalid_box_index", False)):
        box_col = str(filters.get("box_index_col", "BOX_INDEX"))
        names = {c.upper(): c for c in table.dtype.names}
        bi = table[names[box_col.upper()]]
        mask &= bi != -1
    return mask


def build_triple_lookup(
    annotated: Path,
    *,
    chunk_size: int,
    include_derivatives: bool,
) -> tuple[dict[tuple[int, int, int], dict[str, float]], list[str]]:
    with fitsio.FITS(str(annotated)) as ff:
        cols = [str(c) for c in ff[1].get_colnames()]
    fn_c, hi_c, bi_c = (_resolve_col(cols, c) for c in TRIPLE_COLS)
    l1_c, l2_c, l3_c = (_resolve_col(cols, c) for c in EIG_COLS)
    read_cols = list(dict.fromkeys([fn_c, hi_c, bi_c, l1_c, l2_c, l3_c]))
    deriv_resolved: list[str] = []
    if include_derivatives:
        for d in DERIV_COLS:
            rc = _resolve_col(cols, d)
            deriv_resolved.append(rc)
            read_cols.append(rc)

    lookup: dict[tuple[int, int, int], dict[str, float]] = {}
    n_dup_lookup_skip = 0

    for _start, tab in _iter_fits_chunks(annotated, read_cols, chunk_size):
        fn = tab[fn_c].astype(np.int64, copy=False)
        hi = tab[hi_c].astype(np.int64, copy=False)
        bi = tab[bi_c].astype(np.int64, copy=False)
        for i in range(fn.size):
            key = (int(fn[i]), int(hi[i]), int(bi[i]))
            if key in lookup:
                n_dup_lookup_skip += 1
                continue
            row: dict[str, float] = {
                "LAMBDA1": float(tab[l1_c][i]),
                "LAMBDA2": float(tab[l2_c][i]),
                "LAMBDA3": float(tab[l3_c][i]),
            }
            for d, rc in zip(DERIV_COLS, deriv_resolved):
                row[d] = float(tab[rc][i])
            lookup[key] = row

    print(
        f"Triple lookup from {annotated.name}: {len(lookup):,} unique triples "
        f"(skipped {n_dup_lookup_skip:,} duplicate triple rows in CutSky scan)."
    )
    return lookup, ["LAMBDA1", "LAMBDA2", "LAMBDA3"] + (list(DERIV_COLS) if include_derivatives else [])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fiberassign-fits", type=Path, default=Path(DEFAULT_FIBERASSIGN))
    p.add_argument("--annotated-cutsky", type=Path, default=Path(DEFAULT_ANNOTATED_CUTSKY))
    p.add_argument(
        "--graph-metadata",
        type=Path,
        default=None,
        help="Optional graph metadata JSON to reproduce build_abacus_graph row mask/order.",
    )
    p.add_argument("--output", type=Path, required=True, help="Output targets FITS path.")
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument(
        "--include-derivatives",
        action="store_true",
        default=True,
        help="Include 15-d derivative columns when present in annotated CutSky (default: true).",
    )
    p.add_argument(
        "--no-include-derivatives",
        action="store_false",
        dest="include_derivatives",
    )
    p.add_argument(
        "--eig-threshold",
        type=float,
        default=0.2,
        help="Threshold for CWEB class count from eigenvalues.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fiber = args.fiberassign_fits.expanduser().resolve()
    annotated = args.annotated_cutsky.expanduser().resolve()
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    graph_meta = None
    if args.graph_metadata is not None:
        graph_meta = json.loads(args.graph_metadata.expanduser().resolve().read_text())

    t0 = time.time()
    lookup, value_cols = build_triple_lookup(
        annotated, chunk_size=args.chunk_size, include_derivatives=args.include_derivatives
    )

    with fitsio.FITS(str(fiber)) as ff:
        fcols = [str(c) for c in ff[1].get_colnames()]
        n_total = int(ff[1].get_nrows())

    tid_c = _resolve_col(fcols, "TARGETID")
    ra_c, dec_c, z_c = _resolve_col(fcols, "RA"), _resolve_col(fcols, "DEC"), _resolve_col(fcols, "Z")
    fn_c, hi_c, bi_c = (_resolve_col(fcols, c) for c in TRIPLE_COLS)
    read_cols = list(dict.fromkeys([tid_c, ra_c, dec_c, z_c, fn_c, hi_c, bi_c]))

    dtype = [
        ("TARGETID", "i8"),
        ("RA", "f8"),
        ("DEC", "f8"),
        ("Z", "f8"),
        ("FILE_NUM", "i4"),
        ("HALO_INDEX", "i4"),
        ("BOX_INDEX", "i4"),
        ("HAS_LABEL", "i1"),
        ("CWEB", "i1"),
        ("LAMBDA1", "f4"),
        ("LAMBDA2", "f4"),
        ("LAMBDA3", "f4"),
    ]
    for d in DERIV_COLS:
        if d in value_cols:
            dtype.append((d, "f4"))

    parts: list[np.ndarray] = []
    n_join_ok = 0
    n_join_miss = 0
    n_box_minus1 = 0
    n_labeled = 0

    for _start, tab in _iter_fits_chunks(fiber, read_cols, args.chunk_size):
        base_mask = _base_mask_fiberassign(tab, graph_meta)
        if not np.any(base_mask):
            continue
        sub = tab[base_mask]
        n = sub.size
        block = np.empty(n, dtype=dtype)
        block["TARGETID"] = sub[tid_c].astype(np.int64, copy=False)
        block["RA"] = sub[ra_c].astype(np.float64, copy=False)
        block["DEC"] = sub[dec_c].astype(np.float64, copy=False)
        block["Z"] = sub[z_c].astype(np.float64, copy=False)
        block["FILE_NUM"] = sub[fn_c].astype(np.int32, copy=False)
        block["HALO_INDEX"] = sub[hi_c].astype(np.int32, copy=False)
        block["BOX_INDEX"] = sub[bi_c].astype(np.int32, copy=False)

        for i in range(n):
            bi = int(block["BOX_INDEX"][i])
            if bi == -1:
                n_box_minus1 += 1
                block["HAS_LABEL"][i] = 0
                block["LAMBDA1"][i] = np.nan
                block["LAMBDA2"][i] = np.nan
                block["LAMBDA3"][i] = np.nan
                block["CWEB"][i] = -1
                for d in DERIV_COLS:
                    if d in block.dtype.names:
                        block[d][i] = np.nan
                continue

            key = (int(block["FILE_NUM"][i]), int(block["HALO_INDEX"][i]), bi)
            ev = lookup.get(key)
            if ev is None:
                n_join_miss += 1
                block["HAS_LABEL"][i] = 0
                block["LAMBDA1"][i] = np.nan
                block["LAMBDA2"][i] = np.nan
                block["LAMBDA3"][i] = np.nan
                block["CWEB"][i] = -1
                for d in DERIV_COLS:
                    if d in block.dtype.names:
                        block[d][i] = np.nan
                continue

            n_join_ok += 1
            n_labeled += 1
            l1, l2, l3 = ev["LAMBDA1"], ev["LAMBDA2"], ev["LAMBDA3"]
            block["LAMBDA1"][i] = l1
            block["LAMBDA2"][i] = l2
            block["LAMBDA3"][i] = l3
            block["CWEB"][i] = int(
                (l1 > args.eig_threshold) + (l2 > args.eig_threshold) + (l3 > args.eig_threshold)
            )
            block["HAS_LABEL"][i] = 1
            for d in DERIV_COLS:
                if d in block.dtype.names:
                    block[d][i] = ev[d]

        parts.append(block)

    if not parts:
        raise RuntimeError(f"No rows after graph mask for {fiber}")

    table = np.concatenate(parts)
    if graph_meta is not None:
        expected = int(graph_meta.get("n_points", -1))
        if expected > 0 and table.size != expected:
            raise ValueError(
                f"Output row count {table.size:,} != graph n_points {expected:,}. "
                "Check --fiberassign-fits and --graph-metadata alignment."
            )

    fitsio.write(str(out), table, clobber=True)
    meta = {
        "fiberassign_fits": str(fiber),
        "annotated_cutsky": str(annotated),
        "graph_metadata": str(args.graph_metadata) if args.graph_metadata else None,
        "n_rows_fiberassign_total": n_total,
        "n_rows_out": int(table.size),
        "n_join_ok": n_join_ok,
        "n_join_miss": n_join_miss,
        "n_box_index_minus1": n_box_minus1,
        "n_labeled": n_labeled,
        "node_key": "TARGETID",
        "label_join": "annotated_cutsky triple (FILE_NUM, HALO_INDEX, BOX_INDEX)",
        "lookup_dedupe_triple_only": True,
        "graph_dedupe": False,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path = out.with_suffix(out.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {table.size:,} rows -> {out}")
    print(
        f"join_ok={n_join_ok:,} join_miss={n_join_miss:,} "
        f"box_index==-1={n_box_minus1:,} labeled={n_labeled:,}"
    )
    print(f"Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
