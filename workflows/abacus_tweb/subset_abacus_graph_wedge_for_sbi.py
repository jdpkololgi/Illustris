#!/usr/bin/env python3
"""Subset an Abacus graph to a survey-space wedge (RA/DEC/Z) and prepare SBI inputs.

Mirrors `subset_abacus_graph_cube_for_sbi.py` but selects nodes by sky coordinates
read from the annotated CutSky FITS, rather than by Cartesian cube bounds. The
wedge mask is computed in the **same filtered row order** used at graph build
time (`(IN_Y1|IN_Y5) & R_MAG_APP<19.5 & BOX_INDEX!=-1` by default), so node indices stay aligned
with the parent graph artifacts.

Outputs (analogous to the cube subset):
  <out-prefix>_points.npy
  <out-prefix>_points_xyz.npy
  <out-prefix>_edges_combined_idx.npy
  <out-prefix>_tetrahedra_idx.npy
  <out-prefix>_tetrahedra_volumes.npy
  <out-prefix>_metadata.json
  <out-prefix>_global_node_ids.npy   (mapping local->original global node id)
  <out-prefix>_wedge_targets.fits     (rows aligned to wedge node order)

After running this script, run:
  1) workflows/abacus_tweb/subset_cugraph_metrics_for_wedge.py (project full-volume
     cuGraph NPZ onto this wedge; do **not** re-run abacus_graph_features_cugraph.py)
  2) workflows/abacus_tweb/build_abacus_sbi_cache.py with the new gnn metadata + wedge FITS
"""

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

from shared.abacus_cutsky_selection import R_MAG_APP_BRIGHT_LT, cutsky_desi_bgs_mock_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graph-metadata",
        type=Path,
        required=True,
        help="Path to <prefix>_metadata.json produced by build_abacus_graph.py.",
    )
    p.add_argument(
        "--annotated-fits",
        type=Path,
        required=True,
        help="Annotated CutSky FITS containing LAMBDA1/2/3 (and optional 15-d derivative columns).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/graph_constructions"),
        help="Directory to write wedge graph artifacts + wedge FITS.",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Prefix for wedge outputs (e.g. abacus_delaunay_wedge_ra120_280_dec16p5_23p7_z0p218_0p296).",
    )
    p.add_argument(
        "--triples-fits",
        type=Path,
        default=None,
        help=(
            "Optional FITS containing explicit triple list selection. When provided, "
            "the subgraph nodes are selected by matching (FILE_NUM, HALO_INDEX, BOX_INDEX) "
            "against the annotated FITS in the graph-build filtered row order."
        ),
    )
    p.add_argument(
        "--no-wedge-bounds",
        action="store_true",
        help=(
            "Allow omitting RA/DEC/Z wedge bounds when using --triples-fits. Bounds (if provided) "
            "are only stored in metadata; they are not used to select nodes in triples mode."
        ),
    )
    p.add_argument(
        "--skip-base-mask-check",
        action="store_true",
        help=(
            "Skip strict verification that the reproduced base-mask count matches graph_meta['n_points']. "
            "Use only if the annotated FITS row-selection is known to be aligned but metadata differs."
        ),
    )
    # Wedge bounds (degrees / dimensionless redshift). Wedge mode requires all six unless --targets-only.
    p.add_argument("--ra-min", type=float, default=None, help="RA lower bound (deg, [0,360)).")
    p.add_argument("--ra-max", type=float, default=None, help="RA upper bound (deg, [0,360)). If RA_MIN > RA_MAX the wedge wraps through 0.")
    p.add_argument("--dec-min", type=float, default=None, help="DEC lower bound (deg).")
    p.add_argument("--dec-max", type=float, default=None, help="DEC upper bound (deg).")
    p.add_argument("--z-min", type=float, default=None, help="Redshift lower bound.")
    p.add_argument("--z-max", type=float, default=None, help="Redshift upper bound.")
    p.add_argument(
        "--ra-col",
        default="RA",
        help="RA column name in the annotated FITS (default: RA).",
    )
    p.add_argument(
        "--dec-col",
        default="DEC",
        help="DEC column name in the annotated FITS (default: DEC).",
    )
    p.add_argument(
        "--redshift-col",
        default="Z",
        help=(
            "Redshift column to use for wedge selection. Default: Z (observed; "
            "matches build_abacus_graph.py default and preserves RSD)."
        ),
    )
    p.add_argument(
        "--edge-chunk",
        type=int,
        default=5_000_000,
        help="Edges per chunk for streaming passes.",
    )
    p.add_argument(
        "--tet-chunk",
        type=int,
        default=2_000_000,
        help="Tetrahedra per chunk for streaming passes.",
    )
    p.add_argument(
        "--fits-out-name",
        type=str,
        default=None,
        help="Optional explicit output FITS filename. Default: <out-prefix>_wedge_targets.fits",
    )
    p.add_argument(
        "--targets-only",
        action="store_true",
        help=(
            "Only write the wedge targets FITS using an existing <out-prefix>_global_node_ids.npy "
            "in --out-dir. Does not rewrite wedge graph arrays/metadata."
        ),
    )
    return p.parse_args()


def _load_graph_meta(path: Path) -> dict:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_col(colnames: list[str], candidates: tuple[str, ...]) -> str:
    m = {c.upper(): c for c in colnames}
    for cand in candidates:
        r = m.get(cand.upper())
        if r is not None:
            return r
    raise KeyError(f"None of {candidates} found in FITS columns.")


def _fits_cols_available(fits_path: Path) -> list[str]:
    with fitsio.FITS(str(fits_path)) as f:
        return [str(x) for x in f[1].get_colnames()]


def _wedge_mask(
    ra: np.ndarray,
    dec: np.ndarray,
    zz: np.ndarray,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Return boolean wedge mask. Supports RA wrap when ra_min > ra_max."""
    if ra_min <= ra_max:
        m_ra = (ra >= ra_min) & (ra <= ra_max)
    else:
        m_ra = (ra >= ra_min) | (ra <= ra_max)
    return (
        m_ra
        & (dec >= dec_min)
        & (dec <= dec_max)
        & (zz >= z_min)
        & (zz <= z_max)
    )


def _cutsky_in_y1y5_mask(table: np.ndarray, *, cols_upper: dict[str, str]) -> np.ndarray:
    """(IN_Y1 == 1) | (IN_Y5 == 1) without the BGS bright R_MAG_APP cut."""
    in_y1 = cols_upper.get("IN_Y1")
    in_y5 = cols_upper.get("IN_Y5")
    if in_y1 is None or in_y5 is None:
        raise KeyError("IN_Y1 and IN_Y5 are required for CutSky Y1|Y5 footprint selection.")
    return (np.asarray(table[in_y1]) == 1) | (np.asarray(table[in_y5]) == 1)


def _base_mask_from_graph_meta(
    table: np.ndarray, *, graph_meta: dict, cols_upper: dict[str, str]
) -> np.ndarray:
    """Reproduce the graph-build base mask as specified in metadata.

  For CutSky graphs this is typically (IN_Y1|IN_Y5) with optional R_MAG_APP < 19.5
  (only when ``catalog_filters['r_mag_app_lt']`` is set, as in newer
  ``build_abacus_graph.py`` outputs) and optionally BOX_INDEX != -1.

  Legacy full-volume graphs (e.g. ``abacus_delaunay_split_hemis_*``) were built with
  ``apply_y1y5_filter`` but **without** the R_MAG cut; their metadata omits
  ``r_mag_app_lt``. Using the bright mock mask there mis-aligns node indices.
    """
    filters = graph_meta.get("catalog_filters") or {}
    m = np.ones(table.shape[0], dtype=bool)
    if bool(filters.get("apply_y1y5_filter", False)):
        if filters.get("r_mag_app_lt") is not None:
            m &= cutsky_desi_bgs_mock_mask(table)
        else:
            m &= _cutsky_in_y1y5_mask(table, cols_upper=cols_upper)
    if bool(filters.get("exclude_invalid_box_index", False)):
        box_index_col = str(filters.get("box_index_col", "BOX_INDEX"))
        box_col = cols_upper.get(box_index_col.upper()) or cols_upper.get("BOX_INDEX")
        if box_col is None:
            raise KeyError(
                f"exclude_invalid_box_index requested but {box_index_col} not present in FITS."
            )
        m &= np.asarray(table[box_col]) != -1
    return m


def _y1y5_filter_column_names(cols: list[str], *, graph_meta: dict) -> list[str]:
    """Return IN_Y1/IN_Y5/(optional R_MAG_APP) column names required for the base mask."""
    filters = graph_meta.get("catalog_filters") or {}
    if not bool(filters.get("apply_y1y5_filter", False)):
        return []
    out = [
        _resolve_col(cols, ("IN_Y1",)),
        _resolve_col(cols, ("IN_Y5",)),
    ]
    if filters.get("r_mag_app_lt") is not None:
        out.append(_resolve_col(cols, ("R_MAG_APP",)))
    return out


def _map_triples_to_global_ids_and_rows(
    *,
    annotated_fits: Path,
    graph_meta: dict,
    triples_fits: Path,
    skip_base_mask_check: bool,
    chunk_rows: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (wedge_global_ids, keep_rows) aligned to triples list order.

    wedge_global_ids are indices into the parent graph nodes *after* applying the base mask.
    keep_rows are the original (pre-base-mask) FITS row indices corresponding to each selected node.
    """
    triples_cols = _fits_cols_available(triples_fits)
    t_file = _resolve_col(triples_cols, ("FILE_NUM",))
    t_halo = _resolve_col(triples_cols, ("HALO_INDEX",))
    t_box = _resolve_col(triples_cols, ("BOX_INDEX", "BOXINDEX"))
    triples = fitsio.read(str(triples_fits), columns=[t_file, t_halo, t_box])
    targets_file = np.asarray(triples[t_file], dtype=np.int64)
    targets_halo = np.asarray(triples[t_halo], dtype=np.int64)
    targets_box = np.asarray(triples[t_box], dtype=np.int64)
    target_tuples = list(zip(targets_file.tolist(), targets_halo.tolist(), targets_box.tolist()))
    target_set = set(target_tuples)

    cols = _fits_cols_available(annotated_fits)
    cols_upper = {c.upper(): c for c in cols}
    # Filtering columns needed (depending on graph_meta) + the triple key columns.
    needed: list[str] = []
    filters = graph_meta.get("catalog_filters") or {}
    needed += _y1y5_filter_column_names(cols, graph_meta=graph_meta)
    if bool(filters.get("exclude_invalid_box_index", False)):
        box_index_col = str(filters.get("box_index_col", "BOX_INDEX"))
        needed += [_resolve_col(cols, (box_index_col, "BOX_INDEX"))]

    a_file = _resolve_col(cols, ("FILE_NUM",))
    a_halo = _resolve_col(cols, ("HALO_INDEX",))
    a_box = _resolve_col(cols, (str(filters.get("box_index_col", "BOX_INDEX")), "BOX_INDEX"))
    needed += [a_file, a_halo, a_box]
    needed = list(dict.fromkeys(needed))

    expected_n = int(graph_meta.get("n_points", -1))
    found_global: dict[tuple[int, int, int], int] = {}
    found_row: dict[tuple[int, int, int], int] = {}

    with fitsio.FITS(str(annotated_fits)) as f:
        nrows = int(f[1].get_nrows())
        filtered_index = 0
        match_triples = True
        for start in range(0, nrows, int(chunk_rows)):
            stop = min(nrows, start + int(chunk_rows))
            tab = f[1].read(rows=np.arange(start, stop, dtype=np.int64), columns=needed)
            cols_upper_chunk = {name.upper(): name for name in tab.dtype.names}
            base_mask = _base_mask_from_graph_meta(tab, graph_meta=graph_meta, cols_upper=cols_upper_chunk)
            if not np.any(base_mask):
                continue
            idx = np.nonzero(base_mask)[0].astype(np.int64)
            if match_triples:
                file_arr = np.asarray(tab[a_file][base_mask], dtype=np.int64)
                halo_arr = np.asarray(tab[a_halo][base_mask], dtype=np.int64)
                box_arr = np.asarray(tab[a_box][base_mask], dtype=np.int64)

                for k in range(idx.size):
                    tup = (int(file_arr[k]), int(halo_arr[k]), int(box_arr[k]))
                    if tup not in target_set:
                        continue
                    if tup in found_global:
                        continue
                    found_global[tup] = int(filtered_index + k)
                    found_row[tup] = int(start + idx[k])
                    if len(found_global) == len(target_set):
                        match_triples = False
                        break
            filtered_index += int(idx.size)

    if len(found_global) != len(target_set):
        missing = [t for t in target_tuples if t not in found_global]
        ex = ", ".join(str(x) for x in missing[:3])
        raise KeyError(
            f"Failed to match {len(missing):,}/{len(target_tuples):,} triples in annotated FITS. "
            f"Examples: {ex}"
        )

    if expected_n > 0 and not skip_base_mask_check and filtered_index != expected_n:
        raise ValueError(
            f"Base-mask row count after catalog scan is {filtered_index:,} but "
            f"graph metadata expects n_points={expected_n:,}. "
            "The annotated FITS / catalog_filters do not match the parent graph build "
            "(common cause: legacy split-hemis graphs omit catalog_filters['r_mag_app_lt'] "
            "but newer subset code applied the R_MAG_APP<19.5 cut)."
        )

    wedge_global_ids = np.asarray([found_global[t] for t in target_tuples], dtype=np.int64)
    keep_rows = np.asarray([found_row[t] for t in target_tuples], dtype=np.int64)
    return wedge_global_ids, keep_rows


def _load_filtered_ra_dec_z(
    annotated_fits: Path,
    *,
    graph_meta: dict,
    ra_col: str,
    dec_col: str,
    z_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce graph-build base mask, return (filtered_rows, ra, dec, z) for filtered rows.

    `filtered_rows` are the original CutSky row indices that survive the base mask.
    """
    cols = _fits_cols_available(annotated_fits)
    filters = graph_meta.get("catalog_filters") or {}
    needed_filter_cols: list[str] = list(_y1y5_filter_column_names(cols, graph_meta=graph_meta))
    if bool(filters.get("exclude_invalid_box_index", False)):
        box_index_col = str(filters.get("box_index_col", "BOX_INDEX"))
        needed_filter_cols += [_resolve_col(cols, (box_index_col, "BOX_INDEX"))]
    ra_resolved = _resolve_col(cols, (ra_col,))
    dec_resolved = _resolve_col(cols, (dec_col,))
    z_resolved = _resolve_col(cols, (z_col,))

    # Read a minimal column set, then apply the graph-build base mask, then keep RA/DEC/Z.
    needed = needed_filter_cols + [ra_resolved, dec_resolved, z_resolved]
    needed = list(dict.fromkeys(needed))  # de-dupe while preserving order
    table = fitsio.read(str(annotated_fits), columns=needed)

    expected_n = int(graph_meta.get("n_points", -1))
    cols_upper = {name.upper(): name for name in table.dtype.names}
    base_mask = _base_mask_from_graph_meta(table, graph_meta=graph_meta, cols_upper=cols_upper)
    n_after = int(np.count_nonzero(base_mask))
    if expected_n > 0 and n_after != expected_n:
        raise ValueError(
            f"Annotated FITS filter count mismatch: after base-mask got {n_after:,} "
            f"but graph metadata expects n_points={expected_n:,}. "
            "This annotated FITS must match the same selection/order as the graph build."
        )

    filtered_rows = np.nonzero(base_mask)[0].astype(np.int64)
    ra = np.asarray(table[ra_resolved][base_mask], dtype=np.float64)
    dec = np.asarray(table[dec_resolved][base_mask], dtype=np.float64)
    zz = np.asarray(table[z_resolved][base_mask], dtype=np.float64)
    return filtered_rows, ra, dec, zz


def _count_edges_in_mask(edges_path: Path, in_set: np.ndarray, chunk: int) -> int:
    edges = np.load(edges_path, mmap_mode="r")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must be (E,2); got {edges.shape} from {edges_path}")
    total = 0
    for i in range(0, edges.shape[0], chunk):
        e = edges[i : i + chunk]
        m = in_set[e[:, 0]] & in_set[e[:, 1]]
        total += int(np.count_nonzero(m))
    return total


def _write_edges_in_mask(
    edges_in_path: Path,
    *,
    in_set: np.ndarray,
    new_id: np.ndarray,
    out_path: Path,
    chunk: int,
) -> np.ndarray:
    edges = np.load(edges_in_path, mmap_mode="r")
    n_keep = _count_edges_in_mask(edges_in_path, in_set, chunk)
    out = np.empty((n_keep, 2), dtype=np.int32)
    w = 0
    for i in range(0, edges.shape[0], chunk):
        e = edges[i : i + chunk]
        m = in_set[e[:, 0]] & in_set[e[:, 1]]
        if not np.any(m):
            continue
        sel = e[m].astype(np.int64, copy=False)
        mapped = new_id[sel].astype(np.int32, copy=False)
        out[w : w + mapped.shape[0]] = mapped
        w += mapped.shape[0]
    if w != n_keep:
        raise RuntimeError(f"edge write mismatch: wrote {w} != expected {n_keep}")
    np.save(out_path, out)
    return out


def _count_tets_in_mask(tet_path: Path, in_set: np.ndarray, chunk: int) -> int:
    tets = np.load(tet_path, mmap_mode="r")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tetrahedra must be (T,4); got {tets.shape} from {tet_path}")
    total = 0
    for i in range(0, tets.shape[0], chunk):
        t = tets[i : i + chunk]
        m = in_set[t[:, 0]] & in_set[t[:, 1]] & in_set[t[:, 2]] & in_set[t[:, 3]]
        total += int(np.count_nonzero(m))
    return total


def _write_tets_in_mask(
    tet_in_path: Path,
    vol_in_path: Path,
    *,
    in_set: np.ndarray,
    new_id: np.ndarray,
    tet_out_path: Path,
    vol_out_path: Path,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    tets = np.load(tet_in_path, mmap_mode="r")
    vols = np.load(vol_in_path, mmap_mode="r")
    if vols.ndim != 1 or vols.shape[0] != tets.shape[0]:
        raise ValueError("tetrahedra volumes must be (T,) and match tetrahedra rows")
    n_keep = _count_tets_in_mask(tet_in_path, in_set, chunk)
    out_t = np.empty((n_keep, 4), dtype=np.int32)
    out_v = np.empty((n_keep,), dtype=np.float64)
    w = 0
    for i in range(0, tets.shape[0], chunk):
        t = tets[i : i + chunk]
        v = vols[i : i + chunk]
        m = in_set[t[:, 0]] & in_set[t[:, 1]] & in_set[t[:, 2]] & in_set[t[:, 3]]
        if not np.any(m):
            continue
        sel_t = t[m].astype(np.int64, copy=False)
        mapped = new_id[sel_t].astype(np.int32, copy=False)
        out_t[w : w + mapped.shape[0]] = mapped
        out_v[w : w + mapped.shape[0]] = v[m].astype(np.float64, copy=False)
        w += mapped.shape[0]
    if w != n_keep:
        raise RuntimeError(f"tet write mismatch: wrote {w} != expected {n_keep}")
    np.save(tet_out_path, out_t)
    np.save(vol_out_path, out_v)
    return out_t, out_v


def _write_wedge_targets_fits(
    *,
    annotated_fits: Path,
    graph_meta: dict,
    wedge_global_ids: np.ndarray,
    keep_rows: np.ndarray | None,
    out_path: Path,
) -> None:
    cols = _fits_cols_available(annotated_fits)
    if keep_rows is None:
        # Recompute keep_rows from wedge_global_ids by reproducing the base mask.
        filters = graph_meta.get("catalog_filters") or {}
        needed: list[str] = []
        needed += _y1y5_filter_column_names(cols, graph_meta=graph_meta)
        if bool(filters.get("exclude_invalid_box_index", False)):
            needed += [
                _resolve_col(cols, (str(filters.get("box_index_col", "BOX_INDEX")), "BOX_INDEX"))
            ]
        needed = list(dict.fromkeys(needed))
        if needed:
            tab_mask_cols = fitsio.read(str(annotated_fits), columns=needed)
            cols_upper = {name.upper(): name for name in tab_mask_cols.dtype.names}
            base_mask = _base_mask_from_graph_meta(
                tab_mask_cols, graph_meta=graph_meta, cols_upper=cols_upper
            )
        else:
            with fitsio.FITS(str(annotated_fits)) as f:
                nrows = int(f[1].get_nrows())
            base_mask = np.ones(nrows, dtype=bool)

        expected_n = int(graph_meta.get("n_points", -1))
        n_after = int(np.count_nonzero(base_mask))
        if expected_n > 0 and n_after != expected_n:
            raise ValueError(
                f"Annotated FITS filter count mismatch: after base-mask got {n_after:,} "
                f"but graph metadata expects n_points={expected_n:,}. "
                "This annotated FITS must match the same selection/order as the graph build."
            )

        filtered_rows = np.nonzero(base_mask)[0].astype(np.int64)
        keep_rows = filtered_rows[wedge_global_ids.astype(np.int64)]

    want = [
        "RA",
        "DEC",
        "Z",
        "Z_COSMO",
        "FILE_NUM",
        "HALO_INDEX",
        "BOX_INDEX",
        "CWEB",
        "LAMBDA1",
        "LAMBDA2",
        "LAMBDA3",
        "DLAM1_DX",
        "DLAM1_DY",
        "DLAM1_DZ",
        "DLAM2_DX",
        "DLAM2_DY",
        "DLAM2_DZ",
        "DLAM3_DX",
        "DLAM3_DY",
        "DLAM3_DZ",
        "LAP_LAM1",
        "LAP_LAM2",
        "LAP_LAM3",
    ]
    keep_cols = [c for c in want if c in cols]
    if not all(x in keep_cols for x in ("LAMBDA1", "LAMBDA2", "LAMBDA3")):
        raise KeyError("Annotated FITS is missing LAMBDA1/2/3; cannot build SBI targets.")

    sub = fitsio.read(str(annotated_fits), columns=keep_cols, rows=keep_rows.tolist())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(out_path), sub, clobber=True)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_meta_path = args.graph_metadata.expanduser().resolve()
    graph_meta = _load_graph_meta(graph_meta_path)
    base_dir = graph_meta_path.parent
    files = graph_meta.get("files", {})

    if args.targets_only:
        global_ids_path = out_dir / f"{args.out_prefix}_global_node_ids.npy"
        if not global_ids_path.exists():
            raise FileNotFoundError(
                f"targets-only mode requires existing global ids mapping at: {global_ids_path}"
            )
        wedge_global_ids = np.load(global_ids_path).astype(np.int64, copy=False)
        fits_out = out_dir / (args.fits_out_name or f"{args.out_prefix}_wedge_targets.fits")
        _write_wedge_targets_fits(
            annotated_fits=args.annotated_fits.expanduser().resolve(),
            graph_meta=graph_meta,
            wedge_global_ids=wedge_global_ids,
            keep_rows=None,
            out_path=fits_out,
        )
        print(f"Wrote wedge targets FITS: {fits_out}")
        return

    missing_bounds = [
        name
        for name in ("ra_min", "ra_max", "dec_min", "dec_max", "z_min", "z_max")
        if getattr(args, name) is None
    ]
    if missing_bounds and not (args.triples_fits is not None and args.no_wedge_bounds):
        raise SystemExit(
            "Missing required wedge bounds arguments: "
            + ", ".join(f"--{n.replace('_', '-')}" for n in missing_bounds)
            + "\n(Provide bounds to build the wedge subgraph, use --targets-only, "
            + "or pass --triples-fits with --no-wedge-bounds.)"
        )

    points_path = base_dir / files.get("points", f"{graph_meta.get('prefix')}_points.npy")
    points_xyz_path = base_dir / files.get("points_xyz", f"{graph_meta.get('prefix')}_points_xyz.npy")
    edges_path = base_dir / files.get("edges", f"{graph_meta.get('prefix')}_edges_combined_idx.npy")
    tet_path = base_dir / files.get("tetrahedra_idx", f"{graph_meta.get('prefix')}_tetrahedra_idx.npy")
    vol_path = base_dir / files.get("tetrahedra_volumes", f"{graph_meta.get('prefix')}_tetrahedra_volumes.npy")

    for p in (points_path, points_xyz_path, edges_path, tet_path, vol_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    annotated_fits = args.annotated_fits.expanduser().resolve()
    keep_rows: np.ndarray | None = None
    if args.triples_fits is not None:
        wedge_global_ids, keep_rows = _map_triples_to_global_ids_and_rows(
            annotated_fits=annotated_fits,
            graph_meta=graph_meta,
            triples_fits=args.triples_fits.expanduser().resolve(),
            skip_base_mask_check=bool(args.skip_base_mask_check),
        )
        in_wedge = None
        n_full_graph = int(graph_meta.get("n_points", -1))
        if n_full_graph <= 0:
            raise ValueError("graph_meta.n_points is required for triples-fits mode.")
    else:
        filtered_rows, ra, dec, zz = _load_filtered_ra_dec_z(
            annotated_fits,
            graph_meta=graph_meta,
            ra_col=args.ra_col,
            dec_col=args.dec_col,
            z_col=args.redshift_col,
        )
        n_full_graph = int(filtered_rows.size)
        in_wedge = _wedge_mask(
            ra,
            dec,
            zz,
            ra_min=float(args.ra_min),
            ra_max=float(args.ra_max),
            dec_min=float(args.dec_min),
            dec_max=float(args.dec_max),
            z_min=float(args.z_min),
            z_max=float(args.z_max),
        )
        wedge_global_ids = np.nonzero(in_wedge)[0].astype(np.int64)
        if wedge_global_ids.size == 0:
            raise ValueError("Wedge mask selected zero nodes. Check bounds and column choices.")

    bounds = {
        "ra_min": None if args.ra_min is None else float(args.ra_min),
        "ra_max": None if args.ra_max is None else float(args.ra_max),
        "dec_min": None if args.dec_min is None else float(args.dec_min),
        "dec_max": None if args.dec_max is None else float(args.dec_max),
        "z_min": None if args.z_min is None else float(args.z_min),
        "z_max": None if args.z_max is None else float(args.z_max),
        "ra_col": str(args.ra_col),
        "dec_col": str(args.dec_col),
        "redshift_col": str(args.redshift_col),
        "ra_wrap": False if (args.ra_min is None or args.ra_max is None) else bool(args.ra_min > args.ra_max),
        "triples_fits": None if args.triples_fits is None else str(args.triples_fits.expanduser().resolve()),
        "selection_mode": "triples" if args.triples_fits is not None else "wedge_bounds",
    }

    global_ids_out = out_dir / f"{args.out_prefix}_global_node_ids.npy"
    np.save(global_ids_out, wedge_global_ids)

    in_set = np.zeros((n_full_graph,), dtype=bool)
    in_set[wedge_global_ids] = True

    new_id = np.full((n_full_graph,), -1, dtype=np.int32)
    new_id[wedge_global_ids] = np.arange(wedge_global_ids.size, dtype=np.int32)

    points = np.load(points_path, mmap_mode="r")
    xyz = np.load(points_xyz_path, mmap_mode="r")
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"points_xyz must be (N,3); got {xyz.shape}")
    if xyz.shape[0] != n_full_graph:
        raise ValueError(
            f"Parent points_xyz N={xyz.shape[0]:,} disagrees with filtered FITS rows N={n_full_graph:,}."
        )

    wedge_points = np.asarray(points[wedge_global_ids], dtype=np.float64)
    wedge_xyz = np.asarray(xyz[wedge_global_ids, :3], dtype=np.float64)

    out_points = out_dir / f"{args.out_prefix}_points.npy"
    out_xyz = out_dir / f"{args.out_prefix}_points_xyz.npy"
    np.save(out_points, wedge_points)
    np.save(out_xyz, wedge_xyz)

    out_edges = out_dir / f"{args.out_prefix}_edges_combined_idx.npy"
    out_tets = out_dir / f"{args.out_prefix}_tetrahedra_idx.npy"
    out_vols = out_dir / f"{args.out_prefix}_tetrahedra_volumes.npy"

    edges_wedge = _write_edges_in_mask(
        edges_path,
        in_set=in_set,
        new_id=new_id,
        out_path=out_edges,
        chunk=int(args.edge_chunk),
    )
    tets_wedge, vols_wedge = _write_tets_in_mask(
        tet_path,
        vol_path,
        in_set=in_set,
        new_id=new_id,
        tet_out_path=out_tets,
        vol_out_path=out_vols,
        chunk=int(args.tet_chunk),
    )

    fits_out = out_dir / (args.fits_out_name or f"{args.out_prefix}_wedge_targets.fits")
    _write_wedge_targets_fits(
        annotated_fits=annotated_fits,
        graph_meta=graph_meta,
        wedge_global_ids=wedge_global_ids,
        keep_rows=keep_rows,
        out_path=fits_out,
    )

    meta_out = out_dir / f"{args.out_prefix}_metadata.json"
    wedge_meta = dict(graph_meta)
    wedge_meta.update(
        {
            "prefix": args.out_prefix,
            "source": "exp1_triples_subset" if args.triples_fits is not None else "wedge_subset",
            "wedge_bounds": bounds,
            "parent_graph_metadata": str(graph_meta_path),
            "annotated_fits": str(annotated_fits),
            "n_points": int(wedge_points.shape[0]),
            "n_point_columns": int(wedge_points.shape[1]),
            "n_edges": int(edges_wedge.shape[0]),
            "n_tetrahedra": int(tets_wedge.shape[0]),
            "files": {
                "points": out_points.name,
                "points_xyz": out_xyz.name,
                "edges": out_edges.name,
                "tetrahedra_idx": out_tets.name,
                "tetrahedra_volumes": out_vols.name,
                "global_node_ids": global_ids_out.name,
                "wedge_targets_fits": fits_out.name,
            },
        }
    )
    with meta_out.open("w", encoding="utf-8") as f:
        json.dump(wedge_meta, f, indent=2, sort_keys=True)

    print(f"Wedge nodes: {wedge_points.shape[0]:,} / {n_full_graph:,}")
    print(f"Wedge edges: {edges_wedge.shape[0]:,}")
    print(f"Wedge tetrahedra: {tets_wedge.shape[0]:,}")
    print(f"Wrote wedge graph metadata: {meta_out}")
    print(f"Wrote wedge targets FITS: {fits_out}")
    print(
        "Next: run subset_cugraph_metrics_for_wedge.py (full cuGraph -> wedge), "
        "then build_abacus_sbi_cache.py."
    )


if __name__ == "__main__":
    main()
