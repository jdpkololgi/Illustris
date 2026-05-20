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
  1) workflows/abacus_tweb/abacus_graph_features_cugraph.py on the new prefix
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
    in_y1 = _resolve_col(cols, ("IN_Y1",))
    in_y5 = _resolve_col(cols, ("IN_Y5",))
    r_mag_col = _resolve_col(cols, ("R_MAG_APP",))
    box_index_col = str(graph_meta.get("catalog_filters", {}).get("box_index_col", "BOX_INDEX"))
    box_col = _resolve_col(cols, (box_index_col, "BOX_INDEX"))
    ra_resolved = _resolve_col(cols, (ra_col,))
    dec_resolved = _resolve_col(cols, (dec_col,))
    z_resolved = _resolve_col(cols, (z_col,))

    # Read a minimal column set, then apply the graph-build base mask, then keep RA/DEC/Z.
    needed = [in_y1, in_y5, r_mag_col, box_col, ra_resolved, dec_resolved, z_resolved]
    needed = list(dict.fromkeys(needed))  # de-dupe while preserving order
    table = fitsio.read(str(annotated_fits), columns=needed)

    base_mask = cutsky_desi_bgs_mock_mask(table)
    base_mask &= table[box_col] != -1

    expected_n = int(graph_meta.get("n_points", -1))
    n_after = int(np.count_nonzero(base_mask))
    if expected_n > 0 and n_after != expected_n:
        raise ValueError(
            f"Annotated FITS filter count mismatch: after (Y1|Y5 & R_MAG_APP<{R_MAG_APP_BRIGHT_LT:g} "
            f"& {box_col}!=-1) got {n_after:,} "
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
    out_path: Path,
) -> None:
    cols = _fits_cols_available(annotated_fits)

    in_y1 = _resolve_col(cols, ("IN_Y1",))
    in_y5 = _resolve_col(cols, ("IN_Y5",))
    r_mag = _resolve_col(cols, ("R_MAG_APP",))
    box_col = _resolve_col(cols, (str(graph_meta.get("catalog_filters", {}).get("box_index_col", "BOX_INDEX")), "BOX_INDEX"))

    tab_mask_cols = fitsio.read(str(annotated_fits), columns=[in_y1, in_y5, r_mag, box_col])
    base_mask = cutsky_desi_bgs_mock_mask(tab_mask_cols)
    base_mask &= tab_mask_cols[box_col] != -1

    expected_n = int(graph_meta.get("n_points", -1))
    n_after = int(np.count_nonzero(base_mask))
    if expected_n > 0 and n_after != expected_n:
        raise ValueError(
            f"Annotated FITS filter count mismatch: after (Y1|Y5 & R_MAG_APP<{R_MAG_APP_BRIGHT_LT:g} "
            f"& {box_col}!=-1) got {n_after:,} "
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
        in_y1,
        in_y5,
        box_col,
        "FILE_NUM",
        "HALO_INDEX",
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
            out_path=fits_out,
        )
        print(f"Wrote wedge targets FITS: {fits_out}")
        return

    missing_bounds = [
        name
        for name in ("ra_min", "ra_max", "dec_min", "dec_max", "z_min", "z_max")
        if getattr(args, name) is None
    ]
    if missing_bounds:
        raise SystemExit(
            "Missing required wedge bounds arguments: "
            + ", ".join(f"--{n.replace('_', '-')}" for n in missing_bounds)
            + "\n(Provide bounds to build the wedge subgraph, or use --targets-only.)"
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
    filtered_rows, ra, dec, zz = _load_filtered_ra_dec_z(
        annotated_fits,
        graph_meta=graph_meta,
        ra_col=args.ra_col,
        dec_col=args.dec_col,
        z_col=args.redshift_col,
    )

    n_full_graph = int(filtered_rows.size)

    bounds = {
        "ra_min": float(args.ra_min),
        "ra_max": float(args.ra_max),
        "dec_min": float(args.dec_min),
        "dec_max": float(args.dec_max),
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
        "ra_col": str(args.ra_col),
        "dec_col": str(args.dec_col),
        "redshift_col": str(args.redshift_col),
        "ra_wrap": bool(args.ra_min > args.ra_max),
    }

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

    global_ids_out = out_dir / f"{args.out_prefix}_global_node_ids.npy"
    np.save(global_ids_out, wedge_global_ids)

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
        in_set=in_wedge,
        new_id=new_id,
        out_path=out_edges,
        chunk=int(args.edge_chunk),
    )
    tets_wedge, vols_wedge = _write_tets_in_mask(
        tet_path,
        vol_path,
        in_set=in_wedge,
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
        out_path=fits_out,
    )

    meta_out = out_dir / f"{args.out_prefix}_metadata.json"
    wedge_meta = dict(graph_meta)
    wedge_meta.update(
        {
            "prefix": args.out_prefix,
            "source": "wedge_subset",
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
    print("Next: run abacus_graph_features_cugraph.py on the new prefix, then build_abacus_sbi_cache.py.")


if __name__ == "__main__":
    main()
