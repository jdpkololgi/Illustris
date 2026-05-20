#!/usr/bin/env python3
"""Subset an Abacus graph to a user-defined cube and prepare SBI-ready inputs.

This script:
- Loads an existing graph construction (points/edges/tetrahedra) from a graph metadata JSON
  (e.g. produced by `build_abacus_graph.py`).
- Applies a cube cut in **Mpc/h** on the stored comoving xyz (which are in Mpc); it converts via
  `xyz_mpc_h = xyz_mpc * h` where `h` defaults to Planck18.h.
- Writes a new *induced* subgraph with renumbered node ids:
    <out-prefix>_points.npy
    <out-prefix>_points_xyz.npy
    <out-prefix>_edges_combined_idx.npy
    <out-prefix>_tetrahedra_idx.npy
    <out-prefix>_tetrahedra_volumes.npy
    <out-prefix>_metadata.json
    <out-prefix>_global_node_ids.npy   (mapping from local->original global node id)
- Writes a cube-only annotated FITS (rows aligned to the graph node order) for eigenvalue targets.

After running this script, you typically run:
  1) `workflows/abacus_tweb/abacus_graph_features_cugraph.py` on the new prefix to get
     `<out-prefix>_cugraph_gnn_metadata.json`
  2) `workflows/abacus_tweb/build_abacus_sbi_cache.py` using that gnn metadata + the cube FITS
     to produce the pickle consumed by `workflows/sbi/jraph_sbi_flowjax.py`.
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
        help="Annotated CutSky FITS that contains LAMBDA1/2/3 (and optionally CWEB).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/graph_constructions"),
        help="Directory to write cube graph artifacts + cube FITS.",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Prefix for cube outputs (e.g. abacus_delaunay_cube_300mpch_test1).",
    )
    # Bounds are only needed when building the cube subgraph. In --targets-only mode we reuse an
    # existing <out-prefix>_global_node_ids.npy mapping so bounds are not required.
    p.add_argument("--x-min", type=float, default=None, help="Cube min x in Mpc/h.")
    p.add_argument("--x-max", type=float, default=None, help="Cube max x in Mpc/h.")
    p.add_argument("--y-min", type=float, default=None, help="Cube min y in Mpc/h.")
    p.add_argument("--y-max", type=float, default=None, help="Cube max y in Mpc/h.")
    p.add_argument("--z-min", type=float, default=None, help="Cube min z in Mpc/h.")
    p.add_argument("--z-max", type=float, default=None, help="Cube max z in Mpc/h.")
    p.add_argument(
        "--h",
        type=float,
        default=None,
        help="Hubble parameter used for converting stored Mpc -> Mpc/h via xyz*h. Defaults to Planck18.h.",
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
        help="Optional explicit output FITS filename. Default: <out-prefix>_cube_targets.fits",
    )
    p.add_argument(
        "--targets-only",
        action="store_true",
        help=(
            "Only write the cube targets FITS using an existing <out-prefix>_global_node_ids.npy "
            "in --out-dir. Does not rewrite cube graph arrays/metadata."
        ),
    )
    return p.parse_args()


def _planck18_h_default() -> float:
    # Avoid importing astropy unless necessary (this script is primarily I/O bound).
    try:
        from astropy.cosmology import Planck18 as _cosmo  # type: ignore

        return float(_cosmo.h)
    except Exception:
        # Fall back to a commonly used Planck value if astropy is unavailable.
        return 0.6766


def _load_graph_meta(path: Path) -> dict:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return json.load(f)


def _mask_cube_mpc_h(xyz_mpc: np.ndarray, *, h: float, bounds: dict) -> np.ndarray:
    xyz_h = xyz_mpc * float(h)
    x, y, z = xyz_h[:, 0], xyz_h[:, 1], xyz_h[:, 2]
    return (
        (x >= bounds["x_min"])
        & (x <= bounds["x_max"])
        & (y >= bounds["y_min"])
        & (y <= bounds["y_max"])
        & (z >= bounds["z_min"])
        & (z <= bounds["z_max"])
    )


def _count_edges_in_cube(edges_path: Path, in_cube: np.ndarray, chunk: int) -> int:
    edges = np.load(edges_path, mmap_mode="r")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must be (E,2); got {edges.shape} from {edges_path}")
    total = 0
    for i in range(0, edges.shape[0], chunk):
        e = edges[i : i + chunk]
        m = in_cube[e[:, 0]] & in_cube[e[:, 1]]
        total += int(np.count_nonzero(m))
    return total


def _write_edges_in_cube(
    edges_in_path: Path,
    *,
    in_cube: np.ndarray,
    new_id: np.ndarray,
    out_path: Path,
    chunk: int,
) -> np.ndarray:
    edges = np.load(edges_in_path, mmap_mode="r")
    n_keep = _count_edges_in_cube(edges_in_path, in_cube, chunk)
    out = np.empty((n_keep, 2), dtype=np.int32)
    w = 0
    for i in range(0, edges.shape[0], chunk):
        e = edges[i : i + chunk]
        m = in_cube[e[:, 0]] & in_cube[e[:, 1]]
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


def _count_tets_in_cube(tet_path: Path, in_cube: np.ndarray, chunk: int) -> int:
    tets = np.load(tet_path, mmap_mode="r")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tetrahedra must be (T,4); got {tets.shape} from {tet_path}")
    total = 0
    for i in range(0, tets.shape[0], chunk):
        t = tets[i : i + chunk]
        m = in_cube[t[:, 0]] & in_cube[t[:, 1]] & in_cube[t[:, 2]] & in_cube[t[:, 3]]
        total += int(np.count_nonzero(m))
    return total


def _write_tets_in_cube(
    tet_in_path: Path,
    vol_in_path: Path,
    *,
    in_cube: np.ndarray,
    new_id: np.ndarray,
    tet_out_path: Path,
    vol_out_path: Path,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    tets = np.load(tet_in_path, mmap_mode="r")
    vols = np.load(vol_in_path, mmap_mode="r")
    if vols.ndim != 1 or vols.shape[0] != tets.shape[0]:
        raise ValueError("tetrahedra volumes must be (T,) and match tetrahedra rows")
    n_keep = _count_tets_in_cube(tet_in_path, in_cube, chunk)
    out_t = np.empty((n_keep, 4), dtype=np.int32)
    out_v = np.empty((n_keep,), dtype=np.float64)
    w = 0
    for i in range(0, tets.shape[0], chunk):
        t = tets[i : i + chunk]
        v = vols[i : i + chunk]
        m = in_cube[t[:, 0]] & in_cube[t[:, 1]] & in_cube[t[:, 2]] & in_cube[t[:, 3]]
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


def _fits_cols_available(fits_path: Path) -> list[str]:
    with fitsio.FITS(str(fits_path)) as f:
        return [str(x) for x in f[1].get_colnames()]


def _resolve_col(colnames: list[str], candidates: tuple[str, ...]) -> str:
    m = {c.upper(): c for c in colnames}
    for cand in candidates:
        r = m.get(cand.upper())
        if r is not None:
            return r
    raise KeyError(f"None of {candidates} found in FITS columns.")


def _write_cube_targets_fits(
    *,
    annotated_fits: Path,
    graph_meta: dict,
    cube_global_ids: np.ndarray,
    out_path: Path,
) -> None:
    cols = _fits_cols_available(annotated_fits)

    in_y1 = _resolve_col(cols, ("IN_Y1",))
    in_y5 = _resolve_col(cols, ("IN_Y5",))
    r_mag = _resolve_col(cols, ("R_MAG_APP",))
    box_col = _resolve_col(cols, (str(graph_meta.get("catalog_filters", {}).get("box_index_col", "BOX_INDEX")), "BOX_INDEX"))

    # Load only the filtering columns to reproduce graph ordering.
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
    keep_rows = filtered_rows[cube_global_ids.astype(np.int64)]

    # Keep a compact set of useful columns (only those that exist).
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
        # Optional 15-d target derivatives (already dimensionless if present).
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

    # Targets-only mode: reuse an existing cube mapping and only write a new cube targets FITS.
    if args.targets_only:
        global_ids_path = out_dir / f"{args.out_prefix}_global_node_ids.npy"
        if not global_ids_path.exists():
            raise FileNotFoundError(
                f"targets-only mode requires existing global ids mapping at: {global_ids_path}"
            )
        cube_global_ids = np.load(global_ids_path).astype(np.int64, copy=False)
        fits_out = out_dir / (args.fits_out_name or f"{args.out_prefix}_cube_targets.fits")
        _write_cube_targets_fits(
            annotated_fits=args.annotated_fits.expanduser().resolve(),
            graph_meta=graph_meta,
            cube_global_ids=cube_global_ids,
            out_path=fits_out,
        )
        print(f"Wrote cube targets FITS: {fits_out}")
        return

    missing_bounds = [
        name
        for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        if getattr(args, name) is None
    ]
    if missing_bounds:
        raise SystemExit(
            "Missing required cube bounds arguments: "
            + ", ".join(f"--{n.replace('_', '-')}" for n in missing_bounds)
            + "\n(Provide bounds to build the cube subgraph, or use --targets-only.)"
        )

    points_path = base_dir / files.get("points", f"{graph_meta.get('prefix')}_points.npy")
    points_xyz_path = base_dir / files.get("points_xyz", f"{graph_meta.get('prefix')}_points_xyz.npy")
    edges_path = base_dir / files.get("edges", f"{graph_meta.get('prefix')}_edges_combined_idx.npy")
    tet_path = base_dir / files.get("tetrahedra_idx", f"{graph_meta.get('prefix')}_tetrahedra_idx.npy")
    vol_path = base_dir / files.get("tetrahedra_volumes", f"{graph_meta.get('prefix')}_tetrahedra_volumes.npy")

    for p in (points_path, points_xyz_path, edges_path, tet_path, vol_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    h = float(args.h) if args.h is not None else _planck18_h_default()
    bounds = {
        "x_min": float(args.x_min),
        "x_max": float(args.x_max),
        "y_min": float(args.y_min),
        "y_max": float(args.y_max),
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
    }

    xyz = np.load(points_xyz_path, mmap_mode="r")
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"points_xyz must be (N,3); got {xyz.shape}")

    in_cube = _mask_cube_mpc_h(xyz[:, :3], h=h, bounds=bounds)
    cube_global_ids = np.nonzero(in_cube)[0].astype(np.int64)
    if cube_global_ids.size == 0:
        raise ValueError("Cube mask selected zero nodes. Check bounds and h convention.")

    # local->global mapping for downstream joins / debugging.
    global_ids_out = out_dir / f"{args.out_prefix}_global_node_ids.npy"
    np.save(global_ids_out, cube_global_ids)

    # Build global->local map.
    new_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
    new_id[cube_global_ids] = np.arange(cube_global_ids.size, dtype=np.int32)

    # Subset points (Nx4) and xyz.
    points = np.load(points_path, mmap_mode="r")
    cube_points = np.asarray(points[cube_global_ids], dtype=np.float64)
    cube_xyz = np.asarray(xyz[cube_global_ids, :3], dtype=np.float64)

    out_points = out_dir / f"{args.out_prefix}_points.npy"
    out_xyz = out_dir / f"{args.out_prefix}_points_xyz.npy"
    np.save(out_points, cube_points)
    np.save(out_xyz, cube_xyz)

    # Induced edges + tetrahedra.
    out_edges = out_dir / f"{args.out_prefix}_edges_combined_idx.npy"
    out_tets = out_dir / f"{args.out_prefix}_tetrahedra_idx.npy"
    out_vols = out_dir / f"{args.out_prefix}_tetrahedra_volumes.npy"

    edges_cube = _write_edges_in_cube(
        edges_path,
        in_cube=in_cube,
        new_id=new_id,
        out_path=out_edges,
        chunk=int(args.edge_chunk),
    )
    tets_cube, vols_cube = _write_tets_in_cube(
        tet_path,
        vol_path,
        in_cube=in_cube,
        new_id=new_id,
        tet_out_path=out_tets,
        vol_out_path=out_vols,
        chunk=int(args.tet_chunk),
    )

    # Cube targets FITS aligned to node order.
    fits_out = out_dir / (args.fits_out_name or f"{args.out_prefix}_cube_targets.fits")
    _write_cube_targets_fits(
        annotated_fits=args.annotated_fits.expanduser().resolve(),
        graph_meta=graph_meta,
        cube_global_ids=cube_global_ids,
        out_path=fits_out,
    )

    # Metadata for cube graph.
    meta_out = out_dir / f"{args.out_prefix}_metadata.json"
    cube_meta = dict(graph_meta)
    cube_meta.update(
        {
            "prefix": args.out_prefix,
            "source": "cube_subset",
            "cube_bounds_mpc_h": bounds,
            "cube_h_used": h,
            "parent_graph_metadata": str(graph_meta_path),
            "n_points": int(cube_points.shape[0]),
            "n_point_columns": int(cube_points.shape[1]),
            "n_edges": int(edges_cube.shape[0]),
            "n_tetrahedra": int(tets_cube.shape[0]),
            "files": {
                "points": out_points.name,
                "points_xyz": out_xyz.name,
                "edges": out_edges.name,
                "tetrahedra_idx": out_tets.name,
                "tetrahedra_volumes": out_vols.name,
                "global_node_ids": global_ids_out.name,
                "cube_targets_fits": fits_out.name,
            },
        }
    )
    with meta_out.open("w", encoding="utf-8") as f:
        json.dump(cube_meta, f, indent=2, sort_keys=True)

    print(f"Cube nodes: {cube_points.shape[0]:,} / {xyz.shape[0]:,}")
    print(f"Cube edges: {edges_cube.shape[0]:,}")
    print(f"Cube tetrahedra: {tets_cube.shape[0]:,}")
    print(f"Wrote cube graph metadata: {meta_out}")
    print(f"Wrote cube targets FITS: {fits_out}")
    print("Next: run abacus_graph_features_cugraph.py on the new prefix, then build_abacus_sbi_cache.py.")


if __name__ == "__main__":
    main()

