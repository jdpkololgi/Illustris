"""Build Abacus graph artifacts using Gudhi alpha-complex machinery.

Modes:
- delaunay: equivalent to alpha complex with alpha_sq = +inf
- alpha: use user-specified alpha_sq threshold to prune long-boundary simplices
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import fitsio

# Allow workflow script to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config_paths import ABACUS_CARTESIAN_OUTPUT, ABACUS_TWEB_OUTPUT_DIR
from shared.resource_requirements import require_cpu_mpi_slurm


def _tetra_volume(coords: np.ndarray) -> float:
    """Return tetrahedron volume from a (4, 3) coordinate array."""
    v0, v1, v2, v3 = coords
    mat = np.vstack((v0 - v3, v1 - v3, v2 - v3))
    return float(abs(np.linalg.det(mat)) / 6.0)


def _extract_from_simplex_tree(
    simplex_tree,
    points_xyz: np.ndarray,
    alpha_sq: float,
    *,
    node_offset: int = 0,
    progress_every: int = 1_000_000,
) -> tuple[set[tuple[int, int]], list[list[int]], list[float]]:
    """Extract filtered edges + tetrahedra (and volumes) from a simplex tree."""
    edges: set[tuple[int, int]] = set()
    tetrahedra: list[list[int]] = []
    volumes: list[float] = []

    count = 0
    for simplex, filtration in simplex_tree.get_filtration():
        count += 1
        if progress_every > 0 and count % progress_every == 0:
            print(f"  processed {count:,} simplices...")

        if filtration > alpha_sq:
            continue

        if len(simplex) == 2:
            u = int(simplex[0]) + node_offset
            v = int(simplex[1]) + node_offset
            if u > v:
                u, v = v, u
            edges.add((u, v))
        elif len(simplex) == 4:
            tet_local = [int(x) for x in simplex]
            tet_global = [idx + node_offset for idx in tet_local]
            tetrahedra.append(tet_global)
            volumes.append(_tetra_volume(points_xyz[tet_local]))

    return edges, tetrahedra, volumes


def _build_graph_artifacts(
    points_with_flags: np.ndarray,
    *,
    alpha_sq: float,
    split_hemispheres: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build edges, tetrahedra indices, tetrahedra volumes arrays."""
    try:
        import gudhi
    except ImportError as exc:
        raise ImportError(
            "Gudhi is required for Abacus graph construction. "
            "Install `gudhi` in the active environment."
        ) from exc

    if points_with_flags.ndim != 2 or points_with_flags.shape[1] < 3:
        raise ValueError(
            "Expected points array shape (N, 3) or (N, >=4 with hemisphere flag in column 4)."
        )

    xyz = points_with_flags[:, :3].astype(np.float64)

    all_edges: set[tuple[int, int]] = set()
    all_tetrahedra: list[list[int]] = []
    all_volumes: list[float] = []

    if split_hemispheres and points_with_flags.shape[1] >= 4:
        hemisphere_flag = points_with_flags[:, 3].astype(np.int8)
        north_mask = hemisphere_flag == 1
        south_mask = ~north_mask

        north_indices = np.where(north_mask)[0]
        south_indices = np.where(south_mask)[0]
        north_xyz = xyz[north_mask]
        south_xyz = xyz[south_mask]

        print(
            f"Building split hemispheres with Gudhi "
            f"(north={len(north_xyz):,}, south={len(south_xyz):,})"
        )

        # Build per-hemisphere complexes in compact local indexing.
        alpha_n = gudhi.AlphaComplex(points=north_xyz)
        st_n = alpha_n.create_simplex_tree()
        edges_n_local, tets_n_local, vols_n = _extract_from_simplex_tree(
            st_n, north_xyz, alpha_sq
        )

        alpha_s = gudhi.AlphaComplex(points=south_xyz)
        st_s = alpha_s.create_simplex_tree()
        edges_s_local, tets_s_local, vols_s = _extract_from_simplex_tree(
            st_s, south_xyz, alpha_sq
        )

        # Map local indices back to original global point indices.
        for u_local, v_local in edges_n_local:
            u = int(north_indices[u_local])
            v = int(north_indices[v_local])
            if u > v:
                u, v = v, u
            all_edges.add((u, v))
        for u_local, v_local in edges_s_local:
            u = int(south_indices[u_local])
            v = int(south_indices[v_local])
            if u > v:
                u, v = v, u
            all_edges.add((u, v))

        for tet in tets_n_local:
            all_tetrahedra.append([int(north_indices[idx]) for idx in tet])
        for tet in tets_s_local:
            all_tetrahedra.append([int(south_indices[idx]) for idx in tet])

        all_volumes.extend(vols_n)
        all_volumes.extend(vols_s)
    else:
        print(f"Building single alpha complex with Gudhi (N={len(xyz):,})")
        alpha = gudhi.AlphaComplex(points=xyz)
        st = alpha.create_simplex_tree()
        edges, tetrahedra, volumes = _extract_from_simplex_tree(st, xyz, alpha_sq)
        all_edges = edges
        all_tetrahedra = tetrahedra
        all_volumes = volumes

    edges_arr = np.array(sorted(all_edges), dtype=np.int32)
    tetra_arr = np.array(all_tetrahedra, dtype=np.int32) if all_tetrahedra else np.empty((0, 4), dtype=np.int32)
    vol_arr = np.array(all_volumes, dtype=np.float64) if all_volumes else np.empty((0,), dtype=np.float64)
    return edges_arr, tetra_arr, vol_arr


def _default_alpha_sq_from_number_density(
    points_with_flags: np.ndarray,
    boxsize_mpc: float | None,
) -> float:
    """
    Compute alpha_sq using Illustris-style rule:
    alpha = 1.5 * number_density^(-1/3), alpha_sq = alpha^2.
    """
    xyz = points_with_flags[:, :3]
    n_points = len(xyz)
    if n_points == 0:
        raise ValueError("Cannot derive alpha_sq from empty points array.")

    if boxsize_mpc is None:
        spans = xyz.max(axis=0) - xyz.min(axis=0)
        boxsize_mpc = float(np.max(spans))
        print(
            "No --boxsize-mpc provided; inferred boxsize from coordinate spans "
            f"(x={spans[0]:.3f}, y={spans[1]:.3f}, z={spans[2]:.3f}) -> "
            f"boxsize={boxsize_mpc:.3f} Mpc"
        )

    if boxsize_mpc <= 0:
        raise ValueError(f"boxsize_mpc must be positive, got {boxsize_mpc}.")

    number_density = n_points / (boxsize_mpc**3)
    alpha = 1.5 * number_density ** (-1.0 / 3.0)
    alpha_sq = float(alpha**2)
    print(
        "Computed default alpha_sq using Illustris rule: "
        f"N={n_points:,}, boxsize={boxsize_mpc:.3f} Mpc, "
        f"number_density={number_density:.6e}, alpha={alpha:.6f}, alpha_sq={alpha_sq:.6f}"
    )
    return alpha_sq


def _infer_boxsize_from_tweb_npz(tweb_dir: str) -> float | None:
    """
    Infer simulation boxsize from T-Web rank files, mirroring annotate workflow metadata usage.
    """
    pattern = str(Path(tweb_dir) / "abacus_cactus_tweb_rank*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    box_set: set[float] = set()
    for path in files:
        with np.load(path) as d:
            if "boxsize" not in d:
                return None
            box_set.add(float(d["boxsize"]))

    if len(box_set) != 1:
        raise ValueError(
            f"Inconsistent boxsize values across T-Web files in {tweb_dir}: {sorted(box_set)}"
        )
    boxsize = next(iter(box_set))
    print(f"Inferred boxsize={boxsize:.3f} Mpc from T-Web metadata in: {tweb_dir}")
    return boxsize


def _load_points_from_catalog(
    *,
    catalog_path: Path,
    ra_col: str,
    dec_col: str,
    redshift_col: str,
    apply_y1y5_filter: bool,
) -> np.ndarray:
    """
    Load FITS mock catalog and convert RA/DEC/Z to cartesian points + hemisphere flag.

    Uses observed redshift column by default (`Z`) to preserve RSD effects.
    """
    print(f"Loading FITS catalog: {catalog_path}")
    table = fitsio.read(str(catalog_path))
    names_upper = {name.upper(): name for name in table.dtype.names}

    def _resolve_col(name: str) -> str:
        resolved = names_upper.get(name.upper())
        if resolved is None:
            raise KeyError(
                f"Column `{name}` not found in catalog. "
                f"Available columns include: {table.dtype.names[:12]}..."
            )
        return resolved

    ra_name = _resolve_col(ra_col)
    dec_name = _resolve_col(dec_col)
    z_name = _resolve_col(redshift_col)

    if z_name.upper() == "Z_COSMO":
        print(
            "WARNING: using Z_COSMO will remove/ignore RSD information. "
            "Use --redshift-col Z for observed-space graph construction."
        )

    mask = np.ones(len(table), dtype=bool)
    if apply_y1y5_filter:
        in_y1 = names_upper.get("IN_Y1")
        in_y5 = names_upper.get("IN_Y5")
        if in_y1 is not None and in_y5 is not None:
            mask &= (table[in_y1] == 1) | (table[in_y5] == 1)
            print(f"Applied Y1/Y5 filter: kept {mask.sum():,} / {len(mask):,} rows.")
        else:
            print("IN_Y1/IN_Y5 columns not found; skipping Y1/Y5 filtering.")

    ra = table[ra_name][mask]
    dec = table[dec_name][mask]
    z_obs = table[z_name][mask]

    comoving_distance = cosmo.comoving_distance(z_obs).to(u.Mpc)
    sky_icrs = SkyCoord(
        ra=ra,
        dec=dec,
        unit=(u.deg, u.deg),
        distance=comoving_distance,
        frame="icrs",
    )
    sky_gal = sky_icrs.galactic
    north_flag = (sky_gal.b.deg > 0).astype(np.int8)

    cart = sky_icrs.cartesian
    x = cart.x.to(u.Mpc).value.astype(np.float64)
    y = cart.y.to(u.Mpc).value.astype(np.float64)
    z = cart.z.to(u.Mpc).value.astype(np.float64)

    points = np.vstack((x, y, z, north_flag)).T
    print(
        f"Constructed cartesian points from RA/DEC/{z_name}: "
        f"shape={points.shape}, north={north_flag.sum():,}, south={(north_flag == 0).sum():,}"
    )
    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Abacus graph artifacts with Gudhi. "
            "`delaunay` mode keeps all simplices (alpha_sq=inf); "
            "`alpha` mode prunes by user-provided alpha_sq."
        )
    )
    parser.add_argument(
        "--points-path",
        default=ABACUS_CARTESIAN_OUTPUT,
        help="Path to Abacus cartesian points .npy (N x 3 or N x 4 with hemisphere flag).",
    )
    parser.add_argument(
        "--catalog-path",
        default="/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb.fits",
        help=(
            "Optional FITS catalog path for direct RA/DEC/Z ingestion. "
            "When set, this takes precedence over --points-path."
        ),
    )
    parser.add_argument("--ra-col", default="RA", help="RA column name in FITS catalog.")
    parser.add_argument("--dec-col", default="DEC", help="DEC column name in FITS catalog.")
    parser.add_argument(
        "--redshift-col",
        default="Z",
        help=(
            "Observed redshift column to use (default: Z, preserves RSD). "
            "Avoid Z_COSMO unless intentionally removing RSD."
        ),
    )
    parser.add_argument(
        "--apply-y1y5-filter",
        action="store_true",
        default=True,
        help="Apply IN_Y1/IN_Y5 == 1 filter when those columns exist (default: true).",
    )
    parser.add_argument(
        "--no-apply-y1y5-filter",
        dest="apply_y1y5_filter",
        action="store_false",
        help="Disable IN_Y1/IN_Y5 filtering.",
    )
    parser.add_argument(
        "--output-dir",
        default="/pscratch/sd/d/dkololgi/abacus/graph_constructions",
        help="Directory to write graph artifacts.",
    )
    parser.add_argument(
        "--mode",
        choices=("delaunay", "alpha"),
        default="alpha",
        help="Graph mode. delaunay == full alpha complex (alpha_sq=inf).",
    )
    parser.add_argument(
        "--alpha-sq",
        type=float,
        default=None,
        help=(
            "Alpha^2 filtration threshold for alpha mode. "
            "If omitted in alpha mode, it is computed from number density "
            "using the Illustris rule: alpha=1.5*n^(-1/3)."
        ),
    )
    parser.add_argument(
        "--boxsize-mpc",
        type=float,
        default=None,
        help=(
            "Box size in Mpc for default alpha_sq derivation. "
            "If omitted, inferred from T-Web metadata first, then coordinate span."
        ),
    )
    parser.add_argument(
        "--tweb-dir",
        default=ABACUS_TWEB_OUTPUT_DIR,
        help=(
            "Directory containing abacus_cactus_tweb_rank*.npz files. "
            "Used for automatic boxsize inference when --boxsize-mpc is not provided."
        ),
    )
    parser.add_argument(
        "--split-hemispheres",
        action="store_true",
        default=True,
        help="Build north/south complexes separately if hemisphere flag is present (default: true).",
    )
    parser.add_argument(
        "--no-split-hemispheres",
        dest="split_hemispheres",
        action="store_false",
        help="Disable hemisphere split and build one global complex.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output file prefix. Defaults to `abacus_delaunay` or `abacus_alpha`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cpu_mpi_slurm("build_abacus_graph.py", min_tasks=1)

    points_path = Path(args.points_path).expanduser().resolve()
    catalog_path = Path(args.catalog_path).expanduser().resolve() if args.catalog_path else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if catalog_path is None and not points_path.exists():
        raise FileNotFoundError(
            f"Points file not found: {points_path}. "
            "Either provide a valid --points-path or use --catalog-path."
        )

    prefix = args.output_prefix or ("abacus_delaunay" if args.mode == "delaunay" else "abacus_alpha")

    if catalog_path is not None:
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog file not found: {catalog_path}")
        points = _load_points_from_catalog(
            catalog_path=catalog_path,
            ra_col=args.ra_col,
            dec_col=args.dec_col,
            redshift_col=args.redshift_col,
            apply_y1y5_filter=args.apply_y1y5_filter,
        )
    else:
        print(f"Loading points from: {points_path}")
        points = np.load(points_path).astype(np.float64)
        print(f"Loaded points shape: {points.shape}")

    if args.mode == "alpha":
        if args.alpha_sq is None:
            inferred_boxsize = args.boxsize_mpc
            if inferred_boxsize is None:
                inferred_boxsize = _infer_boxsize_from_tweb_npz(args.tweb_dir)
            alpha_sq = _default_alpha_sq_from_number_density(points, inferred_boxsize)
        else:
            if args.alpha_sq < 0:
                raise ValueError("--alpha-sq must be non-negative.")
            alpha_sq = float(args.alpha_sq)
    else:
        # Delaunay via full alpha complex.
        alpha_sq = math.inf

    print(f"Mode: {args.mode} (alpha_sq={alpha_sq})")

    edges, tetrahedra, tetra_volumes = _build_graph_artifacts(
        points,
        alpha_sq=alpha_sq,
        split_hemispheres=args.split_hemispheres,
    )

    edges_path = output_dir / f"{prefix}_edges_combined_idx.npy"
    tetra_idx_path = output_dir / f"{prefix}_tetrahedra_idx.npy"
    tetra_vol_path = output_dir / f"{prefix}_tetrahedra_volumes.npy"

    np.save(edges_path, edges)
    np.save(tetra_idx_path, tetrahedra)
    np.save(tetra_vol_path, tetra_volumes)

    print(f"Saved edges: {edges_path} shape={edges.shape}")
    print(f"Saved tetrahedra idx: {tetra_idx_path} shape={tetrahedra.shape}")
    print(f"Saved tetrahedra volumes: {tetra_vol_path} shape={tetra_volumes.shape}")

    if edges.size > 0:
        print(f"Edge index range: [{edges.min()}, {edges.max()}]")


if __name__ == "__main__":
    main()
