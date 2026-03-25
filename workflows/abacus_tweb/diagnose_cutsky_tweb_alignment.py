#!/usr/bin/env python3
"""Diagnose potential mismatches between CutSky galaxy positions and T-Web labels.

This script audits the mapping used by `annotate_cutsky_with_tweb.py` by:
1) Recomputing slab/grid lookup and eigenvalue assignment for sampled galaxies.
2) Comparing recomputed labels against catalog LAMBDA1/2/3 and CWEB.
3) Testing alternative coordinate assumptions (z-column and observer origin).
4) Checking mismatch concentration near slab boundaries.
5) Measuring local spatial continuity of eigenvalues.

Outputs:
- alignment_report.json
- concise terminal summary
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitsio
import numpy as np
from astropy.cosmology import Planck18 as cosmo


DEFAULT_CATALOG = (
    "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/"
    "cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb.fits"
)
DEFAULT_TWEB_DIR = "/pscratch/sd/d/dkololgi/abacus/tweb_output/slabs"
DEFAULT_OUT_DIR = "/pscratch/sd/d/dkololgi/abacus/alignment_diagnostics"


@dataclass(frozen=True)
class SlabMeta:
    slab_id: int
    path: str
    x_start: int
    x_end: int
    ngrid: int
    boxsize: float
    threshold: float
    rsmooth: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog-path", default=DEFAULT_CATALOG, help="Annotated FITS with CWEB/LAMBDA columns.")
    p.add_argument("--tweb-dir", default=DEFAULT_TWEB_DIR, help="Directory with abacus_cactus_tweb_rank*.npz.")
    p.add_argument("--output-dir", default=DEFAULT_OUT_DIR, help="Directory to write JSON report.")
    p.add_argument("--sample-size", type=int, default=200_000, help="Random catalog rows to audit.")
    p.add_argument("--continuity-size", type=int, default=20_000, help="Rows for local continuity check.")
    p.add_argument("--neighbors", type=int, default=8, help="k for kNN continuity diagnostics.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def discover_slabs(tweb_dir: str) -> list[SlabMeta]:
    pattern = str(Path(tweb_dir) / "abacus_cactus_tweb_rank*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No slab files found: {pattern}")

    slabs: list[SlabMeta] = []
    for i, path in enumerate(files):
        with np.load(path) as d:
            slabs.append(
                SlabMeta(
                    slab_id=i,
                    path=path,
                    x_start=int(d["x_start"]),
                    x_end=int(d["x_end"]),
                    ngrid=int(d["ngrid"]),
                    boxsize=float(d["boxsize"]),
                    threshold=float(d["threshold"]),
                    rsmooth=float(d["Rsmooth"]),
                )
            )
    slabs = sorted(slabs, key=lambda s: s.x_start)
    # Renumber by sorted x_start
    return [
        SlabMeta(
            slab_id=i,
            path=s.path,
            x_start=s.x_start,
            x_end=s.x_end,
            ngrid=s.ngrid,
            boxsize=s.boxsize,
            threshold=s.threshold,
            rsmooth=s.rsmooth,
        )
        for i, s in enumerate(slabs)
    ]


def validate_slabs(slabs: list[SlabMeta]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ngrid_set = {s.ngrid for s in slabs}
    box_set = {s.boxsize for s in slabs}
    thr_set = {s.threshold for s in slabs}
    rsm_set = {s.rsmooth for s in slabs}
    if len(ngrid_set) != 1 or len(box_set) != 1:
        raise ValueError("Inconsistent ngrid/boxsize across slab files.")
    ngrid = next(iter(ngrid_set))
    ix_to_slab = np.full(ngrid, -1, dtype=np.int16)
    slab_xstart = np.full(len(slabs), -1, dtype=np.int32)

    expected = 0
    for s in slabs:
        if s.x_start != expected:
            raise ValueError(f"Slab gap/overlap at x={expected}, next slab starts {s.x_start}")
        if s.x_end <= s.x_start:
            raise ValueError(f"Invalid slab range [{s.x_start}, {s.x_end}) in {s.path}")
        ix_to_slab[s.x_start : s.x_end] = s.slab_id
        slab_xstart[s.slab_id] = s.x_start
        expected = s.x_end
    if expected != ngrid or np.any(ix_to_slab < 0):
        raise ValueError("Slab coverage is incomplete.")

    meta = {
        "n_slabs": len(slabs),
        "ngrid": int(ngrid),
        "boxsize": float(next(iter(box_set))),
        "threshold": float(next(iter(thr_set))),
        "rsmooth": float(next(iter(rsm_set))),
    }
    return ix_to_slab, slab_xstart, meta


def sky_to_box_coords(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    zvals: np.ndarray,
    boxsize: float,
    observer_origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Match annotation convention: Planck18 comoving_distance * h.
    dist = cosmo.comoving_distance(zvals).value * cosmo.h
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x_obs = dist * np.cos(dec) * np.cos(ra)
    y_obs = dist * np.cos(dec) * np.sin(ra)
    z_obs = dist * np.sin(dec)
    x = (x_obs + observer_origin[0]) % boxsize
    y = (y_obs + observer_origin[1]) % boxsize
    z = (z_obs + observer_origin[2]) % boxsize
    return x, y, z


def to_grid_indices(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ngrid: int,
    boxsize: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = boxsize / ngrid
    ix = np.floor(x / cell).astype(np.int32)
    iy = np.floor(y / cell).astype(np.int32)
    iz = np.floor(z / cell).astype(np.int32)
    np.clip(ix, 0, ngrid - 1, out=ix)
    np.clip(iy, 0, ngrid - 1, out=iy)
    np.clip(iz, 0, ngrid - 1, out=iz)
    return ix, iy, iz


def gather_tweb_values(
    slabs: list[SlabMeta],
    ix_to_slab: np.ndarray,
    slab_xstart: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    slab_ids = ix_to_slab[ix]
    local_ix = ix - slab_xstart[slab_ids]
    cweb = np.empty(ix.shape[0], dtype=np.uint8)
    eig = np.empty((ix.shape[0], 3), dtype=np.float32)
    total_rows = ix.shape[0]
    for si, s in enumerate(slabs):
        rows = np.nonzero(slab_ids == s.slab_id)[0]
        if rows.size == 0:
            continue
        if si % 4 == 0:
            print(
                f"    slab {si+1:02d}/{len(slabs)} | rows in slab: {rows.size:,} / {total_rows:,}",
                flush=True,
            )
        with np.load(s.path) as d:
            c_local = d["cweb"]
            e_local = d["eig_vals"]
            li = local_ix[rows].astype(np.int64)
            yj = iy[rows].astype(np.int64)
            zk = iz[rows].astype(np.int64)
            cweb[rows] = c_local[li, yj, zk]
            eig[rows, 0] = e_local[0, li, yj, zk]
            eig[rows, 1] = e_local[1, li, yj, zk]
            eig[rows, 2] = e_local[2, li, yj, zk]
    return cweb, eig


def eval_variant(
    name: str,
    zvals: np.ndarray,
    observer_origin: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    lambdas_catalog: np.ndarray,
    cweb_catalog: np.ndarray,
    slabs: list[SlabMeta],
    ix_to_slab: np.ndarray,
    slab_xstart: np.ndarray,
    ngrid: int,
    boxsize: float,
) -> dict[str, Any]:
    x, y, z = sky_to_box_coords(ra, dec, zvals, boxsize=boxsize, observer_origin=observer_origin)
    ix, iy, iz = to_grid_indices(x, y, z, ngrid=ngrid, boxsize=boxsize)
    cweb_re, eig_re = gather_tweb_values(slabs, ix_to_slab, slab_xstart, ix, iy, iz)

    diff = eig_re - lambdas_catalog
    mae = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    cweb_match = float(np.mean(cweb_re == cweb_catalog))

    # Boundary sensitivity: galaxies near slab x boundaries should not have disproportionate mismatch.
    # "Near" means within one x-cell of any slab boundary.
    boundaries = np.array([s.x_start for s in slabs] + [slabs[-1].x_end], dtype=np.int32)
    d_to_b = np.min(np.abs(ix[:, None] - boundaries[None, :]), axis=1)
    near = d_to_b <= 1
    far = ~near
    mae_near = float(np.mean(np.abs(diff[near]))) if np.any(near) else float("nan")
    mae_far = float(np.mean(np.abs(diff[far]))) if np.any(far) else float("nan")

    return {
        "name": name,
        "observer_origin": observer_origin.tolist(),
        "mae_lambda": [float(x) for x in mae],
        "rmse_lambda": [float(x) for x in rmse],
        "mae_lambda_mean": float(np.mean(mae)),
        "rmse_lambda_mean": float(np.mean(rmse)),
        "cweb_match_fraction": cweb_match,
        "boundary_mae_near_cells_1": mae_near,
        "boundary_mae_far_cells_1": mae_far,
        "n_near_boundary": int(np.sum(near)),
        "n_far_boundary": int(np.sum(far)),
        "coords_preview": {
            "x_minmax": [float(np.min(x)), float(np.max(x))],
            "y_minmax": [float(np.min(y)), float(np.max(y))],
            "z_minmax": [float(np.min(z)), float(np.max(z))],
        },
        "_xyz": (x, y, z),  # internal use for continuity
    }


def local_continuity_metric(
    xyz: np.ndarray,
    eig: np.ndarray,
    *,
    k: int,
    seed: int,
    sample_size: int,
) -> dict[str, float]:
    # kNN continuity using sklearn if available; fallback to chunked brute-force.
    n = xyz.shape[0]
    rng = np.random.default_rng(seed)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        xyz = xyz[idx]
        eig = eig[idx]
        n = sample_size

    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto")
        nn.fit(xyz)
        neigh_idx = nn.kneighbors(return_distance=False)[:, 1:]
    except Exception:
        # Fallback: brute-force for limited n.
        m = min(n, 5000)
        xyz = xyz[:m]
        eig = eig[:m]
        n = m
        d2 = np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        neigh_idx = np.argpartition(d2, kth=min(k, n - 1), axis=1)[:, : min(k, n - 1)]

    neigh_diff = np.mean(np.abs(eig[:, None, :] - eig[neigh_idx, :]), axis=(1, 2))
    neigh_mean = float(np.mean(neigh_diff))

    # Random-pair baseline
    j = rng.integers(0, n, size=n)
    rand_diff = np.mean(np.abs(eig - eig[j]), axis=1)
    rand_mean = float(np.mean(rand_diff))
    ratio = float(neigh_mean / max(rand_mean, 1e-12))
    return {
        "neighbor_absdiff_mean": neigh_mean,
        "random_absdiff_mean": rand_mean,
        "continuity_ratio_neighbor_over_random": ratio,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("[1/6] Discovering slabs...", flush=True)
    slabs = discover_slabs(args.tweb_dir)
    ix_to_slab, slab_xstart, slab_meta = validate_slabs(slabs)
    print(
        f"  found {slab_meta['n_slabs']} slabs | ngrid={slab_meta['ngrid']} | boxsize={slab_meta['boxsize']}",
        flush=True,
    )

    print("[2/6] Reading sampled rows from catalog...", flush=True)
    cat = Path(args.catalog_path).expanduser().resolve()
    with fitsio.FITS(str(cat), "r") as f:
        hdu = f[1]
        nrows = int(hdu.get_nrows())
        take = min(int(args.sample_size), nrows)
        idx = np.sort(rng.choice(nrows, size=take, replace=False))
        cols = ["RA", "DEC", "Z_COSMO", "LAMBDA1", "LAMBDA2", "LAMBDA3", "CWEB"]
        # Optional columns
        all_names = {n.upper() for n in hdu.get_colnames()}
        if "Z" in all_names:
            cols.append("Z")
        if "IN_Y1" in all_names:
            cols.append("IN_Y1")
        if "IN_Y5" in all_names:
            cols.append("IN_Y5")
        s = hdu.read(rows=idx, columns=cols)
    print(f"  sampled {take:,} rows from {nrows:,}", flush=True)

    ra = np.asarray(s["RA"], dtype=np.float64)
    dec = np.asarray(s["DEC"], dtype=np.float64)
    z_cosmo = np.asarray(s["Z_COSMO"], dtype=np.float64)
    z_obs = np.asarray(s["Z"], dtype=np.float64) if "Z" in s.dtype.names else None
    lambdas_cat = np.stack(
        [
            np.asarray(s["LAMBDA1"], dtype=np.float32),
            np.asarray(s["LAMBDA2"], dtype=np.float32),
            np.asarray(s["LAMBDA3"], dtype=np.float32),
        ],
        axis=-1,
    )
    cweb_cat = np.asarray(s["CWEB"], dtype=np.uint8)

    # Core coordinate/label variants to diagnose frame mismatch.
    print("[3/6] Evaluating coordinate variants...", flush=True)
    variants: list[dict[str, Any]] = []
    print("  variant: z_cosmo_origin_-990", flush=True)
    variants.append(
        eval_variant(
            "z_cosmo_origin_-990",
            z_cosmo,
            np.array([-990.0, -990.0, -990.0], dtype=np.float64),
            ra,
            dec,
            lambdas_cat,
            cweb_cat,
            slabs,
            ix_to_slab,
            slab_xstart,
            slab_meta["ngrid"],
            slab_meta["boxsize"],
        )
    )
    print("  variant: z_cosmo_origin_0", flush=True)
    variants.append(
        eval_variant(
            "z_cosmo_origin_0",
            z_cosmo,
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            ra,
            dec,
            lambdas_cat,
            cweb_cat,
            slabs,
            ix_to_slab,
            slab_xstart,
            slab_meta["ngrid"],
            slab_meta["boxsize"],
        )
    )
    if z_obs is not None:
        print("  variant: z_obs_origin_-990", flush=True)
        variants.append(
            eval_variant(
                "z_obs_origin_-990",
                z_obs,
                np.array([-990.0, -990.0, -990.0], dtype=np.float64),
                ra,
                dec,
                lambdas_cat,
                cweb_cat,
                slabs,
                ix_to_slab,
                slab_xstart,
                slab_meta["ngrid"],
                slab_meta["boxsize"],
            )
        )
        print("  variant: z_obs_origin_0", flush=True)
        variants.append(
            eval_variant(
                "z_obs_origin_0",
                z_obs,
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
                ra,
                dec,
                lambdas_cat,
                cweb_cat,
                slabs,
                ix_to_slab,
                slab_xstart,
                slab_meta["ngrid"],
                slab_meta["boxsize"],
            )
        )

    # Best variant by mean MAE
    print("[4/6] Selecting best variant and computing continuity...", flush=True)
    best = min(variants, key=lambda v: v["mae_lambda_mean"])
    x_best, y_best, z_best = best["_xyz"]
    continuity = local_continuity_metric(
        np.stack([x_best, y_best, z_best], axis=-1),
        lambdas_cat,
        k=max(1, int(args.neighbors)),
        seed=args.seed + 17,
        sample_size=max(1000, int(args.continuity_size)),
    )

    print("[5/6] Computing filter consistency stats...", flush=True)
    # Split/filter consistency summary if available.
    filter_stats = {}
    if "IN_Y1" in s.dtype.names and "IN_Y5" in s.dtype.names:
        in_y1 = np.asarray(s["IN_Y1"]) == 1
        in_y5 = np.asarray(s["IN_Y5"]) == 1
        keep = in_y1 | in_y5
        filter_stats = {
            "sample_in_y1_fraction": float(np.mean(in_y1)),
            "sample_in_y5_fraction": float(np.mean(in_y5)),
            "sample_keep_y1y5_fraction": float(np.mean(keep)),
        }

    # Remove internal arrays before serializing.
    variants_out = []
    for v in variants:
        c = {k: val for k, val in v.items() if not k.startswith("_")}
        variants_out.append(c)

    report = {
        "catalog_path": str(cat),
        "tweb_dir": str(Path(args.tweb_dir).expanduser().resolve()),
        "sample_size": int(take),
        "seed": int(args.seed),
        "slab_meta": slab_meta,
        "variants": variants_out,
        "best_variant": {k: val for k, val in best.items() if not k.startswith("_")},
        "continuity": continuity,
        "filter_stats": filter_stats,
        "interpretation_hints": {
            "good_reconstruction": "low mae_lambda_mean and high cweb_match_fraction for baseline variant",
            "frame_mismatch": "alternative variant beats baseline by large margin",
            "boundary_issue": "boundary_mae_near significantly larger than boundary_mae_far",
            "poor_spatial_coherence": "continuity_ratio_neighbor_over_random close to 1.0",
        },
    }

    print("[6/6] Writing report...", flush=True)
    out_json = out_dir / "alignment_report.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("=" * 72)
    print("CutSky/T-Web alignment diagnostics complete")
    print("=" * 72)
    print(f"Catalog sample size: {take:,} / {nrows:,}")
    print(f"Slabs: {slab_meta['n_slabs']}, ngrid={slab_meta['ngrid']}, boxsize={slab_meta['boxsize']}")
    print(f"Best variant: {best['name']}")
    print(f"  mean lambda MAE: {best['mae_lambda_mean']:.6f}")
    print(f"  CWEB match frac: {best['cweb_match_fraction']:.6f}")
    print(
        "Continuity ratio (neighbor/random): "
        f"{continuity['continuity_ratio_neighbor_over_random']:.6f}"
    )
    print(f"Report: {out_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()

