#!/usr/bin/env python3
"""Hard unit audit for P3 observer-frame fields.

This audit deliberately uses independent evidence rather than trusting variable
names such as ``cell_mpc``:

1. graph points must reproduce Planck18 comoving distances in Mpc;
2. the alternative h-scaled interpretation must be strongly rejected;
3. the historical full-range U-Net cache must use the same convention;
4. the historical 5 Mpc lattice must map to exactly ``5*h`` Mpc/h, while a
   literal 5 Mpc/h lattice is recorded as a different, coarser experiment;
5. the union-graph radius and T-Web smoothing scales are converted explicitly.

The resulting JSON is a required input to ``FIELD_COMPLETE``.  It is not a
model-accuracy comparison: changing between Mpc and Mpc/h is only a change of
coordinates when *all* lengths are converted consistently.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import astropy.units as u
import fitsio
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18


def _expected_xyz_mpc(rows: np.ndarray) -> np.ndarray:
    sky = SkyCoord(
        ra=np.asarray(rows["RA"], dtype=np.float64) * u.deg,
        dec=np.asarray(rows["DEC"], dtype=np.float64) * u.deg,
        distance=Planck18.comoving_distance(np.asarray(rows["Z"], dtype=np.float64)),
        frame="icrs",
    )
    return np.column_stack(
        [
            sky.cartesian.x.to_value(u.Mpc),
            sky.cartesian.y.to_value(u.Mpc),
            sky.cartesian.z.to_value(u.Mpc),
        ]
    )


def _error_summary(observed: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.asarray(observed, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    radial_observed = np.linalg.norm(observed, axis=1)
    radial_expected = np.linalg.norm(expected, axis=1)
    return {
        "max_abs_component": float(np.max(np.abs(delta))),
        "median_abs_component": float(np.median(np.abs(delta))),
        "max_abs_radius": float(np.max(np.abs(radial_observed - radial_expected))),
        "median_radius_ratio": float(np.median(radial_observed / radial_expected)),
    }


def _grid_shape(xyz: np.ndarray, cell: float, padding: float) -> list[int]:
    lo = np.min(xyz, axis=0) - padding
    hi = np.max(xyz, axis=0) + padding
    return np.ceil((hi - lo) / cell).astype(np.int64).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parent",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/"
            "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_"
            "ngrid2048_thr0p2_halo_xcom.fits"
        ),
    )
    ap.add_argument(
        "--points",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
            "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"
        ),
    )
    ap.add_argument(
        "--historical-cache",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/sbi_caches/s3c_cnn_fullrange/"
            "cnn_fullrange_cache.pkl"
        ),
    )
    ap.add_argument(
        "--historical-points",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/sbi_caches/s3c_cnn_fullrange/"
            "cnn_fullrange_points.npy"
        ),
    )
    ap.add_argument(
        "--historical-scores",
        type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/C_unet_fullrange/scores.json"),
    )
    ap.add_argument("--sample-size", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--graph-radius-mpc", type=float, default=14.78)
    ap.add_argument("--target-smoothing-mpch", type=float, default=7.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    h = float(Planck18.h)
    points = np.load(args.points, mmap_mode="r")
    with fitsio.FITS(str(args.parent)) as fits:
        n_parent = int(fits[1].get_nrows())
    if points.shape[0] != n_parent or points.shape[1] < 3:
        raise RuntimeError(f"parent/points mismatch: {n_parent} versus {points.shape}")

    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(n_parent, size=min(args.sample_size, n_parent), replace=False))
    rows = fitsio.read(str(args.parent), columns=["RA", "DEC", "Z"], rows=sample)
    expected_mpc = _expected_xyz_mpc(rows)
    graph_xyz = np.asarray(points[sample, :3], dtype=np.float64)
    graph_as_mpc = _error_summary(graph_xyz, expected_mpc)
    graph_as_mpc_h = _error_summary(graph_xyz, expected_mpc * h)

    historical = pickle.loads(args.historical_cache.read_bytes())
    hist_z = np.asarray(historical["z"], dtype=np.float64)
    hist_points = np.load(args.historical_points, mmap_mode="r")
    if len(hist_z) != len(hist_points):
        raise RuntimeError("historical cache/points row mismatch")
    hs = np.sort(rng.choice(len(hist_z), size=min(args.sample_size, len(hist_z)), replace=False))
    hist_expected_radius = Planck18.comoving_distance(hist_z[hs]).to_value(u.Mpc)
    hist_observed_radius = np.linalg.norm(np.asarray(hist_points[hs], dtype=np.float64), axis=1)
    hist_radius_mpc = {
        "max_abs_radius": float(np.max(np.abs(hist_observed_radius - hist_expected_radius))),
        "median_radius_ratio": float(np.median(hist_observed_radius / hist_expected_radius)),
    }
    hist_radius_mpc_h = {
        "max_abs_radius": float(np.max(np.abs(hist_observed_radius - hist_expected_radius * h))),
        "median_radius_ratio": float(np.median(hist_observed_radius / (hist_expected_radius * h))),
    }

    scores = json.loads(args.historical_scores.read_text())
    historical_cell_mpc = float(scores["cell_mpc"])
    historical_padding_mpc = float(scores["pad_mpc"])
    literal_5_mpc_h_in_mpc = 5.0 / h
    equivalent = {
        "historical_cell_mpc": historical_cell_mpc,
        "historical_cell_mpc_h": historical_cell_mpc * h,
        "historical_padding_mpc": historical_padding_mpc,
        "historical_padding_mpc_h": historical_padding_mpc * h,
        "literal_5_mpc_h_cell_in_mpc": literal_5_mpc_h_in_mpc,
        "literal_5_mpc_h_is_historical_cell": bool(
            np.isclose(literal_5_mpc_h_in_mpc, historical_cell_mpc)
        ),
        "historical_grid_shape": list(scores["grid_shape"]),
        "historical_points_grid_shape_at_historical_cell": _grid_shape(
            np.asarray(hist_points), historical_cell_mpc, historical_padding_mpc
        ),
        "historical_points_grid_shape_at_literal_5_mpc_h": _grid_shape(
            np.asarray(hist_points), literal_5_mpc_h_in_mpc, historical_padding_mpc
        ),
    }

    scales = {
        "planck18_h": h,
        "union_radius_mpc": float(args.graph_radius_mpc),
        "union_radius_mpc_h": float(args.graph_radius_mpc * h),
        "target_smoothing_mpc_h": float(args.target_smoothing_mpch),
        "target_smoothing_mpc": float(args.target_smoothing_mpch / h),
    }
    gates = {
        "graph_points_match_planck18_mpc": graph_as_mpc["max_abs_component"] < 1.0e-5,
        "graph_points_reject_h_scaled_interpretation": graph_as_mpc_h["max_abs_component"] > 100.0,
        "historical_unet_points_match_planck18_mpc": hist_radius_mpc["max_abs_radius"] < 1.0e-4,
        "historical_unet_points_reject_h_scaled_interpretation": hist_radius_mpc_h[
            "max_abs_radius"
        ] > 100.0,
        "historical_scores_explicitly_use_5_mpc": np.isclose(historical_cell_mpc, 5.0),
        "historical_grid_shape_reproduced": equivalent[
            "historical_points_grid_shape_at_historical_cell"
        ] == equivalent["historical_grid_shape"],
        "literal_5_mpc_h_is_distinct_resolution": not equivalent[
            "literal_5_mpc_h_is_historical_cell"
        ],
        "union_radius_is_approximately_10_mpc_h": abs(scales["union_radius_mpc_h"] - 10.0)
        < 0.02,
    }
    payload = {
        "schema_version": 1,
        "decision": (
            "Store observer-frame P3 lattices in comoving Mpc and retain the historical "
            "5 Mpc cell for the first matched model comparison. Express every physical "
            "scale in both Mpc and Mpc/h. A literal 5 Mpc/h cell is a separate coarser "
            "resolution ablation, not the historical U-Net representation."
        ),
        "graph_points": {
            "path": str(args.points),
            "sample_size": int(len(sample)),
            "interpreted_as_mpc": graph_as_mpc,
            "interpreted_as_mpc_h": graph_as_mpc_h,
        },
        "historical_unet": {
            "cache": str(args.historical_cache),
            "points": str(args.historical_points),
            "scores": str(args.historical_scores),
            "sample_size": int(len(hs)),
            "interpreted_as_mpc": hist_radius_mpc,
            "interpreted_as_mpc_h": hist_radius_mpc_h,
            "resolution": equivalent,
        },
        "physical_scales": scales,
        "gates": gates,
        "pass": bool(all(gates.values())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj)) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj)))
    if not payload["pass"]:
        raise RuntimeError("P3 unit audit failed")


if __name__ == "__main__":
    main()
