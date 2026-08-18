#!/usr/bin/env python3
"""Compare P10 matter-density slices and labelled mock galaxies across phases.

This is a read-only diagnostic for the visible P10 phases.  It deliberately
rejects ``ph001`` because that phase remains the sealed blind test.  The matter
background is a thin projection of the canonical 10-percent particle TSC grid.
Mock galaxies are mapped back to their host-halo ``x_com`` coordinates, reduced
periodically into the same 2000 Mpc/h box, and coloured by their stored true
``LAMBDA1`` label.

The plot therefore compares like with like across phases:

* identical periodic-box coordinates and slice geometry;
* identical R=7 Mpc/h transverse display smoothing;
* shared density and lambda_1 colour scales;
* a deterministic, equally sized catalogue sample per phase.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import fitsio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.validate_cutsky_eigs_boxindex_vs_halo_xcom import (  # noqa: E402
    load_halo_positions_xcom,
)


P10_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
REGISTRY = REPO_ROOT / "configs/p10_phase_registry_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/figures/p10_multiphase_review_20260818"
VISIBLE_PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")
SEALED_PHASE = "ph001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p10-root", type=Path, default=P10_ROOT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phases", nargs="+", default=list(VISIBLE_PHASES))
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-centre-mpc-h", type=float, default=1000.0)
    parser.add_argument("--slice-half-width-mpc-h", type=float, default=25.0)
    parser.add_argument("--display-smoothing-mpc-h", type=float, default=7.0)
    parser.add_argument("--sample-rows", type=int, default=400_000)
    parser.add_argument("--max-points-per-panel", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def density_manifest(root: Path, phase: str) -> tuple[Path, dict]:
    paths = sorted((root / phase / "targets/density").glob("*.manifest.json"))
    if len(paths) != 1:
        raise RuntimeError(f"{phase}: expected one density manifest, found {paths}")
    payload = json.loads(paths[0].read_text())
    output = Path(payload["build"]["output"])
    if not output.is_file():
        raise FileNotFoundError(output)
    return paths[0], payload


def phase_paths(root: Path, registry: dict, phase: str) -> dict[str, Path]:
    manifest_path, density_meta = density_manifest(root, phase)
    p1 = json.loads((root / phase / "p1_canonical/manifest.json").read_text())
    if phase == "ph000":
        snapshot_root = Path(registry["convention_reference"]["restored_snapshot_root"])
    else:
        template = registry["path_templates"]["snapshot_root"]
        snapshot_root = Path(template.format(phase=phase))
    return {
        "density": Path(density_meta["build"]["output"]),
        "density_manifest": manifest_path,
        "catalogue": Path(p1["parent"]),
        "index": Path(p1["index"]),
        "halo_info": snapshot_root / "halo_info",
    }


def deterministic_active_rows(index_path: Path, n: int, seed: int) -> np.ndarray:
    index = np.load(index_path, mmap_mode="r")
    eligible = np.asarray(index["active"], dtype=bool) & np.asarray(
        index["valid_target"], dtype=bool
    )
    rows = np.flatnonzero(eligible)
    if len(rows) > n:
        rows = np.sort(np.random.default_rng(seed).choice(rows, n, replace=False))
    return rows.astype(np.int64, copy=False)


def read_catalogue_sample(path: Path, rows: np.ndarray) -> np.ndarray:
    columns = [
        "FILE_NUM", "HALO_INDEX", "BOX_INDEX",
        "LAMBDA1", "LAMBDA2", "LAMBDA3",
    ]
    with fitsio.FITS(str(path), "r") as handle:
        table = handle[1].read(rows=rows, columns=columns)
    valid = (
        (np.asarray(table["BOX_INDEX"], dtype=np.int64) >= 0)
        & (np.asarray(table["HALO_INDEX"], dtype=np.int64) >= 0)
        & np.isfinite(np.column_stack([table[f"LAMBDA{i}"] for i in (1, 2, 3)])).all(axis=1)
    )
    table = table[valid]
    # Multiple observed galaxies may share one host.  One point per host avoids
    # overplotting the same x_com location and does not change its tidal label.
    keys = np.rec.fromarrays(
        [table["FILE_NUM"].astype(np.int64), table["HALO_INDEX"].astype(np.int64)],
        names=("file_num", "halo_index"),
    )
    _, first = np.unique(keys, return_index=True)
    return table[np.sort(first)]


def projected_density(
    path: Path,
    *,
    ngrid: int,
    boxsize: float,
    axis: int,
    centre: float,
    half_width: float,
    smoothing: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    field = np.load(path, mmap_mode="r")
    if field.shape != (ngrid, ngrid, ngrid):
        raise RuntimeError(f"{path}: unexpected shape {field.shape}")
    cell = boxsize / ngrid
    lo = max(0, int(np.floor((centre - half_width) / cell)))
    hi = min(ngrid, int(np.ceil((centre + half_width) / cell)))
    slices = [slice(None), slice(None), slice(None)]
    slices[axis] = slice(lo, hi)
    slab = np.asarray(field[tuple(slices)], dtype=np.float32)
    surface = np.mean(slab, axis=axis, dtype=np.float64).astype(np.float32)
    surface = gaussian_filter(surface, sigma=smoothing / cell, mode="wrap")
    mean = float(np.mean(surface, dtype=np.float64))
    if not np.isfinite(mean) or mean <= 0:
        raise RuntimeError(f"{path}: invalid projected mean {mean}")
    return np.log10(np.maximum(surface / mean, 1.0e-4)), (lo, hi)


def plane_coordinates(xyz: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    plane = [value for value in range(3) if value != axis]
    return xyz[:, plane[0]], xyz[:, plane[1]]


def axis_labels(axis: int) -> tuple[str, str]:
    labels = ("x", "y", "z")
    plane = [value for value in range(3) if value != axis]
    return labels[plane[0]], labels[plane[1]]


def run_phase(
    phase: str,
    paths: dict[str, Path],
    *,
    ngrid: int,
    boxsize: float,
    axis: int,
    centre: float,
    half_width: float,
    smoothing: float,
    sample_rows: int,
    max_points: int,
    seed: int,
) -> dict:
    rows = deterministic_active_rows(paths["index"], sample_rows, seed)
    table = read_catalogue_sample(paths["catalogue"], rows)
    xyz_native = load_halo_positions_xcom(
        halo_info_dir=paths["halo_info"],
        file_nums=np.asarray(table["FILE_NUM"], dtype=np.int32),
        halo_indices=np.asarray(table["HALO_INDEX"], dtype=np.int64),
    ).astype(np.float64)
    xyz = np.mod(xyz_native, boxsize)
    distance = np.abs(xyz[:, axis] - centre)
    inside = distance <= half_width
    xyz = xyz[inside]
    lambdas = np.column_stack(
        [np.asarray(table[f"LAMBDA{i}"], dtype=np.float64)[inside] for i in (1, 2, 3)]
    )
    if len(xyz) == 0:
        raise RuntimeError(f"{phase}: no sampled host galaxies in requested slice")
    if len(xyz) > max_points:
        chosen = np.sort(np.random.default_rng(seed + 1).choice(
            len(xyz), max_points, replace=False
        ))
        xyz = xyz[chosen]
        lambdas = lambdas[chosen]

    density, slice_indices = projected_density(
        paths["density"], ngrid=ngrid, boxsize=boxsize, axis=axis,
        centre=centre, half_width=half_width, smoothing=smoothing,
    )
    u, v = plane_coordinates(xyz, axis)
    cell = boxsize / ngrid
    iu = np.clip((u / cell).astype(np.int64), 0, ngrid - 1)
    iv = np.clip((v / cell).astype(np.int64), 0, ngrid - 1)
    local_density = density[iu, iv]
    random_uv = np.random.default_rng(seed + 2).uniform(0.0, boxsize, size=(len(xyz), 2))
    ri = np.clip((random_uv[:, 0] / cell).astype(np.int64), 0, ngrid - 1)
    rj = np.clip((random_uv[:, 1] / cell).astype(np.int64), 0, ngrid - 1)
    random_density = density[ri, rj]
    trace = np.sum(lambdas, axis=1)
    rho = float(spearmanr(local_density, trace).statistic) if len(trace) >= 3 else float("nan")

    return {
        "phase": phase,
        "density": density,
        "u": u,
        "v": v,
        "lambda1": lambdas[:, 0],
        "stats": {
            "sampled_active_rows": int(len(rows)),
            "sampled_unique_hosts_before_slice": int(len(table)),
            "plotted_unique_hosts": int(len(xyz)),
            "native_xcom_min_mpc_h": np.min(xyz_native, axis=0).tolist(),
            "native_xcom_max_mpc_h": np.max(xyz_native, axis=0).tolist(),
            "periodic_xcom_min_mpc_h": np.min(xyz, axis=0).tolist(),
            "periodic_xcom_max_mpc_h": np.max(xyz, axis=0).tolist(),
            "slice_grid_indices": list(slice_indices),
            "galaxy_median_log10_projected_density_ratio": float(np.median(local_density)),
            "random_median_log10_projected_density_ratio": float(np.median(random_density)),
            "galaxy_minus_random_median": float(np.median(local_density) - np.median(random_density)),
            "projected_density_trace_spearman": rho,
            "lambda1_percentiles": np.percentile(lambdas[:, 0], [1, 50, 99]).tolist(),
        },
    }


def draw_panel(
    ax: plt.Axes,
    result: dict,
    *,
    boxsize: float,
    density_norm: mpl.colors.Normalize,
    lambda_norm: mpl.colors.Normalize,
    point_size: float,
) -> tuple[mpl.image.AxesImage, mpl.collections.PathCollection]:
    image = ax.imshow(
        result["density"].T,
        origin="lower",
        extent=(0.0, boxsize, 0.0, boxsize),
        cmap="magma",
        norm=density_norm,
        interpolation="nearest",
        rasterized=True,
    )
    points = ax.scatter(
        result["u"], result["v"], c=result["lambda1"], s=point_size,
        cmap="coolwarm", norm=lambda_norm, linewidths=0, alpha=0.82,
        rasterized=True,
    )
    stats = result["stats"]
    ax.set_title(
        f"{result['phase']}  |  N={stats['plotted_unique_hosts']:,}\n"
        rf"$\Delta_{{\rm gal-rand}}={stats['galaxy_minus_random_median']:.2f}$ dex, "
        rf"$\rho(\delta_{{2D}},\mathrm{{Tr}}T)={stats['projected_density_trace_spearman']:.2f}$",
        fontsize=9,
    )
    ax.set_aspect("equal")
    return image, points


def main() -> None:
    args = parse_args()
    phases = tuple(args.phases)
    if SEALED_PHASE in phases:
        raise RuntimeError("ph001 is sealed and may not be plotted or inspected")
    unknown = sorted(set(phases) - set(VISIBLE_PHASES))
    if unknown:
        raise RuntimeError(f"unsupported visible phases: {unknown}")

    registry = json.loads(args.registry.read_text())
    contract = registry["target_contract"]
    ngrid = int(contract["grid_size"])
    boxsize = float(contract["box_size_mpc_h"])
    axis = ("x", "y", "z").index(args.slice_axis)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    results = []
    inputs = {}
    for offset, phase in enumerate(phases):
        paths = phase_paths(args.p10_root, registry, phase)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{phase}: missing inputs {missing}")
        inputs[phase] = {name: str(path) for name, path in paths.items()}
        print(f"[{phase}] loading matched density slice and host galaxies", flush=True)
        results.append(run_phase(
            phase, paths, ngrid=ngrid, boxsize=boxsize, axis=axis,
            centre=args.slice_centre_mpc_h,
            half_width=args.slice_half_width_mpc_h,
            smoothing=args.display_smoothing_mpc_h,
            sample_rows=args.sample_rows,
            max_points=args.max_points_per_panel,
            seed=args.seed + 101 * offset,
        ))

    density_values = np.concatenate([
        result["density"][::8, ::8].ravel() for result in results
    ])
    density_limits = np.percentile(density_values, [1.0, 99.5])
    lambda_values = np.concatenate([result["lambda1"] for result in results])
    lambda_limits = np.percentile(lambda_values, [1.0, 99.0])
    density_norm = mpl.colors.Normalize(*density_limits)
    if lambda_limits[0] < 0.2 < lambda_limits[1]:
        lambda_norm = mpl.colors.TwoSlopeNorm(
            vmin=float(lambda_limits[0]), vcenter=0.2, vmax=float(lambda_limits[1])
        )
    else:
        lambda_norm = mpl.colors.Normalize(*lambda_limits)

    xlabel, ylabel = axis_labels(axis)
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 9.7), constrained_layout=True)
    flat = axes.ravel()
    image = points = None
    for ax, result in zip(flat, results):
        image, points = draw_panel(
            ax, result, boxsize=boxsize, density_norm=density_norm,
            lambda_norm=lambda_norm, point_size=1.2,
        )
        ax.set_xlabel(rf"${xlabel}\ [h^{{-1}}\,\mathrm{{Mpc}}]$")
        ax.set_ylabel(rf"${ylabel}\ [h^{{-1}}\,\mathrm{{Mpc}}]$")
    for ax in flat[len(results):]:
        ax.set_visible(False)
    if image is None or points is None:
        raise RuntimeError("no panels drawn")
    cbar_density = figure.colorbar(image, ax=flat.tolist(), shrink=0.82, pad=0.02)
    cbar_density.set_label(r"matter density: $\log_{10}(\Sigma/\langle\Sigma\rangle)$")
    cbar_lambda = figure.colorbar(points, ax=flat.tolist(), shrink=0.82, pad=0.06)
    cbar_lambda.set_label(r"mock galaxy true $\lambda_1$ (point colour; centre at $\lambda_{\rm th}=0.2$)")
    figure.suptitle(
        f"P10 visible Abacus phases: {args.slice_axis}={args.slice_centre_mpc_h:g} "
        f"$\\pm$ {args.slice_half_width_mpc_h:g} "
        f"$h^{{-1}}\\,\\mathrm{{Mpc}}$\n"
        f"10% particle TSC density; transverse Gaussian display smoothing "
        f"$R={args.display_smoothing_mpc_h:g}\\,h^{{-1}}\\mathrm{{Mpc}}$; "
        f"host-halo $x_{{\\rm com}}$ overlays",
        fontsize=12,
    )
    comparison_png = output / "phase_density_galaxies_comparison.png"
    comparison_pdf = output / "phase_density_galaxies_comparison.pdf"
    figure.savefig(comparison_png, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(comparison_pdf, bbox_inches="tight")
    plt.close(figure)

    produced = [comparison_png, comparison_pdf]
    for result in results:
        figure, ax = plt.subplots(figsize=(8.6, 7.6), constrained_layout=True)
        image, points = draw_panel(
            ax, result, boxsize=boxsize, density_norm=density_norm,
            lambda_norm=lambda_norm, point_size=2.2,
        )
        ax.set_xlabel(rf"${xlabel}\ [h^{{-1}}\,\mathrm{{Mpc}}]$")
        ax.set_ylabel(rf"${ylabel}\ [h^{{-1}}\,\mathrm{{Mpc}}]$")
        figure.colorbar(image, ax=ax, pad=0.02).set_label(
            r"$\log_{10}(\Sigma/\langle\Sigma\rangle)$"
        )
        figure.colorbar(points, ax=ax, pad=0.08).set_label(r"true $\lambda_1$")
        stem = output / f"phase_density_galaxies_{result['phase']}"
        for suffix in ("png", "pdf"):
            path = stem.with_suffix(f".{suffix}")
            figure.savefig(path, dpi=args.dpi if suffix == "png" else None, bbox_inches="tight")
            produced.append(path)
        plt.close(figure)

    payload = {
        "schema_version": "p10-phase-density-galaxy-overlay-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_phase_accessed": False,
        "phases": list(phases),
        "slice": {
            "axis": args.slice_axis,
            "centre_mpc_h": args.slice_centre_mpc_h,
            "half_width_mpc_h": args.slice_half_width_mpc_h,
            "boxsize_mpc_h": boxsize,
            "ngrid": ngrid,
            "cell_mpc_h": boxsize / ngrid,
            "display_smoothing_mpc_h": args.display_smoothing_mpc_h,
        },
        "shared_scales": {
            "density_log10_ratio": density_limits.tolist(),
            "lambda1": lambda_limits.tolist(),
            "lambda1_diverging_centre": 0.2,
        },
        "sampling": {
            "active_rows_per_phase": args.sample_rows,
            "maximum_points_per_panel": args.max_points_per_panel,
            "seed": args.seed,
            "one_point_per_unique_file_num_halo_index": True,
        },
        "inputs": inputs,
        "phase_statistics": {result["phase"]: result["stats"] for result in results},
        "outputs": [str(path) for path in produced],
        "alignment_contract": (
            "Particle positions and host x_com are both reduced modulo the canonical "
            "2000 Mpc/h periodic box. Density axis order is x,y,z and the plotted "
            "array is transposed only for imshow row/column orientation."
        ),
    }
    manifest = output / "phase_density_galaxies_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": payload["outputs"], "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
