#!/usr/bin/env python3
"""P8 true-matter-field context-growth diagnostic.

The full periodic Abacus density field supplies the reference tensor.  Finite
context solves use the same downsampled field, global density normalization,
Gaussian smoothing, and Fourier tidal operator, so their difference isolates
the missing external field rather than a change in tracer or target definition.

The canonical 2048-grid eigenvalues are retained as a separate resolution-floor
comparison; the context convergence itself is always measured against the
matched 1024-grid full-field tensor.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import fitsio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, authoritative_mask, sha256
from workflows.abacus_tweb.validate_cutsky_eigs_boxindex_vs_halo_xcom import load_halo_positions_xcom


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
DENSITY = Path(
    "/pscratch/sd/d/dkololgi/AbacusSummit_densities/density_fields/"
    "AbacusSummit_base_c000_ph000_z0.200_ngrid_2048_10pc_density_field.npy"
)
CATALOGUE = Path(
    "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/"
    "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_"
    "ngrid2048_thr0p2_halo_xcom.fits"
)
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
HALO_INFO = Path(
    "/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/"
    "AbacusSummit_base_c000_ph000/halos/z0.200/halo_info"
)
BOXSIZE_MPC_H = 2000.0
RSMOOTH_MPC_H = 7.0
CONTEXT_RADII_MPC_H = (60.0, 120.0, 180.0, 240.0, 360.0)


def downsample_mean(source: np.ndarray, destination: np.ndarray, factor: int, slab: int) -> float:
    """Block-average a cubic field without materialising the full source."""
    if source.ndim != 3 or len(set(source.shape)) != 1:
        raise ValueError(f"source must be cubic, got {source.shape}")
    if source.shape[0] % factor or destination.shape != tuple(v // factor for v in source.shape):
        raise ValueError("source/destination/factor shape mismatch")
    slab = max(factor, (int(slab) // factor) * factor)
    total = 0.0
    count = 0
    n = source.shape[0]
    for left in range(0, n, slab):
        right = min(left + slab, n)
        if (right - left) % factor:
            raise ValueError("slab boundaries must align to the downsample factor")
        values = np.asarray(source[left:right], dtype=np.float32)
        reduced = values.reshape(
            (right - left) // factor, factor,
            n // factor, factor,
            n // factor, factor,
        ).mean(axis=(1, 3, 5), dtype=np.float64).astype(np.float32)
        destination[left // factor:right // factor] = reduced
        total += float(values.sum(dtype=np.float64))
        count += int(values.size)
    destination.flush()
    return total / count


def periodic_cube(field: np.ndarray, centre: np.ndarray, half_width: int) -> np.ndarray:
    """Extract an odd periodic cube centred on one grid cell."""
    centre = np.asarray(centre, dtype=np.int64)
    indices = [np.arange(c - half_width, c + half_width + 1) % field.shape[a]
               for a, c in enumerate(centre)]
    return np.asarray(field[np.ix_(*indices)], dtype=np.float32)


def cosine_taper(shape: tuple[int, int, int], width: int, device: str) -> torch.Tensor:
    """Separable Tukey-like taper that is one in the interior and zero at edges."""
    axes = []
    for size in shape:
        axis = torch.ones(size, dtype=torch.float32, device=device)
        use = min(int(width), size // 2)
        if use:
            phase = torch.linspace(0.0, math.pi, use + 2, device=device)[1:-1]
            ramp = 0.5 * (1.0 - torch.cos(phase))
            axis[:use] = ramp
            axis[-use:] = torch.flip(ramp, dims=(0,))
        axes.append(axis)
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]


def _frequency_axes(shape: tuple[int, int, int], cell: float, device: str):
    return (
        torch.fft.fftfreq(shape[0], d=cell, device=device) * (2.0 * math.pi),
        torch.fft.fftfreq(shape[1], d=cell, device=device) * (2.0 * math.pi),
        torch.fft.rfftfreq(shape[2], d=cell, device=device) * (2.0 * math.pi),
    )


def tensor_at_indices(
    field: torch.Tensor,
    indices: np.ndarray,
    *,
    cell_mpc_h: float,
    rsmooth_mpc_h: float,
) -> np.ndarray:
    """Solve the fixed tidal operator and sample six components at integer cells."""
    if field.ndim != 3:
        raise ValueError("field must be three-dimensional")
    indices = np.asarray(indices, dtype=np.int64)
    shape = tuple(int(v) for v in field.shape)
    dk = torch.fft.rfftn(field)
    kx, ky, kz = _frequency_axes(shape, cell_mpc_h, str(field.device))
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    kernel = torch.exp(-0.5 * k2 * rsmooth_mpc_h**2)
    kernel = kernel / torch.where(k2 > 0, k2, torch.ones_like(k2))
    kernel[0, 0, 0] = 0.0
    result = torch.empty((len(indices), 3, 3), dtype=torch.float32, device=field.device)
    axes = (kx, ky, kz)
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        left = axes[i].reshape((-1, 1, 1) if i == 0 else ((1, -1, 1) if i == 1 else (1, 1, -1)))
        right = axes[j].reshape((-1, 1, 1) if j == 0 else ((1, -1, 1) if j == 1 else (1, 1, -1)))
        component = torch.fft.irfftn(dk * kernel * left * right, s=shape)
        sampled = component[indices[:, 0], indices[:, 1], indices[:, 2]]
        result[:, i, j] = sampled
        result[:, j, i] = sampled
        del component, sampled
    output = result.cpu().numpy()
    del result, dk, kernel, k2
    return output


def tensor_invariants(tensor: np.ndarray) -> dict[str, np.ndarray]:
    tensor = np.asarray(tensor, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(tensor)
    trace = np.trace(tensor, axis1=1, axis2=2)
    shear_eigenvalues = eigenvalues - trace[:, None] / 3.0
    shear_frobenius = np.sqrt(np.sum(shear_eigenvalues**2, axis=1))
    return {
        "eigenvalues": eigenvalues,
        "trace": trace,
        "shear_eigenvalues": shear_eigenvalues,
        "shear_frobenius": shear_frobenius,
    }


def choose_anchors(assignment, n_boundary_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One deterministic representative per shell and fold-boundary-distance quantile."""
    auth = authoritative_mask(assignment)
    eligible = auth & np.asarray(assignment["supervised_eligible"], dtype=bool)
    rows, bins, quantiles = [], [], []
    distance = np.asarray(assignment["distance_to_conservative_fold_boundary_mpc"], dtype=np.float64)
    shell = np.asarray(assignment["shell"], dtype=np.int8)
    parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    for shell_id in range(4):
        candidates = np.flatnonzero(eligible & (shell == shell_id))
        values = distance[candidates]
        order = np.argsort(values, kind="mergesort")
        for boundary_bin, quantile in enumerate(np.linspace(0.1, 0.9, n_boundary_bins)):
            selected = candidates[order[int(round(quantile * (len(order) - 1)))]]
            rows.append(selected)
            bins.append(boundary_bin)
            quantiles.append(quantile)
    return np.asarray(rows), np.asarray(bins), np.asarray(quantiles)


def summarize_errors(got: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict:
    def summary(diff, scale):
        diff = np.asarray(diff, dtype=np.float64)
        scale = max(float(scale), 1e-12)
        return {
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "mae": float(np.mean(np.abs(diff))),
            "rmse_over_reference_std": float(np.sqrt(np.mean(diff**2)) / scale),
            "max_abs": float(np.max(np.abs(diff))),
        }
    return {
        "trace": summary(got["trace"] - reference["trace"], np.std(reference["trace"])),
        "traceless_shear_eigenvalues": summary(
            got["shear_eigenvalues"] - reference["shear_eigenvalues"],
            np.std(reference["shear_eigenvalues"]),
        ),
        "shear_frobenius": summary(
            got["shear_frobenius"] - reference["shear_frobenius"],
            np.std(reference["shear_frobenius"]),
        ),
        "eigenvalues": summary(
            got["eigenvalues"] - reference["eigenvalues"],
            np.std(reference["eigenvalues"]),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--density", type=Path, default=DENSITY)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--halo-info", type=Path, default=HALO_INFO)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--downsample-slab", type=int, default=16)
    parser.add_argument("--boundary-bins", type=int, default=3)
    parser.add_argument("--context-radii", type=float, nargs="+", default=CONTEXT_RADII_MPC_H)
    parser.add_argument("--apodization-mpc-h", type=float, default=2.0 * RSMOOTH_MPC_H)
    parser.add_argument("--padding-mpc-h", type=float, default=2.0 * RSMOOTH_MPC_H)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("true-field context diagnostic requires a CUDA allocation")
    started = time.time()
    output = args.p8_root / "true_field_context_v1"
    output.mkdir(parents=True, exist_ok=True)
    source = np.load(args.density, mmap_mode="r")
    ngrid = source.shape[0]
    factor = int(args.downsample_factor)
    downsample_path = output / f"density_blockmean_ngrid{ngrid // factor}.npy"
    if not downsample_path.exists():
        destination = np.lib.format.open_memmap(
            downsample_path, mode="w+", dtype=np.float32,
            shape=(ngrid // factor,) * 3,
        )
        density_mean = downsample_mean(source, destination, factor, args.downsample_slab)
        del destination
        atomic_json(output / "downsample_manifest.json", {
            "source": str(args.density), "source_sha256": sha256(args.density),
            "factor": factor, "method": "exact non-overlapping block mean",
            "shape": [ngrid // factor] * 3, "global_density_mean": density_mean,
            "output": str(downsample_path), "output_sha256": sha256(downsample_path),
        })
    downsample_manifest = json.loads((output / "downsample_manifest.json").read_text())
    density_mean = float(downsample_manifest["global_density_mean"])
    density = np.load(downsample_path, mmap_mode="r")
    assignment = np.load(args.assignment, mmap_mode="r")
    anchor_rows, boundary_bins, boundary_quantiles = choose_anchors(assignment, args.boundary_bins)
    parent = np.asarray(assignment["parent_node_id"][anchor_rows], dtype=np.int64)
    shell = np.asarray(assignment["shell"][anchor_rows], dtype=np.int8)
    cap = np.asarray(assignment["cap"][anchor_rows], dtype=np.int8)
    boundary_distance = np.asarray(
        assignment["distance_to_conservative_fold_boundary_mpc"][anchor_rows], dtype=np.float64
    )
    columns = fitsio.read(
        args.catalogue, rows=parent,
        columns=["FILE_NUM", "HALO_INDEX", "LAMBDA1", "LAMBDA2", "LAMBDA3", "TARGETID"],
    )
    box_positions = load_halo_positions_xcom(
        halo_info_dir=args.halo_info,
        file_nums=np.asarray(columns["FILE_NUM"]),
        halo_indices=np.asarray(columns["HALO_INDEX"]),
    ).astype(np.float64)
    cell = BOXSIZE_MPC_H / density.shape[0]
    full_indices = np.floor(box_positions / cell).astype(np.int64) % density.shape[0]
    full_field = torch.from_numpy(
        np.asarray(density, dtype=np.float32) / density_mean - 1.0
    ).to(args.device)
    full_tensor = tensor_at_indices(
        full_field, full_indices, cell_mpc_h=cell, rsmooth_mpc_h=RSMOOTH_MPC_H
    )
    full_reference = tensor_invariants(full_tensor)
    del full_field
    torch.cuda.empty_cache()
    canonical_eigenvalues = np.column_stack([
        columns["LAMBDA1"], columns["LAMBDA2"], columns["LAMBDA3"]
    ]).astype(np.float64)
    canonical_trace = canonical_eigenvalues.sum(axis=1)
    canonical_shear = canonical_eigenvalues - canonical_trace[:, None] / 3.0
    canonical = {
        "eigenvalues": canonical_eigenvalues,
        "trace": canonical_trace,
        "shear_eigenvalues": canonical_shear,
        "shear_frobenius": np.sqrt(np.sum(canonical_shear**2, axis=1)),
    }
    resolution_floor = summarize_errors(full_reference, canonical)
    radii_report = {}
    local_tensors = []
    for radius in args.context_radii:
        half_width = int(math.ceil(float(radius) / cell))
        apodization = int(math.ceil(args.apodization_mpc_h / cell))
        padding = int(math.ceil(args.padding_mpc_h / cell))
        tensors = []
        for centre in full_indices:
            cube = periodic_cube(density, centre, half_width)
            values = torch.from_numpy(cube / density_mean - 1.0).to(args.device)
            values *= cosine_taper(tuple(values.shape), apodization, args.device)
            if padding:
                values = torch.nn.functional.pad(values[None, None], (padding,) * 6)[0, 0]
            local_centre = np.asarray([[half_width + padding] * 3], dtype=np.int64)
            tensor = tensor_at_indices(
                values, local_centre, cell_mpc_h=cell, rsmooth_mpc_h=RSMOOTH_MPC_H
            )[0]
            tensors.append(tensor)
            del values
            torch.cuda.empty_cache()
        tensors = np.asarray(tensors)
        local_tensors.append(tensors)
        local = tensor_invariants(tensors)
        overall = summarize_errors(local, full_reference)
        by_shell = {
            str(s): summarize_errors(
                {k: v[shell == s] for k, v in local.items()},
                {k: v[shell == s] for k, v in full_reference.items()},
            ) for s in range(4)
        }
        by_boundary = {
            str(b): summarize_errors(
                {k: v[boundary_bins == b] for k, v in local.items()},
                {k: v[boundary_bins == b] for k, v in full_reference.items()},
            ) for b in range(args.boundary_bins)
        }
        radii_report[str(float(radius))] = {
            "radius_mpc_h": float(radius), "half_width_voxels": half_width,
            "cube_side_voxels": 2 * half_width + 1,
            "apodization_voxels": apodization, "padding_voxels": padding,
            "overall": overall, "by_shell": by_shell, "by_boundary_bin": by_boundary,
        }
    local_tensors = np.asarray(local_tensors, dtype=np.float32)
    np.save(output / "anchor_parent_node_id.npy", parent)
    np.save(output / "anchor_box_position_mpc_h.npy", box_positions)
    np.save(output / "full_tensor_ngrid1024.npy", full_tensor.astype(np.float32))
    np.save(output / "local_tensor_by_radius.npy", local_tensors)
    report = {
        "schema_version": 1,
        "stage": "P8 true-field physical-context convergence",
        "status": "TRUE_FIELD_CONTEXT_COMPLETE",
        "reference": {
            "description": "full periodic matched-resolution true matter field",
            "boxsize_mpc_h": BOXSIZE_MPC_H, "ngrid": int(density.shape[0]),
            "cell_mpc_h": cell, "smoothing_mpc_h": RSMOOTH_MPC_H,
            "density_normalization": "global rho/mean(rho)-1",
        },
        "finite_context": {
            "window": "periodic true-density cube, cosine tapered over 2Rs and zero padded by 2Rs",
            "radii_mpc_h": [float(v) for v in args.context_radii],
        },
        "anchors": {
            "n": int(len(parent)), "per_shell": int(args.boundary_bins),
            "selection": "deterministic 0.1/0.5/0.9 quantiles of conservative fold-boundary distance",
            "parent_node_id": parent.tolist(), "targetid": np.asarray(columns["TARGETID"], dtype=np.int64).tolist(),
            "shell": shell.tolist(), "cap": cap.tolist(),
            "boundary_bin": boundary_bins.tolist(),
            "boundary_quantile": boundary_quantiles.tolist(),
            "boundary_distance_mpc": boundary_distance.tolist(),
        },
        "matched_resolution_vs_canonical_2048_floor": resolution_floor,
        "radii": radii_report,
        "inputs": {
            "density": str(args.density),
            "density_sha256": downsample_manifest["source_sha256"],
            "downsampled_density": str(downsample_path),
            "downsample_manifest": str(output / "downsample_manifest.json"),
            "catalogue": str(args.catalogue), "assignment": str(args.assignment),
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output / "context_convergence_report.json", report)
    (output / "TRUE_FIELD_CONTEXT_COMPLETE").write_text(
        f"anchors={len(parent)} radii={len(args.context_radii)} elapsed_seconds={report['elapsed_seconds']:.1f}\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
