#!/usr/bin/env python3
"""Validate the P8.9 one-global-FFT-per-cap tensor contract before D0 training.

The privileged target is already the R=7 Mpc/h smoothed matter contrast, so this
program applies only the unsmoothed tidal projector k_i k_j/k^2.  Four frozen
field/window variants separate implementation closure from missing-survey-volume
physics.  Z_COSMO is an oracle coordinate row; observed Z is the deployable row.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time

from astropy.cosmology import Planck18
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from workflows.abacus_tweb.p8_density_target_alignment import (
    CATALOGUE,
    TARGET_INPUT,
    choose_sample,
    join_target_truth,
    read_rows,
)
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_validate_density_target_trace import (
    ASSIGNMENT,
    CAP_NAME,
    TARGET_MANIFEST,
    scalar_score,
    sky_to_observer_mpc,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
OUTPUT = ROOT / "p8_density_phys_v1/tensor_closure"
SHELLS = ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55))
VARIANTS = (
    "rectangle_raw",
    "rectangle_box_taper",
    "science_hard",
    "science_apodized",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--target-input", type=Path, default=TARGET_INPUT)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--target-manifest", type=Path, default=TARGET_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sample-per-cap-shell", type=int, default=2_000)
    parser.add_argument("--target-chunk", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--padding-voxels", type=int, default=24)
    parser.add_argument("--box-taper-voxels", type=int, default=20)
    parser.add_argument("--radial-taper-mpc", type=float, default=100.0)
    parser.add_argument("--minimum-rectangle-oracle-macro-r2-lambda1", type=float, default=0.90)
    parser.add_argument("--minimum-window-oracle-macro-r2-lambda1", type=float, default=0.50)
    parser.add_argument("--minimum-window-oracle-worst-shell-r2-lambda1", type=float, default=0.0)
    parser.add_argument("--maximum-trace-identity-rmse", type=float, default=2.0e-4)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def grid_coords(frac: np.ndarray, shape: tuple[int, int, int], device: str) -> torch.Tensor:
    frac = np.asarray(frac, dtype=np.float64)
    norm = np.empty_like(frac, dtype=np.float32)
    for axis, size in enumerate(shape):
        norm[:, axis] = 2.0 * frac[:, axis] / max(size - 1, 1) - 1.0
    if np.any(norm < -1.00001) or np.any(norm > 1.00001):
        raise RuntimeError("tensor-closure sample lies outside its cap lattice")
    # grid_sample uses W,H,D = iz,iy,ix for a C,ix,iy,iz field.
    return torch.from_numpy(np.ascontiguousarray(norm[:, (2, 1, 0)])).to(device).view(
        1, 1, 1, -1, 3
    )


def cosine_axis(size: int, width: int, device: str) -> torch.Tensor:
    result = torch.ones(size, dtype=torch.float32, device=device)
    use = min(int(width), size // 2)
    if use:
        phase = (torch.arange(use, device=device, dtype=torch.float32) + 0.5) / use
        ramp = 0.5 - 0.5 * torch.cos(math.pi * phase)
        result[:use] = ramp
        result[-use:] = torch.flip(ramp, dims=(0,))
    return result


def apply_box_taper(field: torch.Tensor, width: int) -> torch.Tensor:
    output = field.clone()
    for axis, size in enumerate(output.shape):
        taper = cosine_axis(int(size), int(width), str(output.device))
        shape = [1, 1, 1]
        shape[axis] = int(size)
        output.mul_(taper.reshape(shape))
    return output


def radial_cosine_window(
    shape: tuple[int, int, int],
    origin_mpc: np.ndarray,
    cell_mpc: float,
    radial_taper_mpc: float,
    device: str,
) -> torch.Tensor:
    """Return a 0-to-1 taper over the frozen z=0.15--0.55 radial support."""
    axes = [
        torch.as_tensor(
            origin_mpc[a] + (np.arange(shape[a], dtype=np.float32) + 0.5) * cell_mpc,
            device=device,
        )
        for a in range(3)
    ]
    radius = torch.sqrt(
        axes[0][:, None, None] ** 2
        + axes[1][None, :, None] ** 2
        + axes[2][None, None, :] ** 2
    )
    r_min = float(Planck18.comoving_distance(SHELLS[0][0]).value)
    r_max = float(Planck18.comoving_distance(SHELLS[-1][1]).value)
    low_phase = torch.clamp((radius - r_min) / radial_taper_mpc, 0.0, 1.0)
    window = 0.5 - 0.5 * torch.cos(math.pi * low_phase)
    high_phase = torch.clamp((r_max - radius) / radial_taper_mpc, 0.0, 1.0)
    window.mul_(0.5 - 0.5 * torch.cos(math.pi * high_phase))
    return window


def sample_scalar(field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(
        field[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0, 0, 0, 0]


def solve_tensors_at_positions(
    field: torch.Tensor,
    *,
    positions: dict[str, np.ndarray],
    origin_mpc: np.ndarray,
    cell_mpc: float,
    padding_voxels: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Apply the unsmoothed projector and sample six components without field retention."""
    device = str(field.device)
    original_shape = tuple(int(v) for v in field.shape)
    grids = {
        name: grid_coords((xyz - origin_mpc[None, :]) / cell_mpc - 0.5, original_shape, device)
        for name, xyz in positions.items()
    }
    padding = int(padding_voxels)
    work = F.pad(field[None, None], (padding,) * 6)[0, 0] if padding else field
    work_shape = tuple(int(v) for v in work.shape)
    removed_mean = float(work.mean().item())
    input_samples = {
        name: (sample_scalar(field, grid) - removed_mean).cpu().numpy()
        for name, grid in grids.items()
    }
    dk = torch.fft.rfftn(work)
    kx = torch.fft.fftfreq(work_shape[0], d=cell_mpc, device=device) * (2.0 * math.pi)
    ky = torch.fft.fftfreq(work_shape[1], d=cell_mpc, device=device) * (2.0 * math.pi)
    kz = torch.fft.rfftfreq(work_shape[2], d=cell_mpc, device=device) * (2.0 * math.pi)
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    kernel = 1.0 / torch.where(k2 > 0, k2, torch.ones_like(k2))
    kernel[0, 0, 0] = 0.0
    tensors = {
        name: torch.empty((len(xyz), 3, 3), dtype=torch.float32, device=device)
        for name, xyz in positions.items()
    }
    axes = (kx, ky, kz)
    peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        left = axes[i].reshape((-1, 1, 1) if i == 0 else ((1, -1, 1) if i == 1 else (1, 1, -1)))
        right = axes[j].reshape((-1, 1, 1) if j == 0 else ((1, -1, 1) if j == 1 else (1, 1, -1)))
        component = torch.fft.irfftn(dk * kernel * left * right, s=work_shape)
        if padding:
            component = component[
                padding:padding + original_shape[0],
                padding:padding + original_shape[1],
                padding:padding + original_shape[2],
            ]
        for name, grid in grids.items():
            sampled = sample_scalar(component, grid)
            tensors[name][:, i, j] = sampled
            tensors[name][:, j, i] = sampled
        peak = max(
            peak,
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
        )
        del component
    output = {name: tensor.cpu().numpy() for name, tensor in tensors.items()}
    trace_identity = {}
    for name, tensor in output.items():
        trace = np.trace(tensor, axis1=1, axis2=2)
        trace_identity[name] = scalar_score(trace, input_samples[name])
    diagnostics = {
        "padding_voxels": padding,
        "work_shape": list(work_shape),
        "removed_k0_mean": removed_mean,
        "maximum_cuda_memory_bytes": peak,
        "trace_identity": trace_identity,
    }
    del tensors, grids, dk, kernel, k2, work
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output, diagnostics


def eigensystem(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.eigh(np.asarray(tensor, dtype=np.float64))


def score_bundle(prediction_tensor: np.ndarray, truth_eigenvalues: np.ndarray) -> dict:
    prediction, _ = eigensystem(prediction_tensor)
    truth = np.asarray(truth_eigenvalues, dtype=np.float64)
    trace_prediction = prediction.sum(axis=1)
    trace_truth = truth.sum(axis=1)
    shear_prediction = prediction - trace_prediction[:, None] / 3.0
    shear_truth = truth - trace_truth[:, None] / 3.0
    return {
        "n": int(len(truth)),
        "eigenvalues": {
            f"lambda{i + 1}": scalar_score(prediction[:, i], truth[:, i])
            for i in range(3)
        },
        "trace": scalar_score(trace_prediction, trace_truth),
        "traceless_shear_eigenvalues": {
            f"s{i + 1}": scalar_score(shear_prediction[:, i], shear_truth[:, i])
            for i in range(3)
        },
        "ordering_violations": int(np.sum(np.any(np.diff(prediction, axis=1) < 0.0, axis=1))),
    }


def stratified_scores(
    tensor: np.ndarray,
    truth: np.ndarray,
    cap: np.ndarray,
    shell: np.ndarray,
) -> dict:
    overall = score_bundle(tensor, truth)
    by_shell = {str(s): score_bundle(tensor[shell == s], truth[shell == s]) for s in range(4)}
    by_cap = {str(c): score_bundle(tensor[cap == c], truth[cap == c]) for c in (0, 1)}
    shell_lambda1 = [by_shell[str(s)]["eigenvalues"]["lambda1"]["r2"] for s in range(4)]
    overall["by_shell"] = by_shell
    overall["by_cap"] = by_cap
    overall["macro_shell_r2_lambda1"] = float(np.mean(shell_lambda1))
    overall["worst_shell_r2_lambda1"] = float(np.min(shell_lambda1))
    return overall


def orientation_stability(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref_values, ref_vectors = eigensystem(reference)
    _, candidate_vectors = eigensystem(candidate)
    gaps = np.minimum(ref_values[:, 1] - ref_values[:, 0], ref_values[:, 2] - ref_values[:, 1])
    angle = np.rad2deg(np.arccos(np.clip(np.abs(np.sum(ref_vectors * candidate_vectors, axis=1)), 0.0, 1.0)))
    edges = np.quantile(gaps, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = {}
    for index in range(4):
        selected = (gaps >= edges[index]) & (
            gaps <= edges[index + 1] if index == 3 else gaps < edges[index + 1]
        )
        bins[str(index)] = {
            "n": int(np.sum(selected)),
            "eigengap_low": float(edges[index]),
            "eigengap_high": float(edges[index + 1]),
            "median_angle_deg": np.median(angle[selected], axis=0).tolist(),
            "p90_angle_deg": np.quantile(angle[selected], 0.9, axis=0).tolist(),
        }
    return {
        "reference": "rectangle_raw privileged tensor",
        "candidate": "science_apodized tensor",
        "sign_invariant_axis_angle": True,
        "truth_orientation_available": False,
        "eigengap_quantile_bins": bins,
    }


def load_variant_inputs(component: dict, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with h5py.File(component["file"], "r") as handle:
        delta = torch.from_numpy(np.asarray(handle["delta_r7"], dtype=np.float32)).to(device)
        support = torch.from_numpy(np.asarray(handle["science_support"], dtype=np.uint8)).to(device)
    with h5py.File(component["source_field"], "r") as handle:
        exposure = torch.from_numpy(np.asarray(handle["exposure_apodized"], dtype=np.float32)).to(device)
    return delta, support, exposure


def make_variant(
    name: str,
    delta: torch.Tensor,
    support: torch.Tensor,
    exposure: torch.Tensor,
    *,
    origin_mpc: np.ndarray,
    cell_mpc: float,
    box_taper_voxels: int,
    radial_taper_mpc: float,
) -> tuple[torch.Tensor, dict]:
    if name == "rectangle_raw":
        field = delta.clone()
    elif name == "rectangle_box_taper":
        field = apply_box_taper(delta, box_taper_voxels)
    elif name == "science_hard":
        field = delta * support
    elif name == "science_apodized":
        radial = radial_cosine_window(
            tuple(int(v) for v in delta.shape), origin_mpc, cell_mpc,
            radial_taper_mpc, str(delta.device),
        )
        field = delta * exposure * radial * support
        del radial
    else:
        raise ValueError(f"unknown field variant: {name}")
    nonzero = int(torch.count_nonzero(field).item())
    report = {
        "nonzero_voxels": nonzero,
        "nonzero_fraction": float(nonzero / field.numel()),
        "mean": float(field.mean().item()),
        "std": float(field.std().item()),
    }
    return field, report


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("global cap tensor closure requires an interactive CUDA allocation")
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.target_manifest.read_text())
    if manifest["contract"].get("double_smoothing_applied") is not False:
        raise RuntimeError("target manifest does not certify the no-double-smoothing contract")

    assignment = np.load(args.assignment, mmap_mode="r")
    sample_rows = choose_sample(assignment, args.sample_per_cap_shell, args.seed)
    parent = np.asarray(assignment["parent_node_id"][sample_rows], dtype=np.int64)
    cap = np.asarray(assignment["cap"][sample_rows], dtype=np.int8)
    shell = np.asarray(assignment["shell"][sample_rows], dtype=np.int8)
    catalogue = read_rows(
        args.catalogue, parent,
        ["TARGETID", "RA", "DEC", "Z", "LAMBDA1", "LAMBDA2", "LAMBDA3"],
    )
    joined = join_target_truth(
        args.target_input, np.asarray(catalogue["TARGETID"], dtype=np.int64),
        chunk_rows=args.target_chunk,
    )
    truth = np.column_stack([
        np.asarray(catalogue[f"LAMBDA{i}"], dtype=np.float64) for i in (1, 2, 3)
    ])
    ra = np.asarray(catalogue["RA"], dtype=np.float64)
    dec = np.asarray(catalogue["DEC"], dtype=np.float64)
    positions = {
        "z_cosmo": sky_to_observer_mpc(ra, dec, joined["Z_COSMO"]),
        "z_observed": sky_to_observer_mpc(ra, dec, np.asarray(catalogue["Z"], dtype=np.float64)),
    }

    tensor_by_variant = {
        variant: {coordinate: np.empty((len(parent), 3, 3), dtype=np.float32) for coordinate in positions}
        for variant in VARIANTS
    }
    cap_reports = {}
    for cap_id, cap_name in CAP_NAME.items():
        selected = cap == cap_id
        component = manifest["components"][cap_name]
        grid = component["grid"]
        origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
        cell = float(grid["cell_mpc"])
        delta, support, exposure = load_variant_inputs(component, args.device)
        cap_report = {"grid": grid, "variants": {}}
        for variant in VARIANTS:
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            field, field_report = make_variant(
                variant, delta, support, exposure, origin_mpc=origin, cell_mpc=cell,
                box_taper_voxels=args.box_taper_voxels,
                radial_taper_mpc=args.radial_taper_mpc,
            )
            sampled, fft_report = solve_tensors_at_positions(
                field,
                positions={name: value[selected] for name, value in positions.items()},
                origin_mpc=origin,
                cell_mpc=cell,
                padding_voxels=args.padding_voxels,
            )
            for coordinate, tensor in sampled.items():
                tensor_by_variant[variant][coordinate][selected] = tensor
            cap_report["variants"][variant] = {"field": field_report, "fft": fft_report}
            del field, sampled
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        cap_reports[cap_name] = cap_report
        del delta, support, exposure
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    scores = {}
    for variant in VARIANTS:
        scores[variant] = {
            coordinate: stratified_scores(tensor, truth, cap, shell)
            for coordinate, tensor in tensor_by_variant[variant].items()
        }
    trace_identity_rmse = max(
        report["fft"]["trace_identity"][coordinate]["rmse"]
        for cap_report in cap_reports.values()
        for report in cap_report["variants"].values()
        for coordinate in positions
    )
    rectangle_macro = scores["rectangle_raw"]["z_cosmo"]["macro_shell_r2_lambda1"]
    window_macro = scores["science_apodized"]["z_cosmo"]["macro_shell_r2_lambda1"]
    window_worst = scores["science_apodized"]["z_cosmo"]["worst_shell_r2_lambda1"]
    gates = {
        "trace_identity_all_variants": bool(trace_identity_rmse <= args.maximum_trace_identity_rmse),
        "rectangle_oracle_macro_lambda1": bool(
            rectangle_macro >= args.minimum_rectangle_oracle_macro_r2_lambda1
        ),
        "science_window_oracle_macro_lambda1": bool(
            window_macro >= args.minimum_window_oracle_macro_r2_lambda1
        ),
        "science_window_oracle_worst_shell_lambda1": bool(
            window_worst >= args.minimum_window_oracle_worst_shell_r2_lambda1
        ),
    }
    arrays = {
        "sample_assignment_row": sample_rows,
        "parent_node_id": parent,
        "cap": cap,
        "shell": shell,
        "truth_eigenvalues": truth.astype(np.float32),
    }
    for variant, coordinate_rows in tensor_by_variant.items():
        for coordinate, tensor in coordinate_rows.items():
            arrays[f"{variant}__{coordinate}__tensor"] = tensor
    predictions = args.output / "sampled_tensors.npz"
    np.savez_compressed(predictions, **arrays)
    payload = {
        "schema_version": "p8-density-tensor-closure-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "stage": "P8.9 one-global-FFT-per-cap tensor/eigenvalue closure",
        "contract": {
            "input": "privileged delta_R7 already smoothed at R=7 Mpc/h",
            "operator": "T_ij(k)=k_i*k_j/k^2*delta_R7(k); k=0 removed",
            "additional_gaussian_smoothing": False,
            "padding_voxels": int(args.padding_voxels),
            "padding_mpc": float(args.padding_voxels * 5.0),
            "box_taper_voxels": int(args.box_taper_voxels),
            "radial_taper_mpc": float(args.radial_taper_mpc),
            "science_apodized_window": "P3 exposure_apodized times 100-Mpc radial cosine taper",
            "oracle_coordinate": "Z_COSMO",
            "deployable_coordinate": "observed Z",
        },
        "sample": {
            "n": int(len(parent)),
            "per_cap_shell": int(args.sample_per_cap_shell),
            "seed": int(args.seed),
        },
        "scores": scores,
        "window_orientation_stability": {
            coordinate: orientation_stability(
                tensor_by_variant["rectangle_raw"][coordinate],
                tensor_by_variant["science_apodized"][coordinate],
            ) for coordinate in positions
        },
        "caps": cap_reports,
        "thresholds": {
            "maximum_trace_identity_rmse": float(args.maximum_trace_identity_rmse),
            "minimum_rectangle_oracle_macro_r2_lambda1": float(
                args.minimum_rectangle_oracle_macro_r2_lambda1
            ),
            "minimum_window_oracle_macro_r2_lambda1": float(
                args.minimum_window_oracle_macro_r2_lambda1
            ),
            "minimum_window_oracle_worst_shell_r2_lambda1": float(
                args.minimum_window_oracle_worst_shell_r2_lambda1
            ),
        },
        "gates": gates,
        "pass": bool(all(gates.values())),
        "predictions": str(predictions),
        "predictions_sha256": sha256(predictions),
        "inputs": {
            "target_manifest": str(args.target_manifest),
            "target_manifest_sha256": sha256(args.target_manifest),
            "assignment": str(args.assignment),
            "assignment_sha256": sha256(args.assignment),
            "catalogue": str(args.catalogue),
            "catalogue_sha256": sha256(args.catalogue),
            "target_input": str(args.target_input),
        },
        "interpretation": (
            "Rectangle rows are privileged volume-availability diagnostics; science-apodized "
            "rows test the current deployable field-support contract. Z_COSMO is never a DESI "
            "performance claim. Failure blocks D0 training until the output/context contract is "
            "revised; it is not repaired by weakening this gate after inspection."
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    report = args.output / "tensor_closure.json"
    atomic_json(report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
