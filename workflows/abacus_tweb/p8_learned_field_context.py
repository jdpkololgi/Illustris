#!/usr/bin/env python3
"""Measure finite-context tidal convergence of the stitched learned D0 field."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from astropy.cosmology import Planck18
import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_density_target_alignment import CATALOGUE, read_rows
from workflows.abacus_tweb.p8_deterministic_common import (
    acquire_run_lock,
    atomic_json,
    authoritative_mask,
    sha256,
)
from workflows.abacus_tweb.p8_evaluate_stitched_density import (
    STITCHED,
    TARGET_MANIFEST,
    predicted_path,
)
from workflows.abacus_tweb.p8_true_field_context import (
    cosine_taper,
    summarize_errors,
    tensor_invariants,
)
from workflows.abacus_tweb.p8_validate_density_target_trace import (
    ASSIGNMENT,
    CAP_NAME,
    sky_to_observer_mpc,
)
from workflows.abacus_tweb.p8_validate_density_tensor_closure import (
    radial_cosine_window,
    solve_tensors_at_positions,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
OUTPUT = ROOT / "p8_density_phys_v1/d0_learned_context/rotation_0/seed_42"
TRUE_CONTEXT = ROOT / "p8_deterministic_v1/true_field_context_v1/context_convergence_report.json"
CONTEXT_RADII_MPC_H = (60.0, 120.0, 180.0, 240.0, 360.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stitched", type=Path, default=STITCHED)
    parser.add_argument("--target-manifest", type=Path, default=TARGET_MANIFEST)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--true-context", type=Path, default=TRUE_CONTEXT)
    parser.add_argument(
        "--context-radii-mpc-h", type=float, nargs="+", default=CONTEXT_RADII_MPC_H
    )
    parser.add_argument("--boundary-bins", type=int, default=3)
    parser.add_argument("--radial-taper-mpc", type=float, default=100.0)
    parser.add_argument(
        "--local-apodization-mpc", type=float, default=2 * 7.0 / float(Planck18.h)
    )
    parser.add_argument(
        "--local-padding-mpc", type=float, default=2 * 7.0 / float(Planck18.h)
    )
    parser.add_argument("--global-padding-voxels", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def choose_anchors(assignment, validation_fold: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
    auth = authoritative_mask(assignment)
    eligible = auth & (np.asarray(assignment["fold"]) == validation_fold)
    cap = np.asarray(assignment["cap"], dtype=np.int8)
    shell = np.asarray(assignment["shell"], dtype=np.int8)
    distance = np.asarray(
        assignment["distance_to_conservative_fold_boundary_mpc"], dtype=np.float64
    )
    rows, boundary_bin = [], []
    for cap_id in (0, 1):
        for shell_id in range(4):
            candidates = np.flatnonzero(eligible & (cap == cap_id) & (shell == shell_id))
            order = candidates[np.argsort(distance[candidates], kind="mergesort")]
            for index, quantile in enumerate(np.linspace(0.1, 0.9, bins)):
                rows.append(int(order[int(round(quantile * (len(order) - 1)))]))
                boundary_bin.append(index)
    return np.asarray(rows, dtype=np.int64), np.asarray(boundary_bin, dtype=np.int8)


def extract_cube(field: torch.Tensor, centre: np.ndarray, half_width: int) -> torch.Tensor:
    centre = np.asarray(centre, dtype=np.int64)
    target_shape = (2 * half_width + 1,) * 3
    result = torch.zeros(target_shape, dtype=field.dtype, device=field.device)
    source_slices, target_slices = [], []
    for axis in range(3):
        low = int(centre[axis] - half_width)
        high = int(centre[axis] + half_width + 1)
        source_low = max(low, 0)
        source_high = min(high, int(field.shape[axis]))
        target_low = source_low - low
        target_high = target_low + (source_high - source_low)
        source_slices.append(slice(source_low, source_high))
        target_slices.append(slice(target_low, target_high))
    result[tuple(target_slices)] = field[tuple(source_slices)]
    return result


def load_predicted_science_field(
    component: dict,
    prediction_path: Path,
    device: str,
    radial_taper_mpc: float,
) -> torch.Tensor:
    with h5py.File(component["file"], "r") as target:
        support = torch.from_numpy(
            np.asarray(target["science_support"], dtype=np.float32)
        ).to(device)
    with h5py.File(component["source_field"], "r") as source:
        exposure = torch.from_numpy(
            np.asarray(source["exposure_apodized"], dtype=np.float32)
        ).to(device)
    with h5py.File(prediction_path, "r") as predicted:
        array = np.nan_to_num(
            np.asarray(predicted["predicted_delta_r7"], dtype=np.float32)
        )
    field = torch.from_numpy(array).to(device)
    grid = component["grid"]
    radial = radial_cosine_window(
        tuple(int(value) for value in field.shape),
        np.asarray(grid["origin_mpc"], dtype=np.float64),
        float(grid["cell_mpc"]),
        radial_taper_mpc,
        device,
    )
    return field * support * exposure * radial


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("learned-field context diagnostic requires an interactive GPU")
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_lock(
        args.output / ".context.lock",
        purpose="P8.9 learned finite-context diagnostic",
    )
    target_manifest = json.loads(args.target_manifest.read_text())
    config = json.loads((ROOT / "p8_density_phys_v1/training_contract/rotation_0/d0_config.json").read_text())
    validation_fold = int(config["roles"]["validation_fold"])
    assignment = np.load(args.assignment, mmap_mode="r")
    anchor_rows, boundary_bin = choose_anchors(
        assignment, validation_fold, args.boundary_bins
    )
    parent = np.asarray(assignment["parent_node_id"][anchor_rows], dtype=np.int64)
    cap = np.asarray(assignment["cap"][anchor_rows], dtype=np.int8)
    shell = np.asarray(assignment["shell"][anchor_rows], dtype=np.int8)
    boundary_distance = np.asarray(
        assignment["distance_to_conservative_fold_boundary_mpc"][anchor_rows],
        dtype=np.float64,
    )
    catalogue = read_rows(args.catalogue, parent, ["TARGETID", "RA", "DEC", "Z"])
    position = sky_to_observer_mpc(
        np.asarray(catalogue["RA"], dtype=np.float64),
        np.asarray(catalogue["DEC"], dtype=np.float64),
        np.asarray(catalogue["Z"], dtype=np.float64),
    )
    global_tensor = np.empty((len(parent), 3, 3), dtype=np.float32)
    radii_mpc_h = np.asarray(args.context_radii_mpc_h, dtype=np.float64)
    radii_mpc = radii_mpc_h / float(Planck18.h)
    local_tensor = np.empty((len(radii_mpc), len(parent), 3, 3), dtype=np.float32)
    cap_reports = {}
    for cap_id, cap_name in CAP_NAME.items():
        selected = cap == cap_id
        component = target_manifest["components"][cap_name]
        grid = component["grid"]
        origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
        cell = float(grid["cell_mpc"])
        predicted = load_predicted_science_field(
            component, predicted_path(args.stitched, cap_name), args.device,
            args.radial_taper_mpc,
        )
        global_sample, global_fft = solve_tensors_at_positions(
            predicted,
            positions={"observed": position[selected]},
            origin_mpc=origin,
            cell_mpc=cell,
            padding_voxels=args.global_padding_voxels,
        )
        global_tensor[selected] = global_sample["observed"]
        index = np.rint((position[selected] - origin[None, :]) / cell - 0.5).astype(np.int64)
        if np.any(index < 0) or np.any(index >= np.asarray(predicted.shape)[None, :]):
            raise RuntimeError(f"anchor outside {cap_name} field lattice")
        for radius_index, radius in enumerate(radii_mpc):
            half_width = int(math.ceil(float(radius) / cell))
            taper = int(math.ceil(args.local_apodization_mpc / cell))
            padding = int(math.ceil(args.local_padding_mpc / cell))
            rows = []
            for centre in index:
                cube = extract_cube(predicted, centre, half_width)
                cube *= cosine_taper(tuple(cube.shape), taper, args.device)
                if padding:
                    cube = F.pad(cube[None, None], (padding,) * 6)[0, 0]
                cube_origin = np.zeros(3, dtype=np.float64)
                centre_position = np.full((1, 3), (half_width + padding + 0.5) * cell)
                sampled, _ = solve_tensors_at_positions(
                    cube,
                    positions={"centre": centre_position},
                    origin_mpc=cube_origin,
                    cell_mpc=cell,
                    padding_voxels=0,
                )
                rows.append(sampled["centre"][0])
                del cube
            local_tensor[radius_index, selected] = np.asarray(rows, dtype=np.float32)
            torch.cuda.empty_cache()
        cap_reports[cap_name] = {"global_fft": global_fft, "anchors": int(selected.sum())}
        del predicted
        torch.cuda.empty_cache()

    reference = tensor_invariants(global_tensor)
    radii_report = {}
    for index, (radius_mpc_h, radius) in enumerate(zip(radii_mpc_h, radii_mpc, strict=True)):
        local = tensor_invariants(local_tensor[index])
        radii_report[str(float(radius_mpc_h))] = {
            "radius_mpc": float(radius),
            "radius_mpc_h": float(radius_mpc_h),
            "overall": summarize_errors(local, reference),
            "by_shell": {
                str(shell_id): summarize_errors(
                    {name: values[shell == shell_id] for name, values in local.items()},
                    {name: values[shell == shell_id] for name, values in reference.items()},
                )
                for shell_id in range(4)
            },
            "by_boundary_bin": {
                str(bin_id): summarize_errors(
                    {name: values[boundary_bin == bin_id] for name, values in local.items()},
                    {name: values[boundary_bin == bin_id] for name, values in reference.items()},
                )
                for bin_id in range(args.boundary_bins)
            },
        }
    np.savez_compressed(
        args.output / "learned_context_tensors.npz",
        parent_node_id=parent,
        global_tensor=global_tensor,
        local_tensor_by_radius=local_tensor,
        context_radii_mpc=radii_mpc,
        context_radii_mpc_h=radii_mpc_h,
    )
    true_context = json.loads(args.true_context.read_text()) if args.true_context.exists() else None
    report = {
        "schema_version": "p8-density-learned-context-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_sha(),
        "status": "PASS",
        "reference": "stitched predicted science-window field; one global cap FFT",
        "finite_context": {
            "radii_mpc": radii_mpc.tolist(),
            "radii_mpc_h": radii_mpc_h.tolist(),
            "unit_conversion": "radius_mpc = radius_mpc_h / Planck18.h",
            "local_window": "cosine taper over 2 R_s and zero padding by 2 R_s",
            "double_smoothing_applied": False,
        },
        "anchors": {
            "n": int(len(parent)),
            "selection": "rotation-0 validation fold; cap x shell x boundary-distance quantile",
            "parent_node_id": parent.tolist(),
            "targetid": np.asarray(catalogue["TARGETID"], dtype=np.int64).tolist(),
            "cap": cap.tolist(),
            "shell": shell.tolist(),
            "boundary_bin": boundary_bin.tolist(),
            "boundary_distance_mpc": boundary_distance.tolist(),
        },
        "radii": radii_report,
        "true_field_floor": {
            "path": str(args.true_context),
            "sha256": sha256(args.true_context) if args.true_context.exists() else None,
            "radii": true_context.get("radii") if true_context else None,
        },
        "caps": cap_reports,
        "inputs": {
            "stitched_manifest": str(args.stitched / "stitched_field_manifest.json"),
            "stitched_manifest_sha256": sha256(
                args.stitched / "stitched_field_manifest.json"
            ),
            "target_manifest": str(args.target_manifest),
            "target_manifest_sha256": sha256(args.target_manifest),
        },
        "tensor_artifact": str(args.output / "learned_context_tensors.npz"),
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_json(args.output / "learned_context_report.json", report)
    (args.output / "D0_LEARNED_CONTEXT_COMPLETE").write_text(
        f"anchors={len(parent)} radii={len(radii_mpc)}\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    run_lock.close()


if __name__ == "__main__":
    main()
