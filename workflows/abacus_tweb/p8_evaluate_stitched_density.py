#!/usr/bin/env python3
"""Evaluate the stitched P8.9 density field and its one-global-FFT tidal product."""
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_density_target_alignment import (
    CATALOGUE,
    TARGET_INPUT,
    join_target_truth,
    read_rows,
)
from workflows.abacus_tweb.p8_deterministic_common import (
    acquire_run_lock,
    atomic_json,
    authoritative_mask,
    evaluate_complete_fold,
    fit_affine_on_training,
    sha256,
)
from workflows.abacus_tweb.p8_train_density_patch import RegressionAccumulator
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
STITCHED = ROOT / "p8_density_phys_v1/d0_stitched/rotation_0/seed_42"
TARGET_MANIFEST = ROOT / "p8_density_phys_v1/targets/target_manifest.json"
PARENT_TRUTH = ROOT / "p8_deterministic_v1/parent_eigenvalues.npy"
OUTPUT = ROOT / "p8_density_phys_v1/d0_evaluation/rotation_0/seed_42"
SHELLS = ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55))
SHELL_NAMES = ("0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55")
HISTOGRAM_EDGES = np.linspace(-1.0, 12.0, 261)
TAIL_THRESHOLDS = (-0.8, -0.5, 1.0, 3.0, 6.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stitched", type=Path, default=STITCHED)
    parser.add_argument("--target-manifest", type=Path, default=TARGET_MANIFEST)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--target-input", type=Path, default=TARGET_INPUT)
    parser.add_argument("--parent-truth", type=Path, default=PARENT_TRUTH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-chunk", type=int, default=1_000_000)
    parser.add_argument("--padding-voxels", type=int, default=24)
    parser.add_argument("--radial-taper-mpc", type=float, default=100.0)
    parser.add_argument("--spectral-bins", type=int, default=30)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def predicted_path(root: Path, cap_name: str) -> Path:
    return root / f"{cap_name.lower()}_predicted_delta_r7.h5"


def shell_boundaries_mpc() -> np.ndarray:
    return np.asarray(
        [Planck18.comoving_distance(SHELLS[0][0]).value]
        + [Planck18.comoving_distance(row[1]).value for row in SHELLS],
        dtype=np.float64,
    )


def shell_grid(
    start: int,
    stop: int,
    shape: tuple[int, int, int],
    origin: np.ndarray,
    cell: float,
) -> np.ndarray:
    x = origin[0] + (np.arange(start, stop, dtype=np.float64) + 0.5) * cell
    y = origin[1] + (np.arange(shape[1], dtype=np.float64) + 0.5) * cell
    z = origin[2] + (np.arange(shape[2], dtype=np.float64) + 0.5) * cell
    radius = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None, :] ** 2)
    return np.searchsorted(shell_boundaries_mpc(), radius, side="right") - 1


class DistributionAccumulator:
    def __init__(self):
        self.regression = RegressionAccumulator()
        self.truth_hist = np.zeros(len(HISTOGRAM_EDGES) - 1, dtype=np.int64)
        self.prediction_hist = np.zeros(len(HISTOGRAM_EDGES) - 1, dtype=np.int64)
        self.truth_underflow = self.truth_overflow = 0
        self.prediction_underflow = self.prediction_overflow = 0
        self.tail = {
            str(value): {"truth_count": 0, "prediction_count": 0, "truth_sum": 0.0,
                         "prediction_on_truth_selected_sum": 0.0}
            for value in TAIL_THRESHOLDS
        }

    def add(self, prediction: np.ndarray, truth: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        self.regression.add(prediction, truth)
        self.truth_hist += np.histogram(truth, HISTOGRAM_EDGES)[0]
        self.prediction_hist += np.histogram(prediction, HISTOGRAM_EDGES)[0]
        self.truth_underflow += int(np.count_nonzero(truth < HISTOGRAM_EDGES[0]))
        self.truth_overflow += int(np.count_nonzero(truth >= HISTOGRAM_EDGES[-1]))
        self.prediction_underflow += int(np.count_nonzero(prediction < HISTOGRAM_EDGES[0]))
        self.prediction_overflow += int(np.count_nonzero(prediction >= HISTOGRAM_EDGES[-1]))
        for threshold in TAIL_THRESHOLDS:
            selected_truth = truth < threshold if threshold < 0 else truth > threshold
            selected_prediction = prediction < threshold if threshold < 0 else prediction > threshold
            row = self.tail[str(threshold)]
            row["truth_count"] += int(np.count_nonzero(selected_truth))
            row["prediction_count"] += int(np.count_nonzero(selected_prediction))
            row["truth_sum"] += float(np.sum(truth[selected_truth], dtype=np.float64))
            row["prediction_on_truth_selected_sum"] += float(
                np.sum(prediction[selected_truth], dtype=np.float64)
            )

    def report(self) -> dict:
        base = self.regression.report()
        n = int(base["n"])
        tails = {}
        for threshold, row in self.tail.items():
            truth_n = row["truth_count"]
            tails[threshold] = {
                **row,
                "truth_fraction": float(truth_n / n),
                "prediction_fraction": float(row["prediction_count"] / n),
                "count_ratio_prediction_to_truth": float(
                    row["prediction_count"] / max(truth_n, 1)
                ),
                "truth_selected_mean_truth": (
                    float(row["truth_sum"] / truth_n) if truth_n else None
                ),
                "truth_selected_mean_prediction": (
                    float(row["prediction_on_truth_selected_sum"] / truth_n) if truth_n else None
                ),
            }
        return {
            **base,
            "histogram": {
                "edges": HISTOGRAM_EDGES.tolist(),
                "truth_counts": self.truth_hist.tolist(),
                "prediction_counts": self.prediction_hist.tolist(),
                "truth_underflow": self.truth_underflow,
                "truth_overflow": self.truth_overflow,
                "prediction_underflow": self.prediction_underflow,
                "prediction_overflow": self.prediction_overflow,
            },
            "tails": tails,
        }


def field_metrics(stitched: Path, manifest: dict) -> dict:
    overall = DistributionAccumulator()
    by_shell = [DistributionAccumulator() for _ in range(4)]
    by_cap = {}
    for cap_name in ("SGC", "NGC"):
        component = manifest["components"][cap_name]
        grid = component["grid"]
        shape = tuple(int(value) for value in grid["shape"])
        origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
        cell = float(grid["cell_mpc"])
        cap_accumulator = DistributionAccumulator()
        with h5py.File(component["file"], "r") as truth_handle, h5py.File(
            predicted_path(stitched, cap_name), "r"
        ) as pred_handle:
            truth = truth_handle["delta_r7"]
            support = truth_handle["science_support"]
            prediction = pred_handle["predicted_delta_r7"]
            step = int(prediction.chunks[0])
            for left in range(0, shape[0], step):
                right = min(left + step, shape[0])
                local_support = np.asarray(support[left:right], dtype=bool)
                local_prediction = np.asarray(prediction[left:right], dtype=np.float32)
                local_truth = np.asarray(truth[left:right], dtype=np.float32)
                if np.any(~np.isfinite(local_prediction[local_support])):
                    raise RuntimeError(f"non-finite supported prediction in {cap_name}")
                p = local_prediction[local_support]
                t = local_truth[local_support]
                overall.add(p, t)
                cap_accumulator.add(p, t)
                shells = shell_grid(left, right, shape, origin, cell)
                for shell in range(4):
                    selected = local_support & (shells == shell)
                    if np.any(selected):
                        by_shell[shell].add(local_prediction[selected], local_truth[selected])
        by_cap[cap_name] = cap_accumulator.report()
    shell_reports = {SHELL_NAMES[index]: item.report() for index, item in enumerate(by_shell)}
    return {
        "overall": overall.report(),
        "by_cap": by_cap,
        "by_shell": shell_reports,
        "macro_shell_r2_delta_r7": float(np.mean([row["r2"] for row in shell_reports.values()])),
    }


def spectral_sums(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    cell_mpc: float,
    edges_h_mpc: np.ndarray,
) -> dict[str, np.ndarray]:
    fp = torch.fft.rfftn(prediction)
    ft = torch.fft.rfftn(truth)
    shape = tuple(int(value) for value in prediction.shape)
    kx = torch.fft.fftfreq(shape[0], d=cell_mpc, device=prediction.device) * (2 * math.pi)
    ky = torch.fft.fftfreq(shape[1], d=cell_mpc, device=prediction.device) * (2 * math.pi)
    kz = torch.fft.rfftfreq(shape[2], d=cell_mpc, device=prediction.device) * (2 * math.pi)
    edges = torch.as_tensor(edges_h_mpc * float(Planck18.h), device=prediction.device)
    bins = len(edges_h_mpc) - 1
    sums = {
        name: torch.zeros(bins, dtype=torch.float64, device=prediction.device)
        for name in ("count", "cross", "prediction", "truth")
    }
    kz_weight = torch.full_like(kz, 2.0, dtype=torch.float64)
    kz_weight[0] = 1.0
    if shape[2] % 2 == 0:
        kz_weight[-1] = 1.0
    for left in range(0, shape[0], 8):
        right = min(left + 8, shape[0])
        kval = torch.sqrt(
            kx[left:right, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
        )
        index = torch.bucketize(kval.reshape(-1), edges) - 1
        valid = (index >= 0) & (index < bins)
        weight = kz_weight[None, None, :].expand(right - left, shape[1], -1).reshape(-1)
        index = index[valid]
        weight = weight[valid]
        local_p = fp[left:right].reshape(-1)[valid]
        local_t = ft[left:right].reshape(-1)[valid]
        values = {
            "count": weight,
            "cross": weight * torch.real(local_p * torch.conj(local_t)).double(),
            "prediction": weight * torch.abs(local_p).square().double(),
            "truth": weight * torch.abs(local_t).square().double(),
        }
        for name, value in values.items():
            sums[name] += torch.bincount(index, weights=value, minlength=bins)
    del fp, ft
    return {name: value.cpu().numpy() for name, value in sums.items()}


def spectra_report(sums: dict[str, np.ndarray], edges: np.ndarray) -> dict:
    count = sums["count"]
    cross = sums["cross"] / np.maximum(count, 1)
    pred = sums["prediction"] / np.maximum(count, 1)
    truth = sums["truth"] / np.maximum(count, 1)
    valid = count > 0
    r = np.full_like(count, np.nan, dtype=np.float64)
    transfer = np.full_like(count, np.nan, dtype=np.float64)
    power_ratio = np.full_like(count, np.nan, dtype=np.float64)
    r[valid] = cross[valid] / np.sqrt(np.maximum(pred[valid] * truth[valid], 1e-300))
    transfer[valid] = cross[valid] / np.maximum(truth[valid], 1e-300)
    power_ratio[valid] = pred[valid] / np.maximum(truth[valid], 1e-300)
    return {
        "k_edges_h_mpc": edges.tolist(),
        "k_centres_h_mpc": np.sqrt(edges[:-1] * edges[1:]).tolist(),
        "mode_count": count.tolist(),
        "cross_correlation_r": r.tolist(),
        "cross_transfer": transfer.tolist(),
        "power_ratio": power_ratio.tolist(),
    }


def eigensystem(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty((len(tensor), 3), dtype=np.float32)
    vectors = np.empty((len(tensor), 3, 3), dtype=np.float32)
    for left in range(0, len(tensor), 250_000):
        right = min(left + 250_000, len(tensor))
        v, q = np.linalg.eigh(np.asarray(tensor[left:right], dtype=np.float64))
        values[left:right] = v
        vectors[left:right] = q
    return values, vectors


def tensor_component_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    result = {}
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        row = RegressionAccumulator()
        row.add(prediction[:, i, j], truth[:, i, j])
        result[f"t{i}{j}"] = row.report()
    return result


def orientation_report(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref_values, ref_vectors = eigensystem(reference)
    _, candidate_vectors = eigensystem(candidate)
    gaps = np.minimum(ref_values[:, 1] - ref_values[:, 0], ref_values[:, 2] - ref_values[:, 1])
    angle = np.rad2deg(np.arccos(np.clip(
        np.abs(np.sum(ref_vectors * candidate_vectors, axis=1)), 0.0, 1.0
    )))
    edges = np.quantile(gaps, (0.0, 0.25, 0.5, 0.75, 1.0))
    bins = {}
    for index in range(4):
        selected = (gaps >= edges[index]) & (
            gaps <= edges[index + 1] if index == 3 else gaps < edges[index + 1]
        )
        bins[str(index)] = {
            "n": int(np.count_nonzero(selected)),
            "eigengap_low": float(edges[index]),
            "eigengap_high": float(edges[index + 1]),
            "median_angle_deg": np.median(angle[selected], axis=0).tolist(),
            "p90_angle_deg": np.quantile(angle[selected], 0.9, axis=0).tolist(),
        }
    return {
        "sign_invariant_axis_angle": True,
        "reference": "true delta_R7 through identical survey window and global tidal solve",
        "bins": bins,
    }


def load_science_fields(component: dict, prediction_path: Path, device: str, taper_mpc: float):
    with h5py.File(component["file"], "r") as target:
        truth = torch.from_numpy(np.asarray(target["delta_r7"], dtype=np.float32)).to(device)
        support = torch.from_numpy(np.asarray(target["science_support"], dtype=np.float32)).to(device)
    with h5py.File(component["source_field"], "r") as source:
        exposure = torch.from_numpy(np.asarray(source["exposure_apodized"], dtype=np.float32)).to(device)
    with h5py.File(prediction_path, "r") as predicted:
        array = np.nan_to_num(np.asarray(predicted["predicted_delta_r7"], dtype=np.float32))
    prediction = torch.from_numpy(array).to(device)
    grid = component["grid"]
    radial = radial_cosine_window(
        tuple(int(value) for value in prediction.shape),
        np.asarray(grid["origin_mpc"], dtype=np.float64),
        float(grid["cell_mpc"]),
        taper_mpc,
        device,
    )
    window = support * exposure * radial
    return prediction * window, truth * window, window


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P8.9 global field/tidal evaluation requires an interactive GPU")
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_lock(
        args.output / ".evaluation.lock",
        purpose="P8.9 stitched field and global tidal evaluation",
    )
    stitched_manifest = json.loads((args.stitched / "stitched_field_manifest.json").read_text())
    target_manifest = json.loads(args.target_manifest.read_text())
    if stitched_manifest.get("double_smoothing_applied") is not False:
        raise RuntimeError("stitched artifact does not certify no double smoothing")
    field = field_metrics(args.stitched, target_manifest)

    assignment = np.load(args.assignment, mmap_mode="r")
    config = json.loads((ROOT / "p8_density_phys_v1/training_contract/rotation_0/d0_config.json").read_text())
    train_folds = tuple(int(value) for value in config["roles"]["train_folds"])
    validation_fold = int(config["roles"]["validation_fold"])
    auth = authoritative_mask(assignment)
    fold = np.asarray(assignment["fold"], dtype=np.int8)
    active_rows = np.flatnonzero(auth & np.isin(fold, (*train_folds, validation_fold)))
    parent = np.asarray(assignment["parent_node_id"][active_rows], dtype=np.int64)
    cap = np.asarray(assignment["cap"][active_rows], dtype=np.int8)
    train = np.isin(fold[active_rows], train_folds)
    validation = fold[active_rows] == validation_fold
    catalogue = read_rows(args.catalogue, parent, ["TARGETID", "RA", "DEC", "Z"])
    joined = join_target_truth(
        args.target_input, np.asarray(catalogue["TARGETID"], dtype=np.int64),
        chunk_rows=args.target_chunk,
    )
    positions = {
        "z_cosmo_oracle": sky_to_observer_mpc(
            np.asarray(catalogue["RA"], dtype=np.float64),
            np.asarray(catalogue["DEC"], dtype=np.float64),
            joined["Z_COSMO"],
        ),
        "z_observed_deployable": sky_to_observer_mpc(
            np.asarray(catalogue["RA"], dtype=np.float64),
            np.asarray(catalogue["DEC"], dtype=np.float64),
            np.asarray(catalogue["Z"], dtype=np.float64),
        ),
    }
    predicted_tensor = {
        name: np.empty((len(parent), 3, 3), dtype=np.float32) for name in positions
    }
    reference_tensor = np.empty((len(parent), 3, 3), dtype=np.float32)
    edges = np.geomspace(0.002, 1.0, args.spectral_bins + 1)
    spectral_total = {
        name: np.zeros(args.spectral_bins, dtype=np.float64)
        for name in ("count", "cross", "prediction", "truth")
    }
    spectral_caps = {}
    fft_caps = {}
    for cap_id, cap_name in CAP_NAME.items():
        selected = cap == cap_id
        component = target_manifest["components"][cap_name]
        grid = component["grid"]
        predicted_field, truth_field, window = load_science_fields(
            component, predicted_path(args.stitched, cap_name), args.device,
            args.radial_taper_mpc,
        )
        weight = torch.sum(window)
        pred_mean = torch.sum(predicted_field) / torch.clamp(weight, min=1e-30)
        truth_mean = torch.sum(truth_field) / torch.clamp(weight, min=1e-30)
        pred_spectrum_field = (predicted_field - pred_mean * window) 
        truth_spectrum_field = (truth_field - truth_mean * window)
        sums = spectral_sums(
            pred_spectrum_field, truth_spectrum_field,
            cell_mpc=float(grid["cell_mpc"]), edges_h_mpc=edges,
        )
        spectral_caps[cap_name] = spectra_report(sums, edges)
        for name in spectral_total:
            spectral_total[name] += sums[name]
        sampled_pred, pred_fft = solve_tensors_at_positions(
            predicted_field,
            positions={name: value[selected] for name, value in positions.items()},
            origin_mpc=np.asarray(grid["origin_mpc"], dtype=np.float64),
            cell_mpc=float(grid["cell_mpc"]),
            padding_voxels=args.padding_voxels,
        )
        sampled_truth, truth_fft = solve_tensors_at_positions(
            truth_field,
            positions={"z_cosmo_oracle": positions["z_cosmo_oracle"][selected]},
            origin_mpc=np.asarray(grid["origin_mpc"], dtype=np.float64),
            cell_mpc=float(grid["cell_mpc"]),
            padding_voxels=args.padding_voxels,
        )
        for name in positions:
            predicted_tensor[name][selected] = sampled_pred[name]
        reference_tensor[selected] = sampled_truth["z_cosmo_oracle"]
        fft_caps[cap_name] = {
            "predicted": pred_fft,
            "reference": truth_fft,
            "window_nonzero_voxels": int(torch.count_nonzero(window).item()),
        }
        del predicted_field, truth_field, window, pred_spectrum_field, truth_spectrum_field
        torch.cuda.empty_cache()

    truth_by_parent = np.load(args.parent_truth, mmap_mode="r")
    truth_active = np.asarray(truth_by_parent[parent], dtype=np.float64)
    coordinate_reports = {}
    eigenvalues = {}
    for name, tensor in predicted_tensor.items():
        values, _ = eigensystem(tensor)
        eigenvalues[name] = values
        calibrated, affine = fit_affine_on_training(values, truth_active, train)
        coordinate_reports[name] = {
            "raw_physical": evaluate_complete_fold(
                parent_node_id=parent[validation],
                predicted_eigenvalues=values[validation],
                truth_by_parent=truth_by_parent,
                assignment=assignment,
                validation_fold=validation_fold,
            ),
            "train_fold_affine_diagnostic": evaluate_complete_fold(
                parent_node_id=parent[validation],
                predicted_eigenvalues=calibrated[validation],
                truth_by_parent=truth_by_parent,
                assignment=assignment,
                validation_fold=validation_fold,
            ),
            "affine": affine,
        }
        np.savez_compressed(
            args.output / f"validation_{name}_predictions.npz",
            parent_node_id=parent[validation],
            raw_physical_eigenvalues=values[validation],
            train_fold_affine_eigenvalues=calibrated[validation].astype(np.float32),
        )

    tensor_metrics = tensor_component_metrics(
        predicted_tensor["z_cosmo_oracle"][validation], reference_tensor[validation]
    )
    orientation = orientation_report(
        reference_tensor[validation], predicted_tensor["z_cosmo_oracle"][validation]
    )
    report = {
        "schema_version": "p8-density-field-downstream-evaluation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_sha(),
        "status": "PASS",
        "model": "U-DENSITY-PHYS-v1",
        "rotation": 0,
        "seed": 42,
        "field_metrics": field,
        "spectra": {
            "window": "science support times P3 apodized exposure times 100-Mpc radial taper",
            "mean_subtraction": "separate window-weighted mean for truth and prediction",
            "pooled_caps": spectra_report(spectral_total, edges),
            "by_cap": spectral_caps,
        },
        "tidal": {
            "operator": "one global unsmoothed k_i k_j/k^2 solve per cap; input already R=7",
            "padding_voxels": int(args.padding_voxels),
            "double_smoothing_applied": False,
            "coordinates": coordinate_reports,
            "predicted_vs_windowed_true_tensor_components_z_cosmo": tensor_metrics,
            "orientation_z_cosmo": orientation,
            "fft_by_cap": fft_caps,
        },
        "inputs": {
            "stitched_manifest": str(args.stitched / "stitched_field_manifest.json"),
            "stitched_manifest_sha256": sha256(args.stitched / "stitched_field_manifest.json"),
            "target_manifest": str(args.target_manifest),
            "target_manifest_sha256": sha256(args.target_manifest),
            "assignment": str(args.assignment),
            "assignment_sha256": sha256(args.assignment),
        },
        "exact_predictions": {
            name: str(args.output / f"validation_{name}_predictions.npz") for name in positions
        },
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_json(args.output / "field_downstream_metrics.json", report)
    (args.output / "D0_FIELD_DOWNSTREAM_EVALUATED").write_text(
        "raw physical and train-fold-only affine rows complete\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    run_lock.close()


if __name__ == "__main__":
    main()
