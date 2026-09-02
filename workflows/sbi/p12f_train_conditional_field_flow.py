#!/usr/bin/env python3
"""Bounded conditional flow-matching canary for coherent ``delta_R7`` patches.

This is the first *proper field-posterior* experiment in the roadmap.  It learns
``p(delta_R7 | V_final, random response, H_fid)`` from visible Abacus phases.
The held-out ph006 truth is used only for the frozen canary report, while ph001
is rejected by both the data builder and this trainer.

The posterior mean R2 is diagnostic only.  Promotion is decided by whether the
held-out truth behaves like a draw from the conditional ensemble in voxel,
Fourier-mode, and fixed-physics-derived eigenvalue coordinates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from astropy.cosmology import Planck18
import h5py
import numpy as np
import torch
import torch.nn as nn

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_train_unet_patch import (
    CHANNELS,
    UNet3D,
    model_inputs,
)
from workflows.sbi.p12f_field_posterior_diagnostics import (
    conditional_reports,
    crps_ensemble,
    fixed_tidal_eigenvalues,
    physics_closure_report,
    quantile_labels,
    scalar_posterior_report,
    standard_normal_rank_reference,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f_conditional_field_flow_v1.json"
DEFAULT_CONTRACT = Path(
    "/global/homes/d/dkololgi/p11_contracts/"
    "training_contract_r1_random_repair_v2_20260901"
)
DEFAULT_PHASE_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
DEFAULT_OUTPUT = DEFAULT_PHASE_ROOT / "p12f_conditional_field_flow_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract-root", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--phase-root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="cfm_canary_seed42_v1")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class ConditionalVelocityUNet(nn.Module):
    """Velocity field ``v(x_t,t,condition)`` for conditional rectified flow."""

    def __init__(self, *, condition_channels: int = 3, base: int = 8):
        super().__init__()
        self.condition_channels = int(condition_channels)
        # noised field + condition + broadcast time
        self.net = UNet3D(
            in_channels=1 + self.condition_channels + 1,
            latent_channels=1,
            base=int(base),
        )

    def forward(
        self, state: torch.Tensor, time_value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if state.ndim != 5 or state.shape[1] != 1:
            raise ValueError("state must have shape [batch,1,nx,ny,nz]")
        if condition.shape[0] != state.shape[0] or condition.shape[2:] != state.shape[2:]:
            raise ValueError("condition/state geometry mismatch")
        if condition.shape[1] != self.condition_channels:
            raise ValueError("unexpected condition channel count")
        t = torch.as_tensor(time_value, device=state.device, dtype=state.dtype)
        if t.ndim == 0:
            t = t.repeat(state.shape[0])
        if t.shape != (state.shape[0],):
            raise ValueError("time must be scalar or one value per batch")
        time_channel = t.view(-1, 1, 1, 1, 1).expand(
            -1, 1, *state.shape[2:]
        )
        return self.net(torch.cat((state, condition, time_channel), dim=1))


def rectified_flow_training_pair(
    target: torch.Tensor, *, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.ndim != 5 or target.shape[1] != 1:
        raise ValueError("target must have shape [batch,1,nx,ny,nz]")
    noise = torch.randn(
        target.shape, device=target.device, dtype=target.dtype, generator=generator
    )
    t = torch.rand(
        target.shape[0], device=target.device, dtype=target.dtype, generator=generator
    )
    blend = t.view(-1, 1, 1, 1, 1)
    state = (1.0 - blend) * noise + blend * target
    velocity = target - noise
    return state, t, velocity, noise


@torch.inference_mode()
def sample_heun(
    model: ConditionalVelocityUNet,
    condition: torch.Tensor,
    *,
    draws: int,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if draws <= 0 or steps <= 0:
        raise ValueError("draw and ODE-step counts must be positive")
    condition_batch = condition.expand(draws, -1, -1, -1, -1)
    state = torch.randn(
        (draws, 1, *condition.shape[2:]),
        device=condition.device,
        dtype=condition.dtype,
        generator=generator,
    )
    dt = 1.0 / steps
    for index in range(steps):
        t0 = torch.full(
            (draws,), index / steps, device=state.device, dtype=state.dtype
        )
        velocity0 = model(state, t0, condition_batch)
        proposal = state + dt * velocity0
        t1 = torch.full(
            (draws,), (index + 1) / steps, device=state.device, dtype=state.dtype
        )
        velocity1 = model(proposal, t1, condition_batch)
        state = state + 0.5 * dt * (velocity0 + velocity1)
    return state[:, 0]


class FieldTargetStore:
    def __init__(self, phase_root: Path, phases: tuple[str, ...]):
        self.phase_root = Path(phase_root)
        self.manifests = {}
        self.handles: dict[tuple[str, int], h5py.File] = {}
        self.response_handles: dict[tuple[str, int], h5py.File] = {}
        for phase in phases:
            if phase == "ph001":
                raise PermissionError("ph001 is sealed for P12-F")
            path = self.phase_root / phase / "p12f_field_targets_v1/FIELD_TARGET_READY.json"
            payload = json.loads(path.read_text())
            if (
                not payload.get("pass")
                or payload.get("phase") != phase
                or payload.get("ph001_opened")
            ):
                raise RuntimeError(f"{phase}: field target marker does not pass")
            self.manifests[phase] = payload

    def close(self) -> None:
        for handle in (*self.handles.values(), *self.response_handles.values()):
            handle.close()
        self.handles.clear()
        self.response_handles.clear()

    def _component(self, phase: str, cap: int) -> dict:
        return self.manifests[phase]["components"]["NGC" if int(cap) == 1 else "SGC"]

    def target_handle(self, phase: str, cap: int) -> h5py.File:
        key = (phase, int(cap))
        if key not in self.handles:
            self.handles[key] = h5py.File(self._component(phase, cap)["file"], "r")
        return self.handles[key]

    def response_handle(self, phase: str, cap: int) -> h5py.File:
        key = (phase, int(cap))
        if key not in self.response_handles:
            self.response_handles[key] = h5py.File(
                self._component(phase, cap)["support_random_source"], "r"
            )
        return self.response_handles[key]

    def grid(self, phase: str, cap: int) -> dict:
        return self._component(phase, cap)["grid"]

    def extract(self, phase: str, patch) -> dict[str, np.ndarray]:
        selection = tuple(
            slice(int(left), int(right))
            for left, right in zip(patch.context_start, patch.context_stop)
        )
        target = self.target_handle(phase, patch.cap)
        response = self.response_handle(phase, patch.cap)
        output = {
            "delta": np.asarray(target["delta_r7"][selection], dtype=np.float32),
            "support": np.asarray(response["support_random"][selection], dtype=bool),
            "angular_response": np.asarray(
                response["angular_response"][selection], dtype=np.float32
            ),
            "boundary_distance": np.asarray(
                response["distance_to_support_boundary"][selection], dtype=np.float32
            ),
        }
        if output["delta"].shape != patch.values.shape[1:]:
            raise RuntimeError("target/conditioning patch geometry mismatch")
        return output


def deterministic_subset(values: np.ndarray, count: int, *, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if count <= 0 or count > len(values):
        raise ValueError("requested subset size lies outside source population")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(values, size=count, replace=False))


def selected_core_contract(loader, config: dict) -> dict[str, np.ndarray]:
    seed = int(config["canary"]["seed"])
    count = int(config["canary"]["training_cores_per_phase"])
    output = {}
    for index, phase in enumerate(config["roles"]["training"]):
        source = np.load(
            loader.root / "phases" / phase / "training_core_id.npy", mmap_mode="r"
        )
        output[phase] = deterministic_subset(source, count, seed=seed + 101 * index)
    source = np.load(
        loader.root / "phases" / loader.validation_phase / "validation_core_id.npy",
        mmap_mode="r",
    )
    output[loader.validation_phase] = deterministic_subset(
        source, int(config["canary"]["validation_cores"]), seed=seed + 999
    )
    return output


def _core_slice(values: np.ndarray, core_slice: tuple[slice, slice, slice]) -> np.ndarray:
    return values[core_slice]


def fit_target_scaler(loader, store, core_ids: dict[str, np.ndarray], config: dict) -> dict:
    total = 0
    sum_value = 0.0
    sum_square = 0.0
    halo = int(config["patch"]["context_halo_voxels"])
    alignment = int(config["patch"]["alignment_voxels"])
    for phase in config["roles"]["training"]:
        adapter = loader.field_adapter(phase)
        for core_id in core_ids[phase]:
            patch = adapter.extract(
                int(core_id), halo, CHANNELS, alignment_voxels=alignment
            )
            core = _core_slice(store.extract(phase, patch)["delta"], patch.core_slice)
            total += core.size
            sum_value += float(core.sum(dtype=np.float64))
            sum_square += float(np.square(core, dtype=np.float64).sum())
    mean = sum_value / total
    variance = max(sum_square / total - mean * mean, 1e-12)
    return {
        "fit_scope": "selected training-phase authoritative core voxels only",
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "voxels": int(total),
        "phases": list(config["roles"]["training"]),
    }


def target_tensor(values: np.ndarray, scaler: dict, device: str) -> torch.Tensor:
    scaled = (np.asarray(values, dtype=np.float32) - np.float32(scaler["mean"])) / np.float32(
        scaler["std"]
    )
    return torch.from_numpy(scaled[None, None]).to(device)


def unscale(values: np.ndarray, scaler: dict) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * np.float32(scaler["std"]) + np.float32(
        scaler["mean"]
    )


def _voxel_radius_and_shell(grid: dict, patch) -> tuple[np.ndarray, np.ndarray]:
    shape = patch.values.shape[1:]
    axes = [
        float(grid["origin_mpc"][axis])
        + (np.arange(shape[axis]) + int(patch.context_start[axis]) + 0.5)
        * float(grid["cell_mpc"])
        for axis in range(3)
    ]
    radius = np.sqrt(
        axes[0][:, None, None] ** 2
        + axes[1][None, :, None] ** 2
        + axes[2][None, None, :] ** 2
    )
    bounds = np.asarray(
        [Planck18.comoving_distance(z).value for z in (0.15, 0.25, 0.35, 0.45, 0.55)]
    )
    return radius, np.digitize(radius, bounds[1:-1], right=False).astype(np.int8)


def _fourier_parts(samples: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_k = np.fft.rfftn(samples, axes=(-3, -2, -1), norm="ortho")
    truth_k = np.fft.rfftn(truth, axes=(-3, -2, -1), norm="ortho")
    shape = truth.shape
    kx = np.fft.fftfreq(shape[0])[:, None, None]
    ky = np.fft.fftfreq(shape[1])[None, :, None]
    kz = np.fft.rfftfreq(shape[2])[None, None, :]
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    selected = kmag > 0
    stacked_samples = np.concatenate(
        (sample_k.real[:, selected], sample_k.imag[:, selected]), axis=1
    ).astype(np.float32)
    stacked_truth = np.concatenate(
        (truth_k.real[selected], truth_k.imag[selected]), axis=0
    ).astype(np.float32)
    stacked_k = np.concatenate((kmag[selected], kmag[selected]), axis=0).astype(np.float32)
    return stacked_samples, stacked_truth, stacked_k


def evaluate(
    model,
    loader,
    store,
    core_ids,
    scaler,
    config,
    device,
) -> tuple[dict, dict[str, np.ndarray]]:
    phase = config["roles"]["validation_and_selection"]
    adapter = loader.field_adapter(phase)
    normalization = loader.field_normalization
    halo = int(config["patch"]["context_halo_voxels"])
    alignment = int(config["patch"]["alignment_voxels"])
    draws = int(config["canary"]["posterior_draws"])
    ode_steps = int(config["canary"]["ode_steps"])
    max_voxels = int(config["canary"]["maximum_saved_voxels_per_core"])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["canary"]["seed"]) + 12000)
    model.eval()

    voxel_samples, voxel_truth = [], []
    lambda_samples = [[], [], []]
    lambda_truth = [[], [], []]
    shell_parts, response_parts, boundary_parts, environment_parts = [], [], [], []
    fourier_samples, fourier_truth, fourier_k = [], [], []
    patch_ids = []
    physics_rows = []

    for core_ordinal, core_id in enumerate(core_ids):
        patch = adapter.extract(
            int(core_id), halo, CHANNELS, alignment_voxels=alignment
        )
        condition, _ = model_inputs(patch, normalization, device)
        target_data = store.extract(phase, patch)
        target_scaled = target_tensor(target_data["delta"], scaler, device)
        sampled_scaled = sample_heun(
            model,
            condition,
            draws=draws,
            steps=ode_steps,
            generator=generator,
        )
        samples = unscale(sampled_scaled.cpu().numpy(), scaler)
        truth = target_data["delta"]

        core = patch.core_slice
        support_core = _core_slice(target_data["support"], core)
        valid = np.flatnonzero(support_core.ravel())
        if len(valid) == 0:
            raise RuntimeError(f"ph006 core {core_id} has no exact random support")
        if len(valid) > max_voxels:
            valid = valid[np.linspace(0, len(valid) - 1, max_voxels, dtype=np.int64)]
        sample_core = samples[(slice(None),) + core].reshape(draws, -1)[:, valid]
        truth_core = truth[core].reshape(-1)[valid]
        voxel_samples.append(sample_core)
        voxel_truth.append(truth_core)
        patch_ids.append(np.full(len(valid), int(core_id), dtype=np.int64))

        radius, shell = _voxel_radius_and_shell(store.grid(phase, patch.cap), patch)
        shell_parts.append(shell[core].reshape(-1)[valid])
        response_parts.append(
            target_data["angular_response"][core].reshape(-1)[valid]
        )
        boundary_parts.append(
            target_data["boundary_distance"][core].reshape(-1)[valid]
        )
        environment_parts.append(truth_core)

        sample_eigen = fixed_tidal_eigenvalues(
            torch.from_numpy(samples).to(device)
        ).cpu().numpy()
        truth_tensor = torch.from_numpy(truth).to(device)
        truth_eigen = fixed_tidal_eigenvalues(truth_tensor).cpu().numpy()
        physics_rows.append(physics_closure_report(truth_tensor))
        sample_eigen_core = sample_eigen[(slice(None),) + core + (slice(None),)]
        truth_eigen_core = truth_eigen[core + (slice(None),)]
        for index in range(3):
            lambda_samples[index].append(
                sample_eigen_core[..., index].reshape(draws, -1)[:, valid]
            )
            lambda_truth[index].append(
                truth_eigen_core[..., index].reshape(-1)[valid]
            )

        fs, ft, fk = _fourier_parts(samples[(slice(None),) + core], truth[core])
        fourier_samples.append(fs)
        fourier_truth.append(ft)
        fourier_k.append(fk)
        print(
            f"validation {core_ordinal + 1}/{len(core_ids)} core={int(core_id)}",
            flush=True,
        )

    samples_all = np.concatenate(voxel_samples, axis=1)
    truth_all = np.concatenate(voxel_truth)
    shell_all = np.concatenate(shell_parts)
    response_all = np.concatenate(response_parts)
    boundary_all = np.concatenate(boundary_parts)
    environment_all = np.concatenate(environment_parts)
    patch_all = np.concatenate(patch_ids)
    lambda_samples_all = [np.concatenate(parts, axis=1) for parts in lambda_samples]
    lambda_truth_all = [np.concatenate(parts) for parts in lambda_truth]
    fourier_samples_all = np.concatenate(fourier_samples, axis=1)
    fourier_truth_all = np.concatenate(fourier_truth)
    fourier_k_all = np.concatenate(fourier_k)
    seed = int(config["canary"]["seed"])

    voxel_report = scalar_posterior_report(samples_all, truth_all, seed=seed + 1)
    voxel_report["crps_ensemble"] = crps_ensemble(samples_all, truth_all)
    voxel_report["positive_width_fraction"] = float(
        np.mean(np.std(samples_all, axis=0, ddof=1) > 1e-6)
    )
    lambda_report = {
        f"lambda{index + 1}": scalar_posterior_report(
            lambda_samples_all[index], lambda_truth_all[index], seed=seed + 10 + index
        )
        for index in range(3)
    }
    k_labels = quantile_labels(fourier_k_all, bins=4)
    report = {
        "schema_version": "p12f-conditional-field-flow-canary-report-v1",
        "created_utc": utc_now(),
        "phase": phase,
        "validation_cores": [int(value) for value in core_ids],
        "posterior_draws": draws,
        "ode_steps": ode_steps,
        "voxel": voxel_report,
        "fourier_modes": {
            "pooled_real_and_imaginary": scalar_posterior_report(
                fourier_samples_all, fourier_truth_all, seed=seed + 20
            ),
            "by_k_quartile": conditional_reports(
                fourier_samples_all,
                fourier_truth_all,
                k_labels,
                seed=seed + 21,
            ),
            "window": "none; bounded local-periodic patch diagnostic",
        },
        "derived_local_tidal_eigenvalues": lambda_report,
        "conditional_voxel_coverage": {
            "redshift_shell": conditional_reports(
                samples_all, truth_all, shell_all, seed=seed + 30
            ),
            "angular_response_quartile": conditional_reports(
                samples_all,
                truth_all,
                quantile_labels(response_all),
                seed=seed + 31,
            ),
            "boundary_distance_quartile": conditional_reports(
                samples_all,
                truth_all,
                quantile_labels(boundary_all),
                seed=seed + 32,
            ),
            "true_environment_quartile": conditional_reports(
                samples_all,
                truth_all,
                quantile_labels(environment_all),
                seed=seed + 33,
            ),
        },
        "physics_closure": {
            "maximum_trace_max_abs": float(max(row["trace_max_abs"] for row in physics_rows)),
            "maximum_trace_rmse": float(max(row["trace_rmse"] for row in physics_rows)),
            "all_finite": bool(all(row["all_finite"] for row in physics_rows)),
            "all_ordered": bool(all(row["ordered"] for row in physics_rows)),
            "additional_gaussian_smoothing": False,
        },
        "rank_reference": standard_normal_rank_reference(draws),
        "spatial_dependence_warning": (
            "voxel/mode rows are correlated; this bounded canary does not turn them "
            "into independent calibration trials"
        ),
        "posterior_predictive_reobservation": {
            "implemented": False,
            "reason": (
                "requires a frozen stochastic HOD plus fibre/redshift observation "
                "operator; a Poisson counts shortcut is not accepted as DESI closure"
            ),
        },
        "ph001_opened": False,
    }
    arrays = {
        "voxel_samples": samples_all.astype(np.float32),
        "voxel_truth": truth_all.astype(np.float32),
        "patch_core_id": patch_all,
        "redshift_shell": shell_all,
        "angular_response": response_all.astype(np.float32),
        "boundary_distance_mpc": boundary_all.astype(np.float32),
        "true_environment_delta_r7": environment_all.astype(np.float32),
        "lambda_samples": np.stack(lambda_samples_all, axis=-1).astype(np.float32),
        "lambda_truth": np.stack(lambda_truth_all, axis=-1).astype(np.float32),
        "fourier_samples": fourier_samples_all.astype(np.float32),
        "fourier_truth": fourier_truth_all.astype(np.float32),
        "fourier_k": fourier_k_all.astype(np.float32),
    }
    return report, arrays


def scientific_pass(report: dict, config: dict) -> tuple[bool, list[str]]:
    gate = config["primary_gates"]
    reasons = []
    voxel = report["voxel"]
    if voxel["coverage"]["0.68"]["absolute_error"] > float(
        gate["voxel_68_coverage_absolute_error_maximum"]
    ):
        reasons.append("voxel 68-percent coverage")
    if voxel["rank_cdf_maximum_deviation"] > float(
        gate["voxel_rank_cdf_maximum_deviation"]
    ):
        reasons.append("voxel rank-CDF deviation")
    if voxel["positive_width_fraction"] < float(
        gate["posterior_width_positive_fraction_minimum"]
    ):
        reasons.append("degenerate voxel posterior widths")
    for name, row in report["derived_local_tidal_eigenvalues"].items():
        if row["coverage"]["0.68"]["absolute_error"] > float(
            gate["lambda_68_coverage_absolute_error_maximum"]
        ):
            reasons.append(f"{name} 68-percent coverage")
    if not report["physics_closure"]["all_finite"] or not report["physics_closure"][
        "all_ordered"
    ]:
        reasons.append("fixed-physics closure")
    return len(reasons) == 0, reasons


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P12-F canary requires a CUDA interactive allocation")
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != "p12f-conditional-field-flow-canary-v1":
        raise RuntimeError("unsupported P12-F config")
    if config["roles"]["sealed_blind_test"] != "ph001":
        raise RuntimeError("P12-F blind phase contract changed")
    visible = tuple(config["roles"]["training"]) + (
        config["roles"]["validation_and_selection"],
    )
    if "ph001" in visible:
        raise PermissionError("ph001 entered a visible P12-F role")
    output = args.output_root / args.run_name
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"non-empty P12-F run requires a new run name: {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(config["canary"]["seed"]))
    np.random.seed(int(config["canary"]["seed"]))
    loader = P10RandomResponseLoader(args.contract_root, include_blind=False)
    if tuple(loader.training_phases) != tuple(config["roles"]["training"]):
        raise RuntimeError("training phase contract mismatch")
    if loader.validation_phase != config["roles"]["validation_and_selection"]:
        raise RuntimeError("validation phase contract mismatch")
    selected = selected_core_contract(loader, config)
    store = FieldTargetStore(args.phase_root, visible)
    scaler = fit_target_scaler(loader, store, selected, config)
    atomic_json(output / "target_scaler.json", scaler)

    source_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "workflows/sbi/p12f_field_posterior_diagnostics.py",
        REPO_ROOT / "workflows/abacus_tweb/p12f_build_field_targets.py",
        args.config.resolve(),
    ]
    source_hashes = {
        str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path): sha256(path)
        for path in source_paths
    }
    target_markers = {
        phase: {
            "path": str(
                (args.phase_root / phase / "p12f_field_targets_v1/FIELD_TARGET_READY.json").resolve()
            ),
            "sha256": sha256(
                args.phase_root / phase / "p12f_field_targets_v1/FIELD_TARGET_READY.json"
            ),
        }
        for phase in visible
    }
    run_manifest = {
        "schema_version": "p12f-conditional-field-flow-run-v1",
        "created_utc": utc_now(),
        "git_revision_at_launch": git_revision(),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "source_hashes": source_hashes,
        "contract_root": str(args.contract_root.resolve()),
        "training_ready_sha256": sha256(args.contract_root / "TRAINING_LOADER_READY.json"),
        "field_target_markers": target_markers,
        "selected_core_ids": {
            phase: [int(value) for value in values] for phase, values in selected.items()
        },
        "target_scaler": scaler,
        "ph001_opened": False,
    }
    atomic_json(output / "run_manifest.json", run_manifest)

    model = ConditionalVelocityUNet(
        condition_channels=3, base=int(config["model"]["unet_base"])
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["canary"]["learning_rate"]),
        weight_decay=float(config["canary"]["weight_decay"]),
    )
    refs = [
        (phase, int(core_id))
        for phase in config["roles"]["training"]
        for core_id in selected[phase]
    ]
    rng = np.random.default_rng(int(config["canary"]["seed"]) + 440)
    halo = int(config["patch"]["context_halo_voxels"])
    alignment = int(config["patch"]["alignment_voxels"])
    normalization = loader.field_normalization
    updates = int(config["canary"]["updates"])
    trace_path = output / "loss_trace.jsonl"
    trace_path.write_text("")
    started = time.monotonic()
    finite = True
    last_gradient_norm = None

    for update in range(1, updates + 1):
        if (update - 1) % len(refs) == 0:
            rng.shuffle(refs)
        phase, core_id = refs[(update - 1) % len(refs)]
        adapter = loader.field_adapter(phase)
        patch = adapter.extract(core_id, halo, CHANNELS, alignment_voxels=alignment)
        condition, _ = model_inputs(patch, normalization, args.device)
        target = target_tensor(store.extract(phase, patch)["delta"], scaler, args.device)
        state, t, target_velocity, _ = rectified_flow_training_pair(target)
        prediction = model(state, t, condition)
        core = (slice(None), slice(None)) + patch.core_slice
        loss = torch.mean(torch.square(prediction[core] - target_velocity[core]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["canary"]["gradient_clip"])
            ).detach().cpu()
        )
        optimizer.step()
        finite = finite and bool(
            torch.isfinite(loss)
            and np.isfinite(last_gradient_norm)
            and all(torch.isfinite(parameter).all() for parameter in model.parameters())
        )
        if not finite:
            raise RuntimeError(f"non-finite P12-F state at update {update}")
        if update == 1 or update % 25 == 0 or update == updates:
            row = {
                "update": update,
                "phase": phase,
                "core_id": core_id,
                "flow_matching_loss": float(loss.detach().cpu()),
                "preclip_gradient_norm": last_gradient_norm,
                "elapsed_seconds": time.monotonic() - started,
            }
            with trace_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

    checkpoint = {
        "schema_version": "p12f-conditional-field-flow-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "updates": updates,
        "config_sha256": sha256(args.config),
        "source_hashes": source_hashes,
        "target_scaler": scaler,
        "ph001_opened": False,
    }
    checkpoint_path = output / "checkpoint.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    reload_model = ConditionalVelocityUNet(
        condition_channels=3, base=int(config["model"]["unet_base"])
    ).to(args.device)
    reload_model.load_state_dict(reloaded["model"])
    checkpoint_reload = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), reload_model.state_dict().values())
    )
    if not checkpoint_reload:
        raise RuntimeError("P12-F checkpoint reload parity failed")

    report, arrays = evaluate(
        reload_model,
        loader,
        store,
        selected[loader.validation_phase],
        scaler,
        config,
        args.device,
    )
    passed, reasons = scientific_pass(report, config)
    report.update(
        {
            "technical": {
                "updates": updates,
                "all_finite": finite,
                "checkpoint_reload": checkpoint_reload,
                "last_preclip_gradient_norm": last_gradient_norm,
                "elapsed_seconds": time.monotonic() - started,
            },
            "primary_gate_pass": bool(passed),
            "primary_gate_failure_reasons": reasons,
            "interpretation": (
                "A pass licenses a larger field-posterior experiment, not production. "
                "It does not establish global-mode coherence, DESI closure, HOD "
                "marginalization, or blind-phase calibration."
            ),
        }
    )
    arrays_path = output / "ph006_posterior_samples.npz"
    np.savez_compressed(arrays_path, **arrays)
    report["posterior_samples"] = str(arrays_path.resolve())
    report["posterior_samples_sha256"] = sha256(arrays_path)
    atomic_json(output / "P12F_CANARY_REPORT.json", report)
    marker = {
        "schema_version": "p12f-conditional-field-flow-canary-complete-v1",
        "created_utc": utc_now(),
        "technical_pass": bool(finite and checkpoint_reload),
        "scientific_canary_pass": bool(passed),
        "failure_reasons": reasons,
        "report": str((output / "P12F_CANARY_REPORT.json").resolve()),
        "report_sha256": sha256(output / "P12F_CANARY_REPORT.json"),
        "ph001_opened": False,
    }
    atomic_json(output / "P12F_CANARY_COMPLETE.json", marker)
    store.close()
    loader.close()
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
