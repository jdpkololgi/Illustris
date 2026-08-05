#!/usr/bin/env python3
"""Bounded U-PATCH correction around the frozen full-cap CIC solution.

Checkpoint zero reproduces the train-affine CIC eigenvalues exactly (up to the
registered positive eigengap floor).  The local U-Net predicts only an additive
lambda1 correction and multiplicative positive eigengap corrections.  This is
the first *trained* P8 classical-plus-learned corrective model; the earlier P9
linear blends were diagnostics, not residual training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from workflows.abacus_tweb import p8_train_unet_patch as unet_impl
from workflows.abacus_tweb.p8_deterministic_common import linear_increments


GAP_FLOOR = 1.0e-6


def load_cic_anchor(
    p8_root: Path,
    rotation: int,
    n_parent: int,
) -> tuple[np.ndarray, dict]:
    """Load a parent-ID keyed dense CIC anchor for registered active folds."""
    directory = p8_root / "classical" / f"rotation_{rotation}"
    parent_path = directory / "active_parent_node_id.npy"
    eigen_path = directory / "cic_train_affine_active_eigenvalues.npy"
    if not parent_path.is_file() or not eigen_path.is_file():
        raise RuntimeError(
            "CIC residual anchor is absent; rerun p8_classical_fullcap.py with "
            "the frozen rotation before residual training"
        )
    parent = np.load(parent_path, mmap_mode="r")
    eigen = np.load(eigen_path, mmap_mode="r")
    if len(parent) != len(eigen) or len(np.unique(parent)) != len(parent):
        raise RuntimeError("CIC residual anchor parent IDs are not one-to-one")
    if np.any(parent < 0) or np.any(parent >= n_parent):
        raise RuntimeError("CIC residual anchor contains invalid parent IDs")
    if not np.all(np.isfinite(eigen)):
        raise RuntimeError("CIC residual anchor contains non-finite eigenvalues")
    gaps = np.diff(np.asarray(eigen), axis=1)
    violations = int(np.sum(np.any(gaps <= 0, axis=1)))
    if violations:
        raise RuntimeError(f"CIC residual anchor has {violations} unordered rows")
    dense = np.full((n_parent, 3), np.nan, dtype=np.float32)
    dense[np.asarray(parent, dtype=np.int64)] = np.asarray(eigen, dtype=np.float32)
    report = {
        "parent_path": str(parent_path),
        "eigenvalue_path": str(eigen_path),
        "active_rows": int(len(parent)),
        "ordered_rows": int(len(parent)),
        "minimum_gap12": float(gaps[:, 0].min()),
        "minimum_gap23": float(gaps[:, 1].min()),
        "gap_floor": GAP_FLOOR,
        "floor_affected_rows": int(np.sum(np.any(gaps < GAP_FLOOR, axis=1))),
    }
    return dense, report


class UCICResidual(nn.Module):
    """Selection-aware local U-Net residual with an exact CIC null model."""

    def __init__(
        self,
        scaler: dict,
        *,
        base: int = 24,
        latent_channels: int = 32,
        head_width: int = 128,
        gap_floor: float = GAP_FLOOR,
        lambda1_max_sigma: float = 1.0,
        log_gap_max: float = 1.5,
    ):
        super().__init__()
        self.unet = unet_impl.UNet3D(3, latent_channels, base)
        self.head = nn.Sequential(
            nn.Linear(latent_channels + 3, head_width),
            nn.SiLU(),
            nn.Linear(head_width, head_width),
            nn.SiLU(),
            nn.Linear(head_width, 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer(
            "target_mean", torch.as_tensor(scaler["mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "target_std", torch.as_tensor(scaler["std"], dtype=torch.float32)
        )
        self.gap_floor = float(gap_floor)
        self.lambda1_max_sigma = float(lambda1_max_sigma)
        self.log_gap_max = float(log_gap_max)

    def forward(
        self,
        values: torch.Tensor,
        points: torch.Tensor,
        cic_eigenvalues: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.unet(values)
        sampled = F.grid_sample(
            latent, points, mode="bilinear", align_corners=True, padding_mode="border"
        )[0, :, 0, 0].T
        cic_increment = torch.stack(
            (
                cic_eigenvalues[:, 0],
                cic_eigenvalues[:, 1] - cic_eigenvalues[:, 0],
                cic_eigenvalues[:, 2] - cic_eigenvalues[:, 1],
            ),
            dim=1,
        )
        cic_scaled = (cic_increment - self.target_mean) / self.target_std
        raw_correction = self.head(torch.cat((sampled, cic_scaled), dim=1))
        correction = torch.stack(
            (
                self.lambda1_max_sigma * torch.tanh(raw_correction[:, 0]),
                self.log_gap_max * torch.tanh(raw_correction[:, 1]),
                self.log_gap_max * torch.tanh(raw_correction[:, 2]),
            ),
            dim=1,
        )
        lambda1 = cic_eigenvalues[:, 0] + self.target_std[0] * correction[:, 0]
        gap12 = torch.clamp(cic_increment[:, 1], min=self.gap_floor) * torch.exp(
            correction[:, 1]
        )
        gap23 = torch.clamp(cic_increment[:, 2], min=self.gap_floor) * torch.exp(
            correction[:, 2]
        )
        predicted_increment = torch.stack((lambda1, gap12, gap23), dim=1)
        predicted_scaled = (predicted_increment - self.target_mean) / self.target_std
        predicted_eigenvalues = torch.stack(
            (lambda1, lambda1 + gap12, lambda1 + gap12 + gap23), dim=1
        )
        return predicted_scaled, predicted_eigenvalues, correction


def load_unet_backbone(model: UCICResidual, checkpoint: Path, device: str) -> dict:
    """Warm-start only the local field representation from a frozen U-PATCH winner."""
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    source = payload["state_dict"]
    backbone = {
        key.removeprefix("unet."): value
        for key, value in source.items()
        if key.startswith("unet.")
    }
    missing, unexpected = model.unet.load_state_dict(backbone, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"U-PATCH backbone mismatch: missing={missing}, unexpected={unexpected}"
        )
    return {
        "checkpoint": str(checkpoint),
        "model": payload.get("model"),
        "rotation": payload.get("rotation"),
        "seed": payload.get("seed"),
        "epoch": payload.get("epoch"),
        "score": payload.get("score"),
        "loaded_parameters": len(backbone),
    }


def predict_fold(
    model,
    adapter,
    core_ids,
    normalization,
    cic_by_parent,
    device,
) -> tuple[np.ndarray, np.ndarray, int]:
    model.eval()
    parent_parts, prediction_parts = [], []
    failures = 0
    with torch.no_grad():
        for core_id in core_ids:
            try:
                patch = adapter.extract(
                    int(core_id),
                    unet_impl.HALO_VOXELS,
                    unet_impl.CHANNELS,
                    alignment_voxels=unet_impl.ALIGNMENT_VOXELS,
                )
                parent = patch.authoritative_parent_id
                cic = np.asarray(cic_by_parent[parent], dtype=np.float32)
                if not np.all(np.isfinite(cic)):
                    raise RuntimeError(f"missing CIC anchor rows in core {core_id}")
                values, points = unet_impl.model_inputs(patch, normalization, device)
                prediction, _, _ = model(
                    values, points, torch.from_numpy(cic).to(device)
                )
                parent_parts.append(parent)
                prediction_parts.append(prediction.cpu().numpy())
            except Exception:
                failures += 1
                raise
    return np.concatenate(parent_parts), np.concatenate(prediction_parts), failures


def checkpoint_zero_parity(
    model: UCICResidual,
    values: torch.Tensor,
    points: torch.Tensor,
    cic_eigenvalues: torch.Tensor,
) -> dict:
    """Prove the registered null before the optimizer changes any parameter."""
    model.eval()
    with torch.no_grad():
        scaled, predicted, correction = model(values, points, cic_eigenvalues)
    maximum = float(torch.max(torch.abs(predicted - cic_eigenvalues)).cpu())
    gaps = predicted[:, 1:] - predicted[:, :-1]
    return {
        "maximum_absolute_eigenvalue_difference": maximum,
        "maximum_absolute_correction": float(torch.max(torch.abs(correction)).cpu()),
        "minimum_predicted_gap": float(torch.min(gaps).cpu()),
        "pass": bool(maximum <= 2.0e-6 and torch.all(gaps > 0).item()),
        "scaled_shape": list(scaled.shape),
    }


def physical_to_scaled(eigenvalues: np.ndarray, scaler: dict) -> np.ndarray:
    """Convenience reference used by tests and artifact audits."""
    increments = linear_increments(np.asarray(eigenvalues, dtype=np.float64))
    return (
        (increments - np.asarray(scaler["mean"])) / np.asarray(scaler["std"])
    ).astype(np.float32)
