#!/usr/bin/env python3
"""Train and evaluate the deterministic U-PATCH P8 control."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p6_field_patch_utils import (
    CanonicalFieldPatchAdapter,
    apply_frozen_normalization,
)
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    evaluate_complete_fold,
    increments_to_eigenvalues,
    linear_increments,
    scale_increments,
    sha256,
    unscale_increments,
)


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
ADAPTER = Path("/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter")
SELECTION = ADAPTER / "fullcap_selection_v1/selection_manifest.json"
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
HALO_VOXELS = 24
ALIGNMENT_VOXELS = 8
CHANNELS = ("counts", "exposure_apodized", "log_count_ratio")


class ChannelLayerNorm3d(nn.Module):
    """Learned per-voxel channel normalization with no patch-spatial statistic."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=1, keepdim=True)
        variance = values.var(dim=1, keepdim=True, unbiased=False)
        shape = (1, -1) + (1,) * (values.ndim - 2)
        normalized = (values - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight.view(shape) + self.bias.view(shape)


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1),
        ChannelLayerNorm3d(cout),
        nn.SiLU(),
        nn.Conv3d(cout, cout, 3, padding=1),
        ChannelLayerNorm3d(cout),
        nn.SiLU(),
    )


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 32, base: int = 24):
        super().__init__()
        self.enc0 = conv_block(in_channels, base)
        self.enc1 = conv_block(base, base * 2)
        self.enc2 = conv_block(base * 2, base * 4)
        self.bottleneck = conv_block(base * 4, base * 4)
        self.dec2 = conv_block(base * 8, base * 2)
        self.dec1 = conv_block(base * 4, base)
        self.dec0 = conv_block(base * 2, base)
        self.output = nn.Conv3d(base, latent_channels, 1)
        self.pool = nn.MaxPool3d(2, ceil_mode=True)

    @staticmethod
    def up(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(values, size=reference.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(values)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        bottleneck = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat((self.up(bottleneck, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up(d2, e1), e1), dim=1))
        d0 = self.dec0(torch.cat((self.up(d1, e0), e0), dim=1))
        return self.output(d0)


class UPatch(nn.Module):
    def __init__(self, base: int = 24, latent_channels: int = 32, head_width: int = 128):
        super().__init__()
        self.unet = UNet3D(3, latent_channels, base)
        self.head = nn.Sequential(
            nn.Linear(latent_channels, head_width),
            nn.SiLU(),
            nn.Linear(head_width, head_width),
            nn.SiLU(),
            nn.Linear(head_width, 3),
        )

    def sample_latent(
        self, values: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """Return the exact per-galaxy U-Net summary consumed by the point head."""
        latent = self.unet(values)
        return F.grid_sample(
            latent, points, mode="bilinear", align_corners=True, padding_mode="border"
        )[0, :, 0, 0].T

    def forward(self, values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return self.head(self.sample_latent(values, points))


def grid_coordinates(frac: np.ndarray, shape: tuple[int, int, int], device: str) -> torch.Tensor:
    normalized = np.empty_like(frac, dtype=np.float32)
    for axis, size in enumerate(shape):
        normalized[:, axis] = 2.0 * frac[:, axis] / max(size - 1, 1) - 1.0
    coordinates = np.ascontiguousarray(normalized[:, (2, 1, 0)])
    return torch.from_numpy(coordinates).to(device).view(1, 1, 1, -1, 3)


def model_inputs(patch, normalization: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = apply_frozen_normalization(patch, normalization)
    at = {name: index for index, name in enumerate(patch.channel_names)}
    # This is the P6-registered three-channel selection-aware T2 mapping.
    density_proxy = np.clip(
        np.expm1(np.clip(patch.values[at["log_count_ratio"]], -20.0, 4.0)),
        -1.0,
        20.0,
    )
    values = np.stack(
        (
            normalized[at["counts"]],
            density_proxy,
            patch.values[at["exposure_apodized"]],
        )
    ).astype(np.float32)
    tensor = torch.from_numpy(values[None]).to(device)
    points = grid_coordinates(
        patch.authoritative_frac_index_local, tuple(values.shape[1:]), device
    )
    return tensor, points


def predict_fold(model, adapter, core_ids, normalization, device) -> tuple[np.ndarray, np.ndarray, int]:
    model.eval()
    parent_parts, prediction_parts = [], []
    failures = 0
    with torch.no_grad():
        for core_id in core_ids:
            try:
                patch = adapter.extract(
                    int(core_id), HALO_VOXELS, CHANNELS, alignment_voxels=ALIGNMENT_VOXELS
                )
                values, points = model_inputs(patch, normalization, device)
                prediction = model(values, points).cpu().numpy()
                parent_parts.append(patch.authoritative_parent_id)
                prediction_parts.append(prediction)
            except Exception:
                failures += 1
                raise
    return np.concatenate(parent_parts), np.concatenate(prediction_parts), failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--loss-log-every", type=int, default=25,
                        help="steps between windowed training-loss records (no validation cost)")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("U-PATCH requires a CUDA interactive allocation")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output = args.p8_root / "u_patch" / f"rotation_{args.rotation}" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    truth = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    assignment = np.load(args.assignment, mmap_mode="r")
    rotation_dir = args.p8_root / f"rotation_{args.rotation}"
    roles = json.loads((rotation_dir / "roles.json").read_text())
    scaler = json.loads((rotation_dir / "target_scaler.json").read_text())
    selection_manifest = json.loads(args.selection.read_text())
    normalization = selection_manifest["rotations"][str(args.rotation)]["normalization"]
    training_core = np.load(rotation_dir / "training_core_id.npy")
    training_core_weight = np.load(rotation_dir / "training_core_weight.npy").astype(np.float64)
    training_probability = training_core_weight / training_core_weight.sum()
    validation_core = np.load(rotation_dir / "validation_core_id.npy")
    row_weight = np.load(rotation_dir / "active_training_weight.npy", mmap_mode="r")
    parent_weight = np.zeros(len(truth), dtype=np.float32)
    parent_weight[np.asarray(assignment["parent_node_id"], dtype=np.int64)] = row_weight
    parent_shell = np.full(len(truth), -1, dtype=np.int8)
    parent_shell[np.asarray(assignment["parent_node_id"], dtype=np.int64)] = np.asarray(
        assignment["shell"], dtype=np.int8)
    target_scaled = scale_increments(linear_increments(np.asarray(truth)), scaler)

    model = UPatch(base=args.base, latent_channels=args.latent_channels).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    best_score = -np.inf
    best_state = None
    best_step = -1
    stale = 0
    history = []
    loss_trace: list[dict] = []
    loss_window: list[float] = []
    shell_exposure = np.zeros(4, dtype=np.int64)
    started = time.time()
    maximum_memory = 0

    with CanonicalFieldPatchAdapter(
        args.adapter, selection_manifest=args.selection, rotation=args.rotation
    ) as adapter:
        for step in range(1, args.steps + 1):
            core_id = int(rng.choice(training_core, p=training_probability))
            patch = adapter.extract(
                core_id, HALO_VOXELS, CHANNELS, alignment_voxels=ALIGNMENT_VOXELS
            )
            parent = patch.authoritative_parent_id
            weight = torch.from_numpy(parent_weight[parent]).to(args.device)
            target = torch.from_numpy(target_scaled[parent]).to(args.device)
            values, points = model_inputs(patch, normalization, args.device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(values, points)
            loss_per_row = torch.mean((prediction - target) ** 2, dim=1)
            loss = torch.sum(weight * loss_per_row) / torch.sum(weight).clamp_min(1e-12)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            shell_exposure += np.bincount(
                parent_shell[parent], minlength=4
            )[:4]
            maximum_memory = max(maximum_memory, int(torch.cuda.max_memory_allocated()))

            # Training-curve logging, decoupled from --eval-every (no validation cost).
            # The short screens stored one instantaneous single-patch loss and so had no
            # learning curve; a windowed mean reflects optimization, not patch draw.
            loss_window.append(float(loss.detach().cpu()))
            if step % args.loss_log_every == 0 or step == args.steps:
                loss_trace.append({
                    "step": step,
                    "training_loss_window_mean": float(np.mean(loss_window)),
                    "training_loss_window_min": float(np.min(loss_window)),
                    "training_loss_window_max": float(np.max(loss_window)),
                    "window": len(loss_window),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                })
                with open(output / "loss_trace.jsonl", "a") as handle:
                    handle.write(json.dumps(loss_trace[-1]) + "\n")
                loss_window.clear()

            if step % args.eval_every == 0 or step == args.steps:
                val_parent, val_scaled, failures = predict_fold(
                    model, adapter, validation_core, normalization, args.device
                )
                val_eigen = increments_to_eigenvalues(
                    unscale_increments(val_scaled, scaler)
                ).astype(np.float32)
                report = evaluate_complete_fold(
                    parent_node_id=val_parent,
                    predicted_eigenvalues=val_eigen,
                    truth_by_parent=truth,
                    assignment=assignment,
                    validation_fold=roles["validation_fold"],
                    runtime={
                        "training_step": step,
                        "elapsed_seconds": time.time() - started,
                        "patch_failures": failures,
                        "maximum_cuda_memory_bytes": maximum_memory,
                    },
                )
                score = report["primary_macro_r2_lambda1"]
                history.append({
                    "step": step,
                    "training_loss": float(loss.detach().cpu()),
                    "primary_macro_r2_lambda1": score,
                    "per_shell_lambda1_r2": {
                        name: row["lambda1"]["r2"]
                        for name, row in report["per_shell"].items()
                    },
                })
                print(json.dumps(history[-1]), flush=True)
                if score > best_score:
                    best_score, best_step, stale = score, step, 0
                    best_state = copy.deepcopy(model.state_dict())
                    torch.save(
                        {
                            "state_dict": best_state,
                            "rotation": args.rotation,
                            "seed": args.seed,
                            "step": step,
                            "score": score,
                            "scaler": scaler,
                            "normalization": normalization,
                        },
                        output / "best_checkpoint.pt",
                    )
                    np.save(output / "best_validation_parent_node_id.npy", val_parent)
                    np.save(output / "best_validation_eigenvalues.npy", val_eigen)
                    atomic_json(output / "best_validation_report.json", report)
                else:
                    stale += 1
                if stale >= args.patience:
                    break

    if best_state is None:
        raise RuntimeError("U-PATCH did not produce a complete validation checkpoint")
    final = {
        "schema_version": 1,
        "model": "U-PATCH",
        "rotation": args.rotation,
        "seed": args.seed,
        "status": "screen_complete",
        "best_step": best_step,
        "best_primary_macro_r2_lambda1": best_score,
        "steps_run": step,
        "history": history,
        "loss_trace": loss_trace,
        "architecture": {
            "base": args.base,
            "latent_channels": args.latent_channels,
            "normalization": "per-voxel ChannelLayerNorm3d",
            "channels": list(CHANNELS),
            "halo_voxels": HALO_VOXELS,
            "alignment_voxels": ALIGNMENT_VOXELS,
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        "training": {
            "objective": "sqrt-shell-weighted mean MSE over authoritative core rows",
            "core_sampling": "probability proportional to frozen core weight",
            "shell_row_exposure": shell_exposure.tolist(),
            "maximum_cuda_memory_bytes": maximum_memory,
            "elapsed_seconds": time.time() - started,
        },
        "inputs": {
            "assignment": str(args.assignment),
            "assignment_sha256": sha256(args.assignment),
            "selection": str(args.selection),
            "selection_sha256": sha256(args.selection),
        },
    }
    atomic_json(output / "screen_summary.json", final)
    (output / "U_PATCH_SCREEN_COMPLETE").write_text(
        f"rotation={args.rotation} seed={args.seed} score={best_score:.8f}\n"
    )
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
