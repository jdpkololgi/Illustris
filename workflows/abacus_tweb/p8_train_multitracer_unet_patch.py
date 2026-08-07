#!/usr/bin/env python3
"""Train U-PATCH with separate Bright and Faint observation channels.

Predictions, loss, validation and target scaling are unchanged Bright-only P8.
Faint contributes only three additional field channels: normalized counts,
selection-aware density proxy and effective exposure.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

from workflows.abacus_tweb.p6_field_patch_utils import (
    CAP_NAME,
    CanonicalFieldPatchAdapter,
    apply_frozen_normalization,
    channel_transform,
    derive_selection_channels,
    patch_redshift,
)
from workflows.abacus_tweb.p6_refit_fullcap_selection import radius_to_redshift_grid
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    evaluate_complete_fold,
    increments_to_eigenvalues,
    linear_increments,
    scale_increments,
    sha256,
    unscale_increments,
)
from workflows.abacus_tweb.p8_train_unet_patch import (
    ALIGNMENT_VOXELS,
    HALO_VOXELS,
    CHANNELS,
    UNet3D,
    grid_coordinates,
)


P8 = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
MT = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
P6 = Path("/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter")
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")


class MultitracerUPatch(nn.Module):
    def __init__(self, base: int = 24, latent_channels: int = 32, head_width: int = 128):
        super().__init__()
        self.unet = UNet3D(6, latent_channels, base)
        self.head = nn.Sequential(
            nn.Linear(latent_channels, head_width),
            nn.SiLU(),
            nn.Linear(head_width, head_width),
            nn.SiLU(),
            nn.Linear(head_width, 3),
        )

    def forward(self, values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        latent = self.unet(values)
        sampled = torch.nn.functional.grid_sample(
            latent, points, mode="bilinear", align_corners=True, padding_mode="border"
        )[0, :, 0, 0].T
        return self.head(sampled)


def zscore(values: np.ndarray, spec: dict) -> np.ndarray:
    transformed = channel_transform("counts", values)
    return (
        (transformed - np.float32(spec["mean"]))
        / np.float32(max(spec["std"], 1.0e-6))
    ).astype(np.float32)


class MultitracerFieldAdapter:
    def __init__(self, *, product: str, rotation: int):
        self.product = product
        self.rotation = int(rotation)
        self.selection_path = MT / "selection" / product / "multitracer_selection_manifest.json"
        self.field_path = MT / "fields" / product / "manifest.json"
        self.selection = json.loads(self.selection_path.read_text())
        self.fields = json.loads(self.field_path.read_text())
        bright_selection_path = Path(
            self.selection["tracers"]["BGS_BRIGHT"]["selection_manifest"]
        )
        self.bright_selection = json.loads(bright_selection_path.read_text())
        self.bright_normalization = self.bright_selection["rotations"][str(rotation)][
            "normalization"
        ]
        self.faint_rotation = self.selection["tracers"]["BGS_FAINT"]["rotations"][
            str(rotation)
        ]
        self.faint_normalization = self.faint_rotation["normalization"]
        self.base = CanonicalFieldPatchAdapter(
            P6, selection_manifest=bright_selection_path, rotation=rotation
        )
        self.handles: dict[int, h5py.File] = {}
        self.radius_grid, self.redshift_grid = radius_to_redshift_grid(0.10, 0.60)
        self.core_cap = self.base.core_cap

    def close(self) -> None:
        self.base.close()
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _handle(self, cap: int) -> h5py.File:
        if cap not in self.handles:
            path = self.fields["components"][CAP_NAME[cap]]["file"]
            self.handles[cap] = h5py.File(path, "r")
        return self.handles[cap]

    def extract(self, core_id: int):
        bright = self.base.extract(
            core_id, HALO_VOXELS, CHANNELS, alignment_voxels=ALIGNMENT_VOXELS
        )
        selection = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(bright.context_start, bright.context_stop)
        )
        handle = self._handle(bright.cap)
        faint_counts = np.asarray(handle["counts"][selection], dtype=np.float32)
        faint_exposure = np.asarray(
            handle["exposure_apodized"][selection], dtype=np.float32
        )
        cap_name = CAP_NAME[bright.cap]
        grid = self.fields["components"][cap_name]["grid"]
        redshift = patch_redshift(
            origin_mpc=np.asarray(grid["origin_mpc"], dtype=np.float64),
            cell_mpc=float(grid["cell_mpc"]),
            context_start=bright.context_start,
            shape=tuple(int(value) for value in bright.context_stop - bright.context_start),
            radius_grid_mpc=self.radius_grid,
            redshift_grid=self.redshift_grid,
        )
        curve = self.faint_rotation["caps"][cap_name]
        faint_derived = derive_selection_channels(
            faint_counts,
            faint_exposure,
            redshift,
            cell_mpc=float(grid["cell_mpc"]),
            grid_z=np.asarray(curve["grid_z"], dtype=np.float64),
            ntilde=np.asarray(curve["ntilde"], dtype=np.float64),
            epsilon=float(self.selection["contrast"]["epsilon"]),
            minimum_exposure=float(self.selection["contrast"]["minimum_exposure"]),
        )
        return bright, faint_counts, faint_exposure, faint_derived["log_count_ratio"]


def model_inputs(adapter: MultitracerFieldAdapter, extracted, device: str):
    bright, faint_counts, faint_exposure, faint_log_ratio = extracted
    normalized = apply_frozen_normalization(bright, adapter.bright_normalization)
    at = {name: index for index, name in enumerate(bright.channel_names)}
    bright_density = np.clip(
        np.expm1(np.clip(bright.values[at["log_count_ratio"]], -20.0, 4.0)), -1.0, 20.0
    )
    faint_density = np.clip(
        np.expm1(np.clip(faint_log_ratio, -20.0, 4.0)), -1.0, 20.0
    )
    faint_count_spec = adapter.faint_normalization["channels"]["counts"]
    values = np.stack(
        (
            normalized[at["counts"]],
            bright_density,
            bright.values[at["exposure_apodized"]],
            zscore(faint_counts, faint_count_spec),
            faint_density,
            faint_exposure,
        )
    ).astype(np.float32)
    tensor = torch.from_numpy(values[None]).to(device)
    points = grid_coordinates(
        bright.authoritative_frac_index_local, tuple(values.shape[1:]), device
    )
    return bright, tensor, points


def predict_fold(model, adapter, core_ids, device):
    model.eval()
    parents, predictions = [], []
    with torch.inference_mode():
        for core_id in core_ids:
            bright, values, points = model_inputs(
                adapter, adapter.extract(int(core_id)), device
            )
            parents.append(bright.authoritative_parent_id)
            predictions.append(model(values, points).cpu().numpy())
    return np.concatenate(parents), np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=("bf_oracle_assigned_v1", "bf_proxy_response_v1"), required=True)
    parser.add_argument("--run-name", default="screen")
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--loss-log-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--p8-root", type=Path, default=P8)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("multitracer U-PATCH requires a GPU allocation")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output = (
        MT / "models/u_patch" / args.product / f"rotation_{args.rotation}"
        / f"seed_{args.seed}" / args.run_name
    )
    output.mkdir(parents=True, exist_ok=True)
    truth = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    assignment = np.load(args.assignment, mmap_mode="r")
    rotation_dir = args.p8_root / f"rotation_{args.rotation}"
    roles = json.loads((rotation_dir / "roles.json").read_text())
    scaler = json.loads((rotation_dir / "target_scaler.json").read_text())
    training_core = np.load(rotation_dir / "training_core_id.npy")
    training_weight = np.load(rotation_dir / "training_core_weight.npy").astype(np.float64)
    training_probability = training_weight / training_weight.sum()
    validation_core = np.load(rotation_dir / "validation_core_id.npy")
    row_weight = np.load(rotation_dir / "active_training_weight.npy", mmap_mode="r")
    active_parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    parent_weight = np.zeros(len(truth), dtype=np.float32)
    parent_weight[active_parent] = row_weight
    target_scaled = scale_increments(linear_increments(np.asarray(truth)), scaler)

    model = MultitracerUPatch(args.base, args.latent_channels).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    history, loss_window = [], []
    best_score, best_step, best_state = -np.inf, -1, None
    started = time.time()
    with MultitracerFieldAdapter(product=args.product, rotation=args.rotation) as adapter:
        for step in range(1, args.steps + 1):
            core_id = int(rng.choice(training_core, p=training_probability))
            bright, values, points = model_inputs(adapter, adapter.extract(core_id), args.device)
            parent = bright.authoritative_parent_id
            weight = torch.from_numpy(parent_weight[parent]).to(args.device)
            target = torch.from_numpy(target_scaled[parent]).to(args.device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(values, points)
            per_row = torch.mean((prediction - target) ** 2, dim=1)
            loss = torch.sum(weight * per_row) / torch.sum(weight).clamp_min(1.0e-12)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            loss_window.append(float(loss.detach().cpu()))
            if step % args.loss_log_every == 0 or step == args.steps:
                record = {
                    "step": step,
                    "training_loss_window_mean": float(np.mean(loss_window)),
                    "training_loss_window_min": float(np.min(loss_window)),
                    "training_loss_window_max": float(np.max(loss_window)),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                with (output / "loss_trace.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                loss_window.clear()
            if step % args.eval_every == 0 or step == args.steps:
                val_parent, val_scaled = predict_fold(
                    model, adapter, validation_core, args.device
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
                    runtime={"training_step": step, "elapsed_seconds": time.time() - started},
                )
                score = report["primary_macro_r2_lambda1"]
                history.append({"step": step, "macro_r2_lambda1": score})
                print(json.dumps(history[-1]), flush=True)
                torch.save(
                    {
                        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "step": step,
                        "product": args.product, "rotation": args.rotation, "seed": args.seed,
                    },
                    output / "latest_checkpoint.pt",
                )
                if score > best_score:
                    best_score, best_step = score, step
                    best_state = copy.deepcopy(model.state_dict())
                    torch.save(
                        {"state_dict": best_state, "step": step, "score": score},
                        output / "best_checkpoint.pt",
                    )
                    np.save(output / "best_validation_parent_node_id.npy", val_parent)
                    np.save(output / "best_validation_eigenvalues.npy", val_eigen)
                    atomic_json(output / "best_validation_report.json", report)
    if best_state is None:
        raise RuntimeError("multitracer U-PATCH produced no validation checkpoint")
    summary = {
        "schema_version": "p8-multitracer-u-patch-v1",
        "model": "U-PATCH-BRIGHT_TARGET-FAINT_CONTEXT",
        "product": args.product,
        "rotation": args.rotation,
        "seed": args.seed,
        "steps": args.steps,
        "best_step": best_step,
        "best_primary_macro_r2_lambda1": best_score,
        "history": history,
        "channels": [
            "bright_counts", "bright_density_proxy", "bright_exposure",
            "faint_counts", "faint_density_proxy", "faint_exposure",
        ],
        "supervision_contract": "BGS_BRIGHT only; Faint context is never scored",
        "selection_manifest": str(
            MT / "selection" / args.product / "multitracer_selection_manifest.json"
        ),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output / "screen_summary.json", summary)
    (output / "MULTITRACER_U_PATCH_SCREEN_COMPLETE").write_text(
        f"product={args.product} score={best_score:.8f}\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
