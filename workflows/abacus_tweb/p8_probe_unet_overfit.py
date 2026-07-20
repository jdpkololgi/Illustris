#!/usr/bin/env python3
"""Overfit one frozen P4 core to test U-PATCH capacity and output range.

This is a diagnostic, not a generalisation result.  It answers whether the
registered U-PATCH model and interpolation head can expand to the target
dynamic range when optimization and data diversity are deliberately removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb import p8_train_unet_patch as unet_impl
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    increments_to_eigenvalues,
    linear_increments,
    scale_increments,
    unscale_increments,
)
from workflows.abacus_tweb.p8_epoch_training import append_jsonl


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
OUTPUT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/probes")


def r2(truth: np.ndarray, prediction: np.ndarray) -> list[float]:
    residual = np.sum((truth - prediction) ** 2, axis=0)
    total = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    normalized_residual = np.divide(
        residual,
        total,
        out=np.full(3, np.nan),
        where=total > 0,
    )
    return (1.0 - normalized_residual).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--core-id", type=int, default=15211)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--adapter", type=Path, default=unet_impl.ADAPTER)
    parser.add_argument("--selection", type=Path, default=unet_impl.SELECTION)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("U-PATCH overfit probe requires a CUDA interactive allocation")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = (
        args.output_root / f"unet_rotation_{args.rotation}"
        / f"core_{args.core_id}_seed_{args.seed}"
    )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"probe output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "loss_trace.jsonl"
    trace_path.write_text("")

    rotation_dir = args.p8_root / f"rotation_{args.rotation}"
    training_core = np.load(rotation_dir / "training_core_id.npy")
    if args.core_id not in set(np.asarray(training_core, dtype=np.int64).tolist()):
        raise RuntimeError("probe core is not owned by the rotation's training folds")
    truth_by_parent = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    scaler = json.loads((rotation_dir / "target_scaler.json").read_text())
    selection = json.loads(args.selection.read_text())
    normalization = selection["rotations"][str(args.rotation)]["normalization"]

    model = unet_impl.UPatch(
        base=args.base, latent_channels=args.latent_channels
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    started = time.time()

    with unet_impl.CanonicalFieldPatchAdapter(
        args.adapter, selection_manifest=args.selection, rotation=args.rotation
    ) as adapter:
        patch = adapter.extract(
            args.core_id,
            unet_impl.HALO_VOXELS,
            unet_impl.CHANNELS,
            alignment_voxels=unet_impl.ALIGNMENT_VOXELS,
        )
        parent = patch.authoritative_parent_id
        truth = np.asarray(truth_by_parent[parent], dtype=np.float64)
        target_scaled = scale_increments(linear_increments(truth), scaler)
        target = torch.from_numpy(target_scaled).to(args.device)
        values, points = unet_impl.model_inputs(patch, normalization, args.device)

        initial_loss = None
        final_prediction_scaled = None
        for step in range(args.steps + 1):
            model.train()
            prediction = model(values, points)
            loss = torch.mean((prediction - target) ** 2)
            if initial_loss is None:
                initial_loss = float(loss.detach().cpu())
            if step % args.log_every == 0 or step == args.steps:
                predicted_scaled = prediction.detach().cpu().numpy()
                predicted = increments_to_eigenvalues(
                    unscale_increments(predicted_scaled, scaler)
                )
                append_jsonl(
                    trace_path,
                    {
                        "step": step,
                        "scaled_mse": float(loss.detach().cpu()),
                        "r2": r2(truth, predicted),
                        "lambda1_prediction_std": float(predicted[:, 0].std()),
                        "lambda1_prediction_min": float(predicted[:, 0].min()),
                        "lambda1_prediction_max": float(predicted[:, 0].max()),
                        "elapsed_seconds": time.time() - started,
                    },
                )
            if step == args.steps:
                final_prediction_scaled = prediction.detach().cpu().numpy()
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    final_prediction = increments_to_eigenvalues(
        unscale_increments(final_prediction_scaled, scaler)
    )
    final_loss = float(np.mean((final_prediction_scaled - target_scaled) ** 2))
    summary = {
        "schema_version": 1,
        "stage": "P8 U-PATCH one-core overfit diagnostic",
        "rotation": args.rotation,
        "core_id": args.core_id,
        "seed": args.seed,
        "steps": args.steps,
        "rows": int(len(parent)),
        "initial_scaled_mse": initial_loss,
        "final_scaled_mse": final_loss,
        "loss_ratio": final_loss / initial_loss,
        "truth_lambda1_range": [float(truth[:, 0].min()), float(truth[:, 0].max())],
        "prediction_lambda1_range": [
            float(final_prediction[:, 0].min()),
            float(final_prediction[:, 0].max()),
        ],
        "r2": r2(truth, final_prediction),
        "elapsed_seconds": time.time() - started,
        "trace": str(trace_path),
    }
    atomic_json(output / "probe_summary.json", summary)
    np.save(output / "parent_node_id.npy", parent)
    np.save(output / "predicted_eigenvalues.npy", final_prediction.astype(np.float32))
    (output / "PROBE_COMPLETE").write_text(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
