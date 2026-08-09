#!/usr/bin/env python3
"""Prove exact interrupted/resumed D0 training parity on real P8.9 patches."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_density_training_utils import (
    DensityUnitAdapter,
    TRAINING_CONTRACT,
    extract_core_prediction,
)
from workflows.abacus_tweb.p8_epoch_training import (
    EpochLossAccumulator,
    epoch_order,
    patch_objective,
    validate_resume_order,
)
from workflows.abacus_tweb.p8_train_density_patch import checkpoint_payload
from workflows.abacus_tweb.p8_train_patch_recovery import atomic_torch_save, torch_load
from workflows.abacus_tweb.p8_train_unet_patch import UNet3D


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
DEFAULT_OUTPUT = ROOT / "p8_density_phys_v1/resume_parity/rotation_0/seed_42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--split-after", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def new_trajectory(initial_state: dict, device: str, config: dict):
    model = UNet3D(in_channels=3, latent_channels=1, base=24).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["optimization"]["epochs"] * config["counts"]["units_train"]),
    )
    return model, optimizer, scheduler


def train_step(model, optimizer, scheduler, adapter, unit, mean_unit_weight, gradient_clip, device):
    patch, values, target, mask, _ = adapter.extract(unit, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = extract_core_prediction(model(values), patch.core_slice)
    loss_per_voxel = (prediction[mask] - target[mask]) ** 2
    shell_weight = float(unit["unit_weight"] / unit["supported_voxels"])
    numerator = shell_weight * torch.sum(loss_per_voxel)
    loss = patch_objective(numerator, mean_core_weight=mean_unit_weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
    optimizer.step()
    scheduler.step()
    loss_np = loss_per_voxel.detach().cpu().numpy()
    return float(loss.detach().cpu()), loss_np, shell_weight


def maximum_tensor_difference(left, right) -> float:
    differences = []
    for key in left:
        if key not in right:
            raise RuntimeError(f"missing state key after resume: {key}")
        a, b = left[key], right[key]
        if torch.is_tensor(a):
            differences.append(float(torch.max(torch.abs(a.detach().cpu() - b.detach().cpu()))))
        elif isinstance(a, dict):
            differences.append(maximum_tensor_difference(a, b))
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                raise RuntimeError("state sequence length differs after resume")
            for aa, bb in zip(a, b):
                if torch.is_tensor(aa):
                    differences.append(float(torch.max(torch.abs(aa.cpu() - bb.cpu()))))
                elif aa != bb:
                    raise RuntimeError("non-tensor state differs after resume")
        elif a != b:
            raise RuntimeError(f"non-tensor state differs after resume: {key}")
    return max(differences, default=0.0)


def main() -> None:
    args = parse_args()
    if args.rotation != 0 or args.seed != 42:
        raise RuntimeError("frozen D0 resume parity is rotation 0, seed 42")
    if args.steps < 3 or not 0 < args.split_after < args.steps:
        raise ValueError("require steps >= 3 and 0 < split-after < steps")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("resume parity requires an interactive CUDA allocation")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "interrupted_checkpoint.pt"
    started = time.time()

    with DensityUnitAdapter(rotation=args.rotation, contract_root=TRAINING_CONTRACT) as adapter:
        config = adapter.config
        train_units = np.asarray(adapter.units)[
            np.isin(adapter.units["fold"], config["roles"]["train_folds"])
        ]
        unit_ids = np.arange(len(train_units), dtype=np.int64)
        weights = np.asarray(train_units["unit_weight"], dtype=np.float64)
        order = epoch_order(unit_ids, seed=args.seed, epoch=1, core_weight=weights)
        chosen = order[: args.steps]
        mean_weight = float(weights.mean())

        base = UNet3D(in_channels=3, latent_channels=1, base=24).to(args.device)
        initial_state = copy.deepcopy(base.state_dict())
        initial_cpu_rng = torch.get_rng_state().clone()
        initial_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        del base

        continuous, continuous_optimizer, continuous_scheduler = new_trajectory(
            initial_state, args.device, config
        )
        torch.set_rng_state(initial_cpu_rng)
        torch.cuda.set_rng_state_all(initial_cuda_rng)
        continuous_loss = []
        for index in chosen:
            loss, _, _ = train_step(
                continuous, continuous_optimizer, continuous_scheduler, adapter,
                train_units[int(index)], mean_weight,
                config["optimization"]["gradient_clip"], args.device,
            )
            continuous_loss.append(loss)
        continuous_cpu_rng = torch.get_rng_state().clone()
        continuous_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]

        resumed, resumed_optimizer, resumed_scheduler = new_trajectory(
            initial_state, args.device, config
        )
        torch.set_rng_state(initial_cpu_rng)
        torch.cuda.set_rng_state_all(initial_cuda_rng)
        resumed_loss = []
        accumulator = EpochLossAccumulator()
        shell_numerator = np.zeros(4, dtype=np.float64)
        shell_denominator = np.zeros(4, dtype=np.float64)
        for cursor, index in enumerate(chosen[: args.split_after]):
            unit = train_units[int(index)]
            loss, loss_np, shell_weight = train_step(
                resumed, resumed_optimizer, resumed_scheduler, adapter, unit,
                mean_weight, config["optimization"]["gradient_clip"], args.device,
            )
            resumed_loss.append(loss)
            row_weight = np.full(len(loss_np), shell_weight, dtype=np.float32)
            accumulator.add(loss_np, row_weight)
            shell = int(unit["shell"])
            shell_numerator[shell] += float(np.sum(loss_np) * shell_weight)
            shell_denominator[shell] += float(len(loss_np) * shell_weight)

        payload = checkpoint_payload(
            model=resumed,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            epoch=1,
            cursor=args.split_after,
            order=order,
            global_step=args.split_after,
            accumulator=accumulator,
            shell_numerator=shell_numerator,
            shell_denominator=shell_denominator,
            history=[],
            best_score=-np.inf,
            best_epoch=-1,
            maximum_memory=int(torch.cuda.max_memory_allocated()),
            arguments={"git_revision": "resume-parity", "rotation": 0, "seed": 42},
            config_sha256="resume-parity",
        )
        atomic_torch_save(payload, checkpoint)
        del resumed, resumed_optimizer, resumed_scheduler

        restored_state = torch_load(checkpoint, args.device)
        validate_resume_order(restored_state["epoch_order"], unit_ids, restored_state["cursor"])
        resumed, resumed_optimizer, resumed_scheduler = new_trajectory(
            initial_state, args.device, config
        )
        resumed.load_state_dict(restored_state["model_state"])
        resumed_optimizer.load_state_dict(restored_state["optimizer_state"])
        resumed_scheduler.load_state_dict(restored_state["scheduler_state"])
        torch.set_rng_state(restored_state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in restored_state["cuda_rng_state_all"]])
        for index in chosen[args.split_after :]:
            loss, _, _ = train_step(
                resumed, resumed_optimizer, resumed_scheduler, adapter,
                train_units[int(index)], mean_weight,
                config["optimization"]["gradient_clip"], args.device,
            )
            resumed_loss.append(loss)

        model_max_abs = maximum_tensor_difference(
            continuous.state_dict(), resumed.state_dict()
        )
        optimizer_max_abs = maximum_tensor_difference(
            continuous_optimizer.state_dict(), resumed_optimizer.state_dict()
        )
        scheduler_equal = continuous_scheduler.state_dict() == resumed_scheduler.state_dict()
        loss_max_abs = float(np.max(np.abs(np.asarray(continuous_loss) - np.asarray(resumed_loss))))
        cpu_rng_equal = bool(torch.equal(continuous_cpu_rng, torch.get_rng_state()))
        current_cuda_rng = torch.cuda.get_rng_state_all()
        cuda_rng_equal = all(
            torch.equal(left, right)
            for left, right in zip(continuous_cuda_rng, current_cuda_rng)
        )

    threshold = 1e-7
    passed = bool(
        model_max_abs <= threshold
        and optimizer_max_abs <= threshold
        and scheduler_equal
        and loss_max_abs <= threshold
        and cpu_rng_equal
        and cuda_rng_equal
        and int(restored_state["cursor"]) == args.split_after
        and np.array_equal(restored_state["epoch_order"], order)
    )
    report = {
        "schema_version": "p8-density-resume-parity-v1",
        "pass": passed,
        "rotation": args.rotation,
        "seed": args.seed,
        "steps": args.steps,
        "split_after": args.split_after,
        "unit_indices": chosen.tolist(),
        "model_state_max_abs": model_max_abs,
        "optimizer_state_max_abs": optimizer_max_abs,
        "scheduler_state_equal": scheduler_equal,
        "loss_trace_max_abs": loss_max_abs,
        "continuous_loss": continuous_loss,
        "resumed_loss": resumed_loss,
        "cpu_rng_equal": cpu_rng_equal,
        "cuda_rng_equal": cuda_rng_equal,
        "epoch_order_equal": bool(np.array_equal(restored_state["epoch_order"], order)),
        "cursor_equal": int(restored_state["cursor"]) == args.split_after,
        "tolerance": threshold,
        "checkpoint": str(checkpoint),
        "elapsed_seconds": time.time() - started,
    }
    temporary = args.output / "resume_parity.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output / "resume_parity.json")
    if passed:
        (args.output / "D0_RESUME_PARITY_PASS").write_text("exact within 1e-7\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
