#!/usr/bin/env python3
"""Exact-resume complete-epoch trainer for U-DENSITY-PHYS-v1."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
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
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_epoch_training import (
    EpochLossAccumulator,
    append_jsonl,
    epoch_order,
    patch_objective,
    reconcile_loss_trace,
    rewrite_jsonl,
    validate_resume_order,
)
from workflows.abacus_tweb.p8_train_patch_recovery import atomic_torch_save, torch_load
from workflows.abacus_tweb.p8_train_unet_patch import UNet3D


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
OUTPUT_ROOT = ROOT / "p8_density_phys_v1/d0_runs"


@dataclass
class RegressionAccumulator:
    n: int = 0
    sum_truth: float = 0.0
    sum_prediction: float = 0.0
    sum_truth_square: float = 0.0
    sum_prediction_square: float = 0.0
    sum_product: float = 0.0
    sum_squared_error: float = 0.0
    sum_absolute_error: float = 0.0

    def add(self, prediction: np.ndarray, truth: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        if (
            prediction.shape != truth.shape
            or not np.all(np.isfinite(prediction))
            or not np.all(np.isfinite(truth))
        ):
            raise ValueError("finite prediction/truth arrays with identical shape required")
        residual = prediction - truth
        self.n += int(truth.size)
        self.sum_truth += float(truth.sum())
        self.sum_prediction += float(prediction.sum())
        self.sum_truth_square += float(np.square(truth).sum())
        self.sum_prediction_square += float(np.square(prediction).sum())
        self.sum_product += float((prediction * truth).sum())
        self.sum_squared_error += float(np.square(residual).sum())
        self.sum_absolute_error += float(np.abs(residual).sum())

    def report(self) -> dict:
        if self.n < 2:
            raise RuntimeError("regression accumulator has fewer than two voxels")
        mean_truth = self.sum_truth / self.n
        mean_prediction = self.sum_prediction / self.n
        sst = self.sum_truth_square - self.n * mean_truth**2
        var_prediction = self.sum_prediction_square - self.n * mean_prediction**2
        covariance = self.sum_product - self.n * mean_truth * mean_prediction
        pearson = covariance / np.sqrt(max(sst * var_prediction, 1e-30))
        return {
            "n": int(self.n),
            "r2": float(1.0 - self.sum_squared_error / max(sst, 1e-30)),
            "pearson": float(pearson),
            "rmse": float(np.sqrt(self.sum_squared_error / self.n)),
            "mae": float(self.sum_absolute_error / self.n),
            "bias": float(mean_prediction - mean_truth),
            "truth_mean": float(mean_truth),
            "prediction_mean": float(mean_prediction),
            "truth_std": float(np.sqrt(max(sst / self.n, 0.0))),
            "prediction_std": float(np.sqrt(max(var_prediction / self.n, 0.0))),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="scientific_v1")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contract-root", type=Path, default=TRAINING_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--loss-log-every", type=int, default=25)
    return parser.parse_args()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def checkpoint_payload(
    *, model, optimizer, scheduler, epoch, cursor, order, global_step,
    accumulator, shell_numerator, shell_denominator, history, best_score,
    best_epoch, maximum_memory, arguments, config_sha256,
) -> dict:
    payload = {
        "schema_version": "p8-density-training-checkpoint-v1",
        "git_revision": arguments["git_revision"],
        "model": "U-DENSITY-PHYS-v1",
        "rotation": int(arguments["rotation"]),
        "seed": int(arguments["seed"]),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(epoch),
        "cursor": int(cursor),
        "epoch_order": np.asarray(order, dtype=np.int64),
        "global_step": int(global_step),
        "epoch_accumulator": accumulator.as_dict(),
        "shell_numerator": np.asarray(shell_numerator, dtype=np.float64),
        "shell_denominator": np.asarray(shell_denominator, dtype=np.float64),
        "history": history,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "maximum_memory": int(maximum_memory),
        "arguments": arguments,
        "config_sha256": config_sha256,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def validate_all_units(
    model: torch.nn.Module,
    adapter: DensityUnitAdapter,
    units: np.ndarray,
    device: str,
) -> dict:
    by_shell = [RegressionAccumulator() for _ in range(4)]
    overall = RegressionAccumulator()
    scaler = adapter.scaler
    model.eval()
    with torch.no_grad():
        for unit in units:
            patch, values, target_scaled, mask, _ = adapter.extract(unit, device)
            prediction_scaled = extract_core_prediction(model(values), patch.core_slice)
            pred = prediction_scaled[mask].cpu().numpy() * scaler["std"] + scaler["mean"]
            truth = target_scaled[mask].cpu().numpy() * scaler["std"] + scaler["mean"]
            shell = int(unit["shell"])
            by_shell[shell].add(pred, truth)
            overall.add(pred, truth)
    shell_reports = {str(shell): row.report() for shell, row in enumerate(by_shell)}
    return {
        "overall": overall.report(),
        "by_shell": shell_reports,
        "macro_shell_r2_delta_r7": float(np.mean([
            shell_reports[str(shell)]["r2"] for shell in range(4)
        ])),
        "macro_shell_pearson_delta_r7": float(np.mean([
            shell_reports[str(shell)]["pearson"] for shell in range(4)
        ])),
        "units": int(len(units)),
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("D0 training requires an interactive CUDA allocation")
    if args.rotation != 0 or args.seed != 42:
        raise RuntimeError("the first frozen D0 trajectory is rotation 0, seed 42")
    config_path = args.contract_root / f"rotation_{args.rotation}/d0_config.json"
    config = json.loads(config_path.read_text())
    frozen = config["optimization"]
    if args.checkpoint_every != frozen["checkpoint_every_updates"]:
        raise RuntimeError("checkpoint interval differs from frozen D0 config")
    if args.loss_log_every != frozen["loss_log_every_updates"]:
        raise RuntimeError("loss log interval differs from frozen D0 config")
    if args.canary and args.run_name != "canary_v1":
        raise RuntimeError("--canary requires --run-name canary_v1")
    if not args.canary and args.run_name != "scientific_v1":
        raise RuntimeError("scientific D0 requires --run-name scientific_v1")
    revision = git_revision()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output = args.output_root / f"rotation_{args.rotation}/seed_{args.seed}/{args.run_name}"
    checkpoint_path = output / "training_checkpoint.pt"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError(f"non-empty D0 output requires --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    arguments = {
        "rotation": args.rotation,
        "seed": args.seed,
        "run_name": args.run_name,
        "canary": args.canary,
        "device": args.device,
        "contract_root": str(args.contract_root),
        "output_root": str(args.output_root),
        "checkpoint_every": args.checkpoint_every,
        "loss_log_every": args.loss_log_every,
        "git_revision": revision,
    }
    config_sha = sha256(config_path)

    with DensityUnitAdapter(rotation=args.rotation, contract_root=args.contract_root) as adapter:
        units = np.asarray(adapter.units)
        train_folds = config["roles"]["train_folds"]
        validation_fold = int(config["roles"]["validation_fold"])
        train_units = units[np.isin(units["fold"], train_folds)]
        validation_units = units[units["fold"] == validation_fold]
        unit_ids = np.arange(len(train_units), dtype=np.int64)
        unit_weight = np.asarray(train_units["unit_weight"], dtype=np.float64)
        mean_unit_weight = float(unit_weight.mean())
        expected_shell_voxels = np.bincount(
            train_units["shell"], weights=train_units["supported_voxels"], minlength=4
        ).astype(np.int64)

        model = UNet3D(in_channels=3, latent_channels=1, base=24).to(args.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=frozen["learning_rate"], weight_decay=frozen["weight_decay"]
        )
        total_updates = int(frozen["epochs"] * len(train_units))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)
        history = []
        best_score = -np.inf
        best_epoch = -1
        start_epoch = 1
        start_cursor = 0
        global_step = 0
        resume_order = None
        resume_accumulator = EpochLossAccumulator()
        resume_shell_numerator = np.zeros(4, dtype=np.float64)
        resume_shell_denominator = np.zeros(4, dtype=np.float64)
        maximum_memory = 0

        if args.resume:
            if not checkpoint_path.exists():
                raise RuntimeError("resume requested without D0 checkpoint")
            state = torch_load(checkpoint_path, args.device)
            for field, expected in (
                ("model", "U-DENSITY-PHYS-v1"),
                ("rotation", args.rotation),
                ("seed", args.seed),
                ("git_revision", revision),
                ("config_sha256", config_sha),
            ):
                if state[field] != expected:
                    raise RuntimeError(f"resume {field} mismatch: {state[field]} != {expected}")
            if state["arguments"] != arguments:
                raise RuntimeError("resume arguments differ from the frozen checkpoint")
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            start_epoch = int(state["epoch"])
            start_cursor = int(state["cursor"])
            global_step = int(state["global_step"])
            resume_order = np.asarray(state["epoch_order"], dtype=np.int64)
            validate_resume_order(resume_order, unit_ids, start_cursor)
            resume_accumulator = EpochLossAccumulator.from_dict(state["epoch_accumulator"])
            resume_shell_numerator = np.asarray(state["shell_numerator"], dtype=np.float64)
            resume_shell_denominator = np.asarray(state["shell_denominator"], dtype=np.float64)
            history = list(state["history"])
            best_score = float(state["best_score"])
            best_epoch = int(state["best_epoch"])
            maximum_memory = int(state["maximum_memory"])
            reconcile_loss_trace(output / "loss_trace.jsonl", maximum_global_step=global_step)
            rewrite_jsonl(output / "epoch_history.jsonl", history)
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if "cuda_rng_state_all" in state:
                torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state_all"]])
        else:
            (output / "loss_trace.jsonl").write_text("")
            (output / "epoch_history.jsonl").write_text("")

        run_manifest = {
            "schema_version": "p8-density-training-run-v1",
            "git_revision": revision,
            "model": "U-DENSITY-PHYS-v1",
            "arguments": arguments,
            "config": str(config_path),
            "config_sha256": config_sha,
            "train_units": int(len(train_units)),
            "validation_units": int(len(validation_units)),
            "expected_training_voxels_by_shell": expected_shell_voxels.tolist(),
            "epoch_sampler": "deterministic weighted without-replacement; every unit once",
            "scientific_epochs": int(frozen["epochs"]),
            "maximum_epochs_this_invocation": 1 if args.canary else int(frozen["epochs"]),
            "direct_eigenvalue_or_tensor_loss": False,
            "resume": bool(args.resume),
        }
        atomic_json(output / "run_manifest.json", run_manifest)
        started = time.time()
        loss_window_numerator = 0.0
        loss_window_denominator = 0.0
        window_objective = []
        maximum_epoch = 1 if args.canary else int(frozen["epochs"])

        for epoch in range(start_epoch, maximum_epoch + 1):
            if epoch == start_epoch and resume_order is not None:
                order = resume_order
                cursor0 = start_cursor
                accumulator = resume_accumulator
                shell_numerator = resume_shell_numerator.copy()
                shell_denominator = resume_shell_denominator.copy()
            else:
                order = epoch_order(unit_ids, seed=args.seed, epoch=epoch, core_weight=unit_weight)
                cursor0 = 0
                accumulator = EpochLossAccumulator()
                shell_numerator = np.zeros(4, dtype=np.float64)
                shell_denominator = np.zeros(4, dtype=np.float64)

            for cursor in range(cursor0, len(order)):
                unit = train_units[int(order[cursor])]
                patch, values, target, mask, _ = adapter.extract(unit, args.device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = extract_core_prediction(model(values), patch.core_slice)
                loss_per_voxel = (prediction[mask] - target[mask]) ** 2
                shell_weight = float(unit["unit_weight"] / unit["supported_voxels"])
                weighted_numerator = shell_weight * torch.sum(loss_per_voxel)
                loss = patch_objective(weighted_numerator, mean_core_weight=mean_unit_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), frozen["gradient_clip"])
                optimizer.step()
                scheduler.step()
                global_step += 1
                loss_np = loss_per_voxel.detach().cpu().numpy()
                weight_np = np.full(len(loss_np), shell_weight, dtype=np.float32)
                accumulator.add(loss_np, weight_np)
                shell = int(unit["shell"])
                shell_numerator[shell] += float(np.sum(loss_np) * shell_weight)
                shell_denominator[shell] += float(len(loss_np) * shell_weight)
                loss_window_numerator += float(np.sum(loss_np) * shell_weight)
                loss_window_denominator += float(len(loss_np) * shell_weight)
                window_objective.append(float(loss.detach().cpu()))
                maximum_memory = max(maximum_memory, int(torch.cuda.max_memory_allocated()))

                if global_step % args.loss_log_every == 0 or cursor + 1 == len(order):
                    append_jsonl(output / "loss_trace.jsonl", {
                        "epoch": epoch,
                        "cursor": cursor + 1,
                        "global_step": global_step,
                        "training_weighted_mse_window": loss_window_numerator / loss_window_denominator,
                        "optimizer_objective_window_mean": float(np.mean(window_objective)),
                        "window_units": len(window_objective),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                    })
                    loss_window_numerator = 0.0
                    loss_window_denominator = 0.0
                    window_objective.clear()

                if global_step % args.checkpoint_every == 0 or cursor + 1 == len(order):
                    atomic_torch_save(checkpoint_payload(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch, cursor=cursor + 1, order=order, global_step=global_step,
                        accumulator=accumulator, shell_numerator=shell_numerator,
                        shell_denominator=shell_denominator, history=history,
                        best_score=best_score, best_epoch=best_epoch,
                        maximum_memory=maximum_memory, arguments=arguments,
                        config_sha256=config_sha,
                    ), checkpoint_path)

            if accumulator.patches != len(train_units):
                raise RuntimeError(f"incomplete D0 epoch: {accumulator.patches}/{len(train_units)}")
            if not np.array_equal(
                np.rint(shell_denominator / (1.0 / np.sqrt(expected_shell_voxels))).astype(np.int64),
                expected_shell_voxels,
            ):
                raise RuntimeError("D0 complete-epoch shell exposure mismatch")
            validation = validate_all_units(model, adapter, validation_units, args.device)
            score = float(validation["macro_shell_r2_delta_r7"])
            epoch_row = {
                "epoch": epoch,
                "global_step": global_step,
                "training_weighted_mse": accumulator.mean,
                "training_units": int(accumulator.patches),
                "unique_units_seen": int(len(np.unique(order))),
                "repeat_units": int(len(order) - len(np.unique(order))),
                "unique_unit_fraction": float(len(np.unique(order)) / len(train_units)),
                "training_voxels_by_shell": expected_shell_voxels.tolist(),
                "validation": validation,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "maximum_cuda_memory_bytes": maximum_memory,
                "elapsed_seconds": float(time.time() - started),
            }
            history.append(epoch_row)
            append_jsonl(output / "epoch_history.jsonl", epoch_row)
            print(json.dumps(epoch_row), flush=True)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                atomic_torch_save({
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "model": "U-DENSITY-PHYS-v1",
                    "rotation": args.rotation,
                    "seed": args.seed,
                    "epoch": epoch,
                    "global_step": global_step,
                    "score": score,
                    "target_scaler": adapter.scaler,
                    "normalization": adapter.normalization,
                    "config_sha256": config_sha,
                }, output / "best_checkpoint.pt")
                atomic_json(output / "best_validation_field_report.json", validation)

            next_epoch = epoch + 1
            next_order = epoch_order(
                unit_ids, seed=args.seed, epoch=next_epoch, core_weight=unit_weight
            )
            atomic_torch_save(checkpoint_payload(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=next_epoch, cursor=0, order=next_order, global_step=global_step,
                accumulator=EpochLossAccumulator(), shell_numerator=np.zeros(4),
                shell_denominator=np.zeros(4), history=history,
                best_score=best_score, best_epoch=best_epoch,
                maximum_memory=maximum_memory, arguments=arguments,
                config_sha256=config_sha,
            ), checkpoint_path)

        status = "D0_CANARY_COMPLETE" if args.canary else "D0_TRAINING_SCHEDULE_COMPLETE"
        summary = {
            "schema_version": "p8-density-training-summary-v1",
            "status": status,
            "model": "U-DENSITY-PHYS-v1",
            "rotation": args.rotation,
            "seed": args.seed,
            "run_name": args.run_name,
            "epochs_completed": int(history[-1]["epoch"] if history else 0),
            "best_epoch": int(best_epoch),
            "best_macro_shell_r2_delta_r7": float(best_score),
            "history": history,
            "checkpoint": str(checkpoint_path),
            "elapsed_seconds": float(time.time() - started),
        }
        atomic_json(output / "training_summary.json", summary)
        (output / status).write_text(
            f"rotation={args.rotation} seed={args.seed} best={best_score:.8f}\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
