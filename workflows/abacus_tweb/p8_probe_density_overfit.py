#!/usr/bin/env python3
"""One exact-owner overfit gate for U-DENSITY-PHYS-v1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from workflows.abacus_tweb.p8_train_unet_patch import UNet3D


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
OUTPUT = ROOT / "p8_density_phys_v1/overfit_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contract-root", type=Path, default=TRAINING_CONTRACT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--maximum-final-to-initial", type=float, default=0.25)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("density overfit probe requires an interactive CUDA allocation")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()
    output = args.output / f"rotation_{args.rotation}/seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    contract_dir = args.contract_root / f"rotation_{args.rotation}"
    config_path = contract_dir / "d0_config.json"
    config = json.loads(config_path.read_text())
    canary = config["canary_unit"]
    model = UNet3D(in_channels=3, latent_channels=1, base=24).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    with DensityUnitAdapter(rotation=args.rotation, contract_root=args.contract_root) as adapter:
        unit = adapter.find_unit(canary["output_core_id"], canary["shell"])
        patch, values, target, mask, diagnostics = adapter.extract(unit, args.device)
        model.train()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            prediction = extract_core_prediction(model(values), patch.core_slice)
            loss = torch.mean((prediction[mask] - target[mask]) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite overfit loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if step <= 10 or step % 10 == 0 or step == args.steps:
                row = {"step": step, "standardized_density_mse": float(loss.detach().cpu())}
                history.append(row)
                print(json.dumps(row), flush=True)
    first = np.asarray([row["standardized_density_mse"] for row in history if row["step"] <= 10])
    last = np.asarray([row["standardized_density_mse"] for row in history[-5:]])
    initial = float(np.mean(first))
    final = float(np.mean(last))
    ratio = final / max(initial, 1e-30)
    passed = bool(np.isfinite(ratio) and ratio <= args.maximum_final_to_initial)
    checkpoint = output / "overfit_checkpoint.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model": "U-DENSITY-PHYS-v1-overfit",
        "rotation": args.rotation,
        "seed": args.seed,
        "steps": args.steps,
        "unit": canary,
        "config_sha256": sha256(config_path),
    }, checkpoint)
    report = {
        "schema_version": "p8-density-overfit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "stage": "P8.9 U-DENSITY-PHYS-v1 one-core overfit",
        "rotation": int(args.rotation),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "learning_rate": float(args.lr),
        "unit": canary,
        "extraction": diagnostics,
        "history": history,
        "initial_window_mean_mse": initial,
        "final_window_mean_mse": final,
        "final_to_initial_ratio": ratio,
        "threshold": {"maximum_final_to_initial": float(args.maximum_final_to_initial)},
        "gates": {
            "loss_finite": True,
            "loss_contracts_by_required_factor": passed,
            "privileged_target_is_input": False,
        },
        "pass": passed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "elapsed_seconds": float(time.time() - started),
    }
    report_path = output / "overfit_report.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
