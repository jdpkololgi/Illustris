#!/usr/bin/env python3
"""Freeze the rotation-0 U-DENSITY-PHYS-v1 voxel training contract."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from astropy.cosmology import Planck18
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
OUTPUT_CORES = ROOT / "p8_density_phys_v1/field_output_tiling/field_output_cores.npz"
TARGET_MANIFEST = ROOT / "p8_density_phys_v1/targets/target_manifest.json"
P8_ROOT = ROOT / "p8_deterministic_v1"
OUTPUT = ROOT / "p8_density_phys_v1/training_contract"
CAP_NAME = {0: "SGC", 1: "NGC"}
SHELLS = ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, default=0, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-cores", type=Path, default=OUTPUT_CORES)
    parser.add_argument("--target-manifest", type=Path, default=TARGET_MANIFEST)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def shell_distance_bounds() -> np.ndarray:
    return np.asarray([
        [
            float(Planck18.comoving_distance(z_low).value),
            float(Planck18.comoving_distance(z_high).value),
        ]
        for z_low, z_high in SHELLS
    ])


def shell_index_for_core(
    start: np.ndarray,
    stop: np.ndarray,
    *,
    origin_mpc: np.ndarray,
    cell_mpc: float,
) -> np.ndarray:
    start = np.asarray(start, dtype=np.int64)
    stop = np.asarray(stop, dtype=np.int64)
    axes = [
        origin_mpc[a]
        + (np.arange(start[a], stop[a], dtype=np.float64) + 0.5) * cell_mpc
        for a in range(3)
    ]
    radius = np.sqrt(
        axes[0][:, None, None] ** 2
        + axes[1][None, :, None] ** 2
        + axes[2][None, None, :] ** 2
    )
    result = np.full(radius.shape, -1, dtype=np.int8)
    for shell, (low, high) in enumerate(shell_distance_bounds()):
        result[(radius >= low) & (radius < high)] = shell
    return result


class RunningMoments:
    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        if not np.all(np.isfinite(values)):
            raise RuntimeError("non-finite privileged density target")
        self.n += int(values.size)
        self.total += float(values.sum(dtype=np.float64))
        self.total_square += float(np.square(values).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def as_dict(self) -> dict:
        if self.n == 0:
            raise RuntimeError("empty density-target moments")
        mean = self.total / self.n
        variance = max(self.total_square / self.n - mean**2, 0.0)
        return {
            "n": int(self.n),
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
        }


def unit_array(rows: list[tuple]) -> np.ndarray:
    dtype = np.dtype([
        ("output_core_id", "i4"),
        ("nominal_core_id", "i4"),
        ("cap", "u1"),
        ("fold", "u1"),
        ("shell", "u1"),
        ("supported_voxels", "i4"),
        ("unit_weight", "f8"),
    ])
    return np.asarray(rows, dtype=dtype)


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output / f"rotation_{args.rotation}"
    output.mkdir(parents=True, exist_ok=True)
    cores = np.load(args.output_cores, mmap_mode="r")
    target_manifest = json.loads(args.target_manifest.read_text())
    roles_path = args.p8_root / f"rotation_{args.rotation}/roles.json"
    roles = json.loads(roles_path.read_text())
    train_folds = set(int(v) for v in roles["train_folds"])
    validation_fold = int(roles["validation_fold"])
    development_fold = int(roles["development_test_fold"])

    handles = {
        cap_id: h5py.File(target_manifest["components"][cap_name]["file"], "r")
        for cap_id, cap_name in CAP_NAME.items()
    }
    unit_rows: list[tuple] = []
    moments = {"train": RunningMoments(), "validation": RunningMoments(), "development": RunningMoments()}
    voxel_count = {"train": np.zeros(4, dtype=np.int64), "validation": np.zeros(4, dtype=np.int64), "development": np.zeros(4, dtype=np.int64)}
    role_by_fold = {fold: "train" for fold in train_folds}
    role_by_fold[validation_fold] = "validation"
    role_by_fold[development_fold] = "development"
    try:
        for row in range(len(cores["output_core_id"])):
            if not bool(cores["owns_density_loss"][row]):
                continue
            fold = int(cores["fold"][row])
            if fold not in role_by_fold:
                raise RuntimeError(f"density-loss owner row has unexpected fold {fold}")
            role = role_by_fold[fold]
            cap = int(cores["cap"][row])
            cap_name = CAP_NAME[cap]
            grid = target_manifest["components"][cap_name]["grid"]
            start = np.asarray(cores["voxel_start"][row], dtype=np.int64)
            stop = np.asarray(cores["voxel_stop"][row], dtype=np.int64)
            selection = tuple(slice(int(start[a]), int(stop[a])) for a in range(3))
            support = np.asarray(handles[cap]["science_support"][selection], dtype=bool)
            if not np.any(support):
                continue
            target = np.asarray(handles[cap]["delta_r7"][selection], dtype=np.float32)
            shell_index = shell_index_for_core(
                start, stop,
                origin_mpc=np.asarray(grid["origin_mpc"], dtype=np.float64),
                cell_mpc=float(grid["cell_mpc"]),
            )
            if np.any(support & (shell_index < 0)):
                raise RuntimeError("science-supported voxel lies outside the four shell contract")
            moments[role].add(target[support])
            for shell in range(4):
                count = int(np.sum(support & (shell_index == shell)))
                if count:
                    voxel_count[role][shell] += count
                    unit_rows.append((
                        int(cores["output_core_id"][row]),
                        int(cores["nominal_core_id"][row]),
                        cap,
                        fold,
                        shell,
                        count,
                        0.0,
                    ))
    finally:
        for handle in handles.values():
            handle.close()

    units = unit_array(unit_rows)
    if len(units) == 0 or np.any(units["nominal_core_id"] < 0):
        raise RuntimeError("training contract contains no units or an inference-only loss owner")
    for role, folds in (
        ("train", train_folds),
        ("validation", {validation_fold}),
        ("development", {development_fold}),
    ):
        selected = np.isin(units["fold"], list(folds))
        counts = voxel_count[role]
        if np.any(counts == 0):
            raise RuntimeError(f"{role} lacks one or more density shells: {counts}")
        shell_weight = 1.0 / np.sqrt(counts.astype(np.float64))
        units["unit_weight"][selected] = (
            units["supported_voxels"][selected] * shell_weight[units["shell"][selected]]
        )
    if np.any(units["unit_weight"] <= 0):
        raise RuntimeError("all density units require positive frozen weight")

    units_path = output / "density_units.npy"
    np.save(units_path, units)
    scaler = moments["train"].as_dict()
    scaler.update({
        "fit_scope": "delta_R7 in exact nominal owners on rotation training folds only",
        "transform": "(delta_R7 - mean) / std",
    })
    scaler_path = output / "target_scaler.json"
    atomic_json(scaler_path, scaler)
    train_units = units[np.isin(units["fold"], list(train_folds))]
    validation_units = units[units["fold"] == validation_fold]
    development_units = units[units["fold"] == development_fold]
    canary = train_units[int(np.argmax(train_units["supported_voxels"]))]
    config = {
        "schema_version": "p8-density-training-contract-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "model": "U-DENSITY-PHYS-v1",
        "rotation": int(args.rotation),
        "seed": int(args.seed),
        "roles": roles,
        "inputs": {
            "channels": ["counts", "derived density proxy", "exposure_apodized"],
            "channel_mapping": "identical to U-PATCH-BRIGHT_REFERENCE",
            "selection_normalization": "P6 rotation-specific train-fold-only normalization",
            "privileged_target_is_input": False,
            "z_cosmo_is_input": False,
        },
        "architecture": {
            "family": "P8 UNet3D",
            "base_channels": 24,
            "output_channels": 1,
            "normalization": "per-voxel ChannelLayerNorm3d",
            "context_halo_voxels": 24,
            "context_halo_mpc": 120.0,
            "alignment_voxels": 8,
        },
        "objective": {
            "target": "R=7 Mpc/h smoothed matter contrast delta_R7",
            "loss": "training-fold-standardized voxelwise MSE",
            "ownership": "exact half-open output core AND science support AND shell",
            "shell_weight": "N_shell^-0.5, matching the P8 galaxy-row convention",
            "epoch": "every positive (nominal exact-owner core, shell) unit exactly once",
            "direct_eigenvalue_or_tensor_loss": False,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": 0.002,
            "weight_decay": 0.0001,
            "gradient_clip": 5.0,
            "epochs": 20,
            "early_stopping": False,
            "scheduler": "cosine to zero over exactly 20 complete epochs",
            "checkpoint_every_updates": 250,
            "loss_log_every_updates": 25,
            "checkpoint_selection": "maximum complete-validation macro-shell R2(delta_R7)",
        },
        "counts": {
            "units_total": int(len(units)),
            "units_train": int(len(train_units)),
            "units_validation": int(len(validation_units)),
            "units_development": int(len(development_units)),
            "voxels_train_by_shell": voxel_count["train"].tolist(),
            "voxels_validation_by_shell": voxel_count["validation"].tolist(),
            "voxels_development_by_shell": voxel_count["development"].tolist(),
        },
        "canary_unit": {name: canary[name].item() for name in canary.dtype.names},
        "artifacts": {
            "density_units": str(units_path),
            "density_units_sha256": sha256(units_path),
            "target_scaler": str(scaler_path),
            "target_scaler_sha256": sha256(scaler_path),
        },
        "inputs_provenance": {
            "output_cores": str(args.output_cores),
            "output_cores_sha256": sha256(args.output_cores),
            "target_manifest": str(args.target_manifest),
            "target_manifest_sha256": sha256(args.target_manifest),
            "roles": str(roles_path),
            "roles_sha256": sha256(roles_path),
        },
        "gates": {
            "all_three_roles_have_all_four_shells": True,
            "all_units_have_positive_voxel_count_and_weight": True,
            "only_nominal_p4_cores_own_loss": True,
            "target_scaler_is_training_fold_only": True,
            "inference_only_rows_own_no_loss": True,
            "privileged_target_absent_from_inputs": True,
        },
        "pass": True,
        "elapsed_seconds": float(time.time() - started),
    }
    config_path = output / "d0_config.json"
    atomic_json(config_path, config)
    (output / "D0_TRAINING_CONTRACT_READY").write_text(
        f"rotation={args.rotation} units={len(units)} config_sha256={sha256(config_path)}\n"
    )
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
