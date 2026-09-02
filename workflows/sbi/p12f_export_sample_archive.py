#!/usr/bin/env python3
"""Export one standardized, resumable ph006 P12-F sample archive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_train_unet_patch import CHANNELS, model_inputs
from workflows.sbi.p12f_gaussian_controls import (
    ConditionalGaussianUNet,
    sample_correlated_gaussian,
    sample_independent_gaussian,
    sample_shell_correlated_gaussian,
)
from workflows.sbi.p12f_score_diffusion import (
    ConditionalVDiffusionUNet,
    sample_ddim,
)
from workflows.sbi.p12f_train_conditional_field_flow import (
    ConditionalVelocityUNet,
    FieldTargetStore,
    sample_heun,
    unscale,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f_matched_challengers_v1.json"
DEFAULT_CONTRACT = Path(
    "/global/homes/d/dkololgi/p11_contracts/"
    "training_contract_r1_random_repair_v2_20260901"
)
DEFAULT_PHASE_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
METHODS = (
    "gaussian_independent_g0",
    "gaussian_correlated_g1",
    "gaussian_shell_correlated_g2",
    "rectified_flow_f1b",
    "score_diffusion_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract-root", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--phase-root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--panel-marker", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--g1-filter", type=Path)
    parser.add_argument("--g2-filter", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value: dict) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True).encode())


def build_model_and_scaler(
    method: str,
    checkpoint: dict,
    config: dict,
    parent: dict,
    device: str,
) -> tuple[torch.nn.Module, dict]:
    if method.startswith("gaussian_"):
        if (
            checkpoint.get("schema_version")
            != "p12f-matched-challenger-checkpoint-v1"
            or checkpoint.get("method") != "gaussian"
        ):
            raise RuntimeError("Gaussian archive received a non-Gaussian checkpoint")
        model = ConditionalGaussianUNet(
            condition_channels=3,
            base=int(config["matched_contract"]["unet_base"]),
        )
    elif method == "score_diffusion_v1":
        if (
            checkpoint.get("schema_version")
            != "p12f-matched-challenger-checkpoint-v1"
            or checkpoint.get("method") != "diffusion"
        ):
            raise RuntimeError("diffusion archive received the wrong checkpoint")
        model = ConditionalVDiffusionUNet(
            condition_channels=3,
            base=int(config["matched_contract"]["unet_base"]),
        )
    elif method == "rectified_flow_f1b":
        if (
            checkpoint.get("schema_version")
            != "p12f-conditional-field-flow-checkpoint-v1"
        ):
            raise RuntimeError("flow archive received a non-F1b checkpoint")
        model = ConditionalVelocityUNet(
            condition_channels=3,
            base=int(parent["model"]["unet_base"]),
        )
    else:
        raise ValueError(f"unsupported method {method}")
    if checkpoint.get("ph001_opened"):
        raise PermissionError("ph001 appeared in field checkpoint provenance")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint["target_scaler"]


@torch.inference_mode()
def draw_samples(
    method: str,
    model: torch.nn.Module,
    condition: torch.Tensor,
    scaler: dict,
    *,
    draws: int,
    seed: int,
    g1_filter: dict | None,
    g2_filter: dict | None,
    shell: int,
) -> np.ndarray:
    generator = torch.Generator(device=condition.device)
    generator.manual_seed(int(seed))
    if method.startswith("gaussian_"):
        mean, log_std = model(condition)
        if method == "gaussian_independent_g0":
            scaled = sample_independent_gaussian(
                mean,
                log_std,
                draws=draws,
                generator=generator,
            )[:, 0].cpu().numpy()
        elif method == "gaussian_correlated_g1":
            if g1_filter is None:
                raise RuntimeError("G1 sampling requires the frozen residual filter")
            scaled = sample_correlated_gaussian(
                mean[0, 0].cpu().numpy(),
                torch.exp(log_std[0, 0]).cpu().numpy(),
                g1_filter,
                draws=draws,
                seed=seed,
            )
        elif method == "gaussian_shell_correlated_g2":
            if g2_filter is None:
                raise RuntimeError("G2 sampling requires the frozen shell filter")
            scaled = sample_shell_correlated_gaussian(
                mean[0, 0].cpu().numpy(),
                torch.exp(log_std[0, 0]).cpu().numpy(),
                g2_filter,
                shell=shell,
                draws=draws,
                seed=seed,
            )
        else:
            raise ValueError(f"unsupported Gaussian method {method}")
    elif method == "score_diffusion_v1":
        scaled = sample_ddim(
            model,
            condition,
            draws=draws,
            steps=50,
            generator=generator,
        ).cpu().numpy()
    elif method == "rectified_flow_f1b":
        scaled = sample_heun(
            model,
            condition,
            draws=draws,
            steps=12,
            generator=generator,
        ).cpu().numpy()
    else:
        raise ValueError(f"unsupported method {method}")
    samples = unscale(scaled, scaler)
    if samples.shape[0] != draws or not np.all(np.isfinite(samples)):
        raise RuntimeError("field sampler produced invalid draws")
    return samples.astype(np.float32)


def core_bounds(patch) -> np.ndarray:
    start = [int(value.start) for value in patch.core_slice]
    stop = [int(value.stop) for value in patch.core_slice]
    return np.asarray((start, stop), dtype=np.int32)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("field sampling requires a compute GPU")
    config = json.loads(args.config.read_text())
    panel = json.loads(args.panel_marker.read_text())
    if (
        panel.get("schema_version") != "p12f-truth-free-selection-panel-v1"
        or not panel.get("pass")
        or panel.get("selection_uses_truth")
        or panel.get("truth_files_read")
        or panel.get("ph001_opened")
    ):
        raise RuntimeError("sample export requires the passing truth-free ph006 panel")
    if config["roles"]["validation_and_selection"] != "ph006":
        raise RuntimeError("sample archive is frozen to ph006")
    if config["roles"]["sealed_blind_test"] != "ph001":
        raise PermissionError("P12-F blind role changed")
    if args.method == "gaussian_correlated_g1" and args.g1_filter is None:
        raise RuntimeError("G1 requires --g1-filter")
    if args.method != "gaussian_correlated_g1" and args.g1_filter is not None:
        raise RuntimeError("only G1 may consume a residual filter")
    if args.method == "gaussian_shell_correlated_g2" and args.g2_filter is None:
        raise RuntimeError("G2 requires --g2-filter")
    if args.method != "gaussian_shell_correlated_g2" and args.g2_filter is not None:
        raise RuntimeError("only G2 may consume a shell residual filter")

    output = args.output_root / args.method
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "P12F_SAMPLE_ARCHIVE.json"
    if manifest_path.exists():
        print(manifest_path.read_text(), flush=True)
        return
    if any(output.iterdir()) and not args.resume:
        raise RuntimeError("non-empty sample archive requires --resume or a new root")
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    parent = json.loads((REPO_ROOT / config["parent_flow_config"]).read_text())
    model, scaler = build_model_and_scaler(
        args.method, checkpoint, config, parent, args.device
    )
    g1_filter_contract = (
        None if args.g1_filter is None else json.loads(args.g1_filter.read_text())
    )
    g2_filter_contract = (
        None if args.g2_filter is None else json.loads(args.g2_filter.read_text())
    )
    if g1_filter_contract is not None:
        if (
            g1_filter_contract.get("schema_version")
            != "p12f-g1-radial-residual-filter-v2"
            or not g1_filter_contract.get("pass")
            or g1_filter_contract.get("ph001_opened")
            or g1_filter_contract.get("checkpoint_sha256") != sha256(args.checkpoint)
        ):
            raise RuntimeError("G1 residual-filter provenance mismatch")
    if g2_filter_contract is not None:
        if (
            g2_filter_contract.get("schema_version")
            != "p12f-g2-shell-radial-residual-filter-v1"
            or not g2_filter_contract.get("pass")
            or g2_filter_contract.get("ph001_opened")
            or g2_filter_contract.get("validation_phase_read")
            or g2_filter_contract.get("checkpoint_sha256") != sha256(args.checkpoint)
        ):
            raise RuntimeError("G2 shell-filter provenance mismatch")

    loader = P10RandomResponseLoader(args.contract_root, include_blind=False)
    store = FieldTargetStore(args.phase_root, ("ph006",))
    adapter = loader.field_adapter("ph006")
    halo = int(parent["patch"]["context_halo_voxels"])
    alignment = int(parent["patch"]["alignment_voxels"])
    draws = int(config["matched_contract"]["posterior_draws"])
    selected = [int(value) for value in panel["selected_core_id"]]
    scaler_path = output / "target_scaler.json"
    if not scaler_path.exists():
        atomic_json(scaler_path, scaler)
    frozen = {
        "config_sha256": sha256(args.config),
        "panel_sha256": sha256(args.panel_marker),
        "checkpoint_sha256": sha256(args.checkpoint),
        "conditioning_contract_sha256": sha256(
            args.contract_root / "TRAINING_LOADER_READY.json"
        ),
        "target_scaler_sha256": sha256(scaler_path),
        "target_scaler_canonical_sha256": canonical_json_sha256(scaler),
        "g1_filter_sha256": (
            None if args.g1_filter is None else sha256(args.g1_filter)
        ),
        "g2_filter_sha256": (
            None if args.g2_filter is None else sha256(args.g2_filter)
        ),
        "method": args.method,
        "selected_core_id": selected,
        "draws": draws,
    }
    frozen_digest = canonical_json_sha256(frozen)
    run_manifest_path = output / "run_manifest.json"
    if run_manifest_path.exists():
        old = json.loads(run_manifest_path.read_text())
        if old.get("frozen_digest") != frozen_digest:
            raise RuntimeError("sample-archive resume contract changed")
    else:
        atomic_json(
            run_manifest_path,
            {
                "schema_version": "p12f-sample-export-run-v1",
                "created_utc": utc_now(),
                "git_revision_at_launch": git_revision(),
                "frozen_digest": frozen_digest,
                "frozen": frozen,
                "truth_files_read": ["ph006"],
                "ph001_opened": False,
            },
        )
    progress_path = output / "SAMPLE_ARCHIVE_PROGRESS.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {
            "schema_version": "p12f-sample-export-progress-v1",
            "frozen_digest": frozen_digest,
            "entries": [],
            "ph001_opened": False,
        }
    )
    if progress.get("frozen_digest") != frozen_digest:
        raise RuntimeError("sample-export progress contract changed")
    completed = {int(row["core_id"]): row for row in progress["entries"]}
    metadata = {
        int(row["core_id"]): row for row in panel["selected_core_metadata"]
    }
    normalization = loader.field_normalization
    started = time.monotonic()
    for ordinal, core_id in enumerate(selected):
        if core_id in completed:
            path = Path(completed[core_id]["path"])
            if not path.is_file() or sha256(path) != completed[core_id]["sha256"]:
                raise RuntimeError("completed sample artifact changed before resume")
            continue
        patch = adapter.extract(
            core_id,
            halo,
            CHANNELS,
            alignment_voxels=alignment,
        )
        condition, _ = model_inputs(patch, normalization, args.device)
        target_data = store.extract("ph006", patch)
        samples = draw_samples(
            args.method,
            model,
            condition,
            scaler,
            draws=draws,
            seed=42_000 + core_id,
            g1_filter=g1_filter_contract,
            g2_filter=g2_filter_contract,
            shell=int(metadata[core_id]["shell"]),
        )
        counts = np.asarray(
            patch.values[patch.channel_names.index("counts")], dtype=np.float32
        )
        path = output / f"core_{core_id:08d}.npz"
        atomic_npz(
            path,
            delta_samples=samples,
            delta_truth=np.asarray(target_data["delta"], dtype=np.float32),
            support=np.asarray(target_data["support"], dtype=np.uint8),
            angular_response=np.asarray(
                target_data["angular_response"], dtype=np.float32
            ),
            boundary_distance_mpc=np.asarray(
                target_data["boundary_distance"], dtype=np.float32
            ),
            tracer_density=counts / np.float32(5.0**3),
            core_bounds=core_bounds(patch),
            galaxy_frac_index_local=np.asarray(
                patch.authoritative_frac_index_local, dtype=np.float32
            ),
        )
        row = {
            "core_id": core_id,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "seed": 42_000 + core_id,
        }
        progress["entries"].append(row)
        completed[core_id] = row
        atomic_json(progress_path, progress)
        print(
            json.dumps(
                {
                    "method": args.method,
                    "core": ordinal + 1,
                    "total": len(selected),
                    "core_id": core_id,
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if time.monotonic() - started >= args.max_wall_seconds:
            store.close()
            loader.close()
            raise SystemExit(75)

    entries = [completed[value] for value in selected]
    archive = {
        "schema_version": "p12f-sample-archive-v1",
        "created_utc": utc_now(),
        "method": args.method,
        "phase": "ph006",
        "draws": draws,
        "panel_marker": str(args.panel_marker.resolve()),
        "panel_sha256": sha256(args.panel_marker),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "conditioning_contract_sha256": frozen[
            "conditioning_contract_sha256"
        ],
        "target_scaler": str(scaler_path.resolve()),
        "target_scaler_sha256": sha256(scaler_path),
        "target_scaler_canonical_sha256": canonical_json_sha256(scaler),
        "g1_filter": (
            None if args.g1_filter is None else str(args.g1_filter.resolve())
        ),
        "g2_filter": (
            None if args.g2_filter is None else str(args.g2_filter.resolve())
        ),
        "g2_filter_sha256": frozen["g2_filter_sha256"],
        "g1_filter_sha256": frozen["g1_filter_sha256"],
        "entries": entries,
        "truth_files_read": ["ph006"],
        "ph001_opened": False,
        "pass": True,
    }
    atomic_json(manifest_path, archive)
    store.close()
    loader.close()
    print(json.dumps(archive, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
