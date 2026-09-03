#!/usr/bin/env python3
"""Export matched ph006 P12-F3 wide-G1 and hybrid field samples.

The two hybrid arms share the same frozen wide-context G1 conditional mean and
the same high-frequency G1 residual realization.  They differ only in whether
the learned low-mode flow sees a local (h8) or wide (h24) response-conditioned
view.  This makes the local/wide comparison an actual context intervention.
"""
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
import h5py

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_train_unet_patch import CHANNELS, model_inputs
from workflows.sbi.p12f3_hierarchical_lowmode import (
    build_low_mode_model,
    crop_tensor_to_patch,
    pool_low_mode_state,
    sample_heun,
    upsample_low_mode_draws,
)
from workflows.sbi.p12f3_train_lowmode_flow import load_g1_model
from workflows.sbi.p12f_gaussian_controls import correlated_unit_residuals


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f3_hierarchical_lowmode_v1.json"
DEFAULT_SOURCE_PANEL = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f_dependency_rescue_v2/"
    "evaluation_sufficiency_seed42/panel_1024/P12F_PH006_PANEL_1024.json"
)
METHODS = (
    "g1_wide_crop_h8",
    "g1_wide_h24",
    "hybrid_local_h8",
    "hybrid_wide_h24",
)


class EvaluationTargetStore:
    """Visible-phase density and random-response reader for archive export."""

    def __init__(self, phase_root: Path):
        marker = phase_root / "ph006/p12f_field_targets_v1/FIELD_TARGET_READY.json"
        self.manifest = json.loads(marker.read_text())
        if (
            not self.manifest.get("pass")
            or self.manifest.get("phase") != "ph006"
            or self.manifest.get("ph001_opened")
        ):
            raise RuntimeError("ph006 field target marker is not passing")
        self.targets: dict[int, h5py.File] = {}
        self.responses: dict[int, h5py.File] = {}

    def _component(self, cap: int) -> dict:
        return self.manifest["components"]["NGC" if int(cap) == 1 else "SGC"]

    def _target(self, cap: int) -> h5py.File:
        cap = int(cap)
        if cap not in self.targets:
            self.targets[cap] = h5py.File(self._component(cap)["file"], "r")
        return self.targets[cap]

    def _response(self, cap: int) -> h5py.File:
        cap = int(cap)
        if cap not in self.responses:
            self.responses[cap] = h5py.File(
                self._component(cap)["support_random_source"], "r"
            )
        return self.responses[cap]

    def extract(self, patch) -> dict[str, np.ndarray]:
        selection = tuple(
            slice(int(left), int(right))
            for left, right in zip(patch.context_start, patch.context_stop)
        )
        target = self._target(patch.cap)
        response = self._response(patch.cap)
        output = {
            "delta": np.asarray(target["delta_r7"][selection], dtype=np.float32),
            "support": np.asarray(response["support_random"][selection], dtype=bool),
            "angular_response": np.asarray(response["angular_response"][selection], dtype=np.float32),
            "boundary_distance": np.asarray(response["distance_to_support_boundary"][selection], dtype=np.float32),
        }
        if output["delta"].shape != patch.values.shape[1:]:
            raise RuntimeError("P12-F3 target and response geometry mismatch")
        return output

    def close(self) -> None:
        for handle in (*self.targets.values(), *self.responses.values()):
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-panel", type=Path, default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--low-checkpoint", type=Path)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--draw-batch", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_200.0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def selected_subpanel(source: dict, expected: int = 256) -> list[dict]:
    """Reproduce the registered causal-autopsy every-fourth-by-shell panel."""
    rows = list(source["selected_core_metadata"])
    selected: list[dict] = []
    for shell in range(4):
        group = sorted(
            (row for row in rows if int(row["shell"]) == shell),
            key=lambda row: int(row["core_id"]),
        )
        selected.extend(group[::4])
    selected.sort(key=lambda row: int(row["core_id"]))
    counts = np.bincount([int(row["shell"]) for row in selected], minlength=4)
    if len(selected) != expected or not np.array_equal(counts, np.full(4, expected // 4)):
        raise RuntimeError("P12-F3 evaluation subpanel is not 64 cores per shell")
    return selected


def freeze_subpanel(source_path: Path, output: Path) -> dict:
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != "p12f-truth-free-selection-panel-v1"
        or not source.get("pass")
        or source.get("selection_uses_truth")
        or source.get("truth_files_read")
        or source.get("ph001_opened")
        or len(source.get("selected_core_id", [])) != 1024
    ):
        raise RuntimeError("P12-F3 requires the frozen truth-free 1024-core ph006 panel")
    rows = selected_subpanel(source)
    marker = {
        "schema_version": "p12f-truth-free-selection-panel-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph006",
        "selected_core_id": [int(row["core_id"]) for row in rows],
        "selected_core_metadata": rows,
        "shell_counts": [64, 64, 64, 64],
        "selection_rule": "source 1024-core panel; core-id sorted every-fourth subpanel 0 within shell",
        "source_panel": str(source_path.resolve()),
        "source_panel_sha256": sha256(source_path),
        "selection_uses_truth": False,
        "truth_files_read": [],
        "target_store_instantiated": False,
        "ph001_opened": False,
        "open_count": 0,
        "pass": True,
    }
    if output.exists():
        prior = json.loads(output.read_text())
        ignored = {"created_utc", "git_revision"}
        stable = {key: value for key, value in marker.items() if key not in ignored}
        old = {key: value for key, value in prior.items() if key not in ignored}
        if old != stable:
            raise RuntimeError("existing P12-F3 evaluation panel changed")
        return prior
    atomic_json(output, marker)
    return marker


def lowpass_numpy(field: np.ndarray, *, voxel_mpc_h: float, maximum_k: float) -> np.ndarray:
    """Apply the exact registered non-DC physical low-pass per draw."""
    values = np.asarray(field, dtype=np.float32)
    shape = values.shape[-3:]
    kx = 2.0 * np.pi * np.fft.fftfreq(shape[0], d=voxel_mpc_h)[:, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(shape[1], d=voxel_mpc_h)[None, :, None]
    kz = 2.0 * np.pi * np.fft.rfftfreq(shape[2], d=voxel_mpc_h)[None, None, :]
    radius = np.sqrt(kx * kx + ky * ky + kz * kz)
    mask = (radius > 0.0) & (radius <= maximum_k)
    mask[0, 0, 0] = False
    result = np.empty_like(values)
    for index in range(len(values)):
        coefficient = np.fft.rfftn(values[index], norm="ortho")
        result[index] = np.fft.irfftn(
            coefficient * mask, s=shape, axes=(-3, -2, -1), norm="ortho"
        ).real.astype(np.float32)
    return result


def core_bounds(patch) -> np.ndarray:
    return np.asarray(
        (
            [int(value.start) for value in patch.core_slice],
            [int(value.stop) for value in patch.core_slice],
        ),
        dtype=np.int32,
    )


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_low_model(
    path: Path, *, arm: str, device: str, config: dict, config_path: Path
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    trained_path = path.parent / "P12F3_TRAINED.json"
    trained = json.loads(trained_path.read_text())
    run_manifest = json.loads((path.parent / "run_manifest.json").read_text())
    if (
        checkpoint.get("schema_version") != "p12f3-lowmode-checkpoint-v1"
        or checkpoint.get("ph001_opened")
        or int(checkpoint.get("update", -1)) != int(config["training"]["science_updates"])
        or trained.get("schema_version") != "p12f3-lowmode-trained-v1"
        or not trained.get("pass")
        or trained.get("arm") != arm
        or int(trained.get("update", -1)) != int(config["training"]["science_updates"])
        or trained.get("checkpoint_sha256") != sha256(path)
        or trained.get("frozen_digest") != run_manifest.get("frozen_digest")
        or checkpoint.get("frozen_digest") != run_manifest.get("frozen_digest")
        or run_manifest.get("frozen", {}).get("arm") != arm
        or run_manifest.get("frozen", {}).get("config_sha256") != sha256(config_path)
        or trained.get("ph001_opened")
        or run_manifest.get("ph001_opened")
    ):
        raise RuntimeError("P12-F3 low-mode checkpoint is incomplete or unsafe")
    model = build_low_mode_model(
        condition_channels=int(config["model"]["condition_channels"]),
        base=int(config["model"]["unet_base"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, checkpoint


@torch.inference_mode()
def sample_low_mode(
    model,
    condition: torch.Tensor,
    *,
    draws: int,
    steps: int,
    seed: int,
    batch: int,
) -> np.ndarray:
    pooled = pool_low_mode_state(condition, 2)
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    parts = []
    for start in range(0, draws, batch):
        count = min(batch, draws - start)
        coarse = sample_heun(
            model, pooled, draws=count, steps=steps, generator=generator
        )
        full = upsample_low_mode_draws(coarse, tuple(condition.shape[-3:]))
        parts.append(full.cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P12-F3 archive export requires a compute GPU")
    if args.draw_batch <= 0:
        raise ValueError("draw batch must be positive")
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != "p12f3-hierarchical-lowmode-v1":
        raise RuntimeError("unsupported P12-F3 config")
    if config["roles"]["validation"] != "ph006" or config["roles"]["sealed_blind_test"] != "ph001":
        raise PermissionError("P12-F3 evaluation phase contract changed")
    path_contract = json.dumps(
        {
            "config_sources": config.get("sources", {}),
            "source_panel": str(args.source_panel),
            "low_checkpoint": None if args.low_checkpoint is None else str(args.low_checkpoint),
            "output_root": str(args.output_root),
        }
    ).lower()
    if "ph001" in path_contract:
        raise PermissionError("ph001 path entered P12-F3 ph006 evaluation")
    is_hybrid = args.method.startswith("hybrid_")
    if is_hybrid != (args.low_checkpoint is not None):
        raise RuntimeError("hybrid methods require exactly one low-mode checkpoint")

    args.output_root.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_root / "P12F3_PH006_PANEL_256.json"
    panel = freeze_subpanel(args.source_panel, panel_path)
    method_output = args.output_root / args.method
    method_output.mkdir(parents=True, exist_ok=True)
    archive_path = method_output / "P12F_SAMPLE_ARCHIVE.json"
    if archive_path.exists():
        print(archive_path.read_text(), flush=True)
        return
    if any(method_output.iterdir()) and not args.resume:
        raise RuntimeError("non-empty P12-F3 archive requires --resume")

    g1_model, scaler = load_g1_model(config, args.device)
    filter_path = Path(config["sources"]["g1_filter"])
    filter_contract = json.loads(filter_path.read_text())
    if filter_contract.get("schema_version") != "p12f-g1-radial-residual-filter-v2":
        raise RuntimeError("P12-F3 hybrid requires the frozen G1 filter")
    low_model = None
    arm = None
    if is_hybrid:
        arm = "local_h8" if args.method == "hybrid_local_h8" else "wide_h24"
        low_model, _ = load_low_model(
            args.low_checkpoint,
            arm=arm,
            device=args.device,
            config=config,
            config_path=args.config,
        )

    contract_root = Path(config["sources"]["conditioning_contract"])
    phase_root = Path(config["sources"]["phase_root"])
    loader = P10RandomResponseLoader(contract_root, include_blind=False)
    store = EvaluationTargetStore(phase_root)
    adapter = loader.field_adapter("ph006")
    normalization = loader.field_normalization
    common_halo = int(config["patch"]["common_target_halo_voxels"])
    crop_control = args.method == "g1_wide_crop_h8"
    arm_halo = (
        int(config["arms"]["local_h8"]["conditioning_halo_voxels"])
        if crop_control
        else common_halo if arm is None else int(config["arms"][arm]["conditioning_halo_voxels"])
    )
    alignment = int(config["patch"]["alignment_voxels"])
    draws = int(config["science_gate"]["evaluation_draws"])
    voxel = float(config["target"]["voxel_mpc_h"])
    maximum_k = float(config["target"]["maximum_k_h_mpc_inclusive"])
    entries = []
    progress_path = method_output / "SAMPLE_ARCHIVE_PROGRESS.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        entries = list(progress["entries"])
    frozen = {
        "config_sha256": sha256(args.config),
        "panel_sha256": sha256(panel_path),
        "source_panel_sha256": sha256(args.source_panel),
        "g1_checkpoint_sha256": sha256(Path(config["sources"]["g1_checkpoint"])),
        "g1_filter_sha256": sha256(filter_path),
        "low_checkpoint_sha256": None if args.low_checkpoint is None else sha256(args.low_checkpoint),
        "method": args.method,
        "draws": draws,
        "selected_core_id": panel["selected_core_id"],
        "source_hashes": {
            "exporter": sha256(Path(__file__)),
            "hierarchical_model": sha256(
                REPO_ROOT / "workflows/sbi/p12f3_hierarchical_lowmode.py"
            ),
            "common_evaluator": sha256(
                REPO_ROOT / "workflows/sbi/p12f_common_evaluator.py"
            ),
        },
    }
    digest = canonical_digest(frozen)
    if progress_path.exists() and progress.get("frozen_digest") != digest:
        raise RuntimeError("P12-F3 sample progress contract changed")
    complete = {int(row["core_id"]): row for row in entries}
    run_manifest = method_output / "run_manifest.json"
    if run_manifest.exists():
        if json.loads(run_manifest.read_text()).get("frozen_digest") != digest:
            raise RuntimeError("P12-F3 archive resume contract changed")
    else:
        atomic_json(run_manifest, {
            "schema_version": "p12f3-hybrid-export-run-v1",
            "created_utc": utc_now(),
            "git_revision_at_launch": git_revision(),
            "frozen_digest": digest,
            "frozen": frozen,
            "truth_files_read": ["ph006"],
            "ph001_opened": False,
        })
    started = time.monotonic()
    for ordinal, core_id in enumerate(panel["selected_core_id"]):
        core_id = int(core_id)
        if core_id in complete:
            row = complete[core_id]
            if sha256(Path(row["path"])) != row["sha256"]:
                raise RuntimeError("completed P12-F3 core artifact changed")
            continue
        wide = adapter.extract(core_id, common_halo, CHANNELS, alignment_voxels=alignment)
        patch = wide if arm_halo == common_halo else adapter.extract(
            core_id, arm_halo, CHANNELS, alignment_voxels=alignment
        )
        if not np.array_equal(wide.authoritative_parent_id, patch.authoritative_parent_id):
            raise RuntimeError("P12-F3 local/wide authoritative parent mismatch")
        wide_condition, _ = model_inputs(wide, normalization, args.device)
        mean, log_std = g1_model(wide_condition)
        mean_wide = mean[0, 0].cpu().numpy().astype(np.float32)
        std_wide = torch.exp(log_std[0, 0]).cpu().numpy().astype(np.float32)
        seed = 43_000 + core_id
        unit = correlated_unit_residuals(
            filter_contract, draws=draws, seed=seed, shape=tuple(mean_wide.shape)
        )
        residual_wide = std_wide[None] * unit
        low_g1 = lowpass_numpy(residual_wide, voxel_mpc_h=voxel, maximum_k=maximum_k)
        high_wide = residual_wide - low_g1
        if args.method in {"g1_wide_h24", "g1_wide_crop_h8"}:
            wide_scaled = mean_wide[None] + residual_wide
            if crop_control:
                scaled = crop_tensor_to_patch(
                    torch.from_numpy(wide_scaled[:, None]),
                    source_start=wide.context_start,
                    target_start=patch.context_start,
                    target_stop=patch.context_stop,
                )[:, 0].numpy()
            else:
                scaled = wide_scaled
        else:
            arm_condition, _ = model_inputs(patch, normalization, args.device)
            learned_low = sample_low_mode(
                low_model,
                arm_condition,
                draws=draws,
                steps=int(config["model"]["ode_steps"]),
                seed=seed + 100_000_000,
                batch=args.draw_batch,
            )
            mean_arm = crop_tensor_to_patch(
                torch.from_numpy(mean_wide[None, None]),
                source_start=wide.context_start,
                target_start=patch.context_start,
                target_stop=patch.context_stop,
            )[0, 0].numpy()
            high_arm = crop_tensor_to_patch(
                torch.from_numpy(high_wide[:, None]),
                source_start=wide.context_start,
                target_start=patch.context_start,
                target_stop=patch.context_stop,
            )[:, 0].numpy()
            scaled = mean_arm[None] + high_arm + learned_low
        samples = (scaled * np.float32(scaler["std"]) + np.float32(scaler["mean"])).astype(np.float32)
        target = store.extract(patch)
        counts = np.asarray(patch.values[patch.channel_names.index("counts")], dtype=np.float32)
        path = method_output / f"core_{core_id:08d}.npz"
        atomic_npz(
            path,
            delta_samples=samples,
            delta_truth=np.asarray(target["delta"], dtype=np.float32),
            support=np.asarray(target["support"], dtype=np.uint8),
            angular_response=np.asarray(target["angular_response"], dtype=np.float32),
            boundary_distance_mpc=np.asarray(target["boundary_distance"], dtype=np.float32),
            tracer_density=counts / np.float32(voxel**3),
            core_bounds=core_bounds(patch),
            galaxy_frac_index_local=np.asarray(patch.authoritative_frac_index_local, dtype=np.float32),
        )
        row = {"core_id": core_id, "path": str(path.resolve()), "sha256": sha256(path), "seed": seed}
        entries.append(row)
        complete[core_id] = row
        atomic_json(progress_path, {
            "schema_version": "p12f3-sample-export-progress-v1",
            "frozen_digest": digest,
            "entries": entries,
            "ph001_opened": False,
        })
        print(json.dumps({"method": args.method, "core": ordinal + 1, "total": len(panel["selected_core_id"]), "core_id": core_id, "elapsed_seconds": time.monotonic() - started}), flush=True)
        if time.monotonic() - started >= args.max_wall_seconds:
            store.close(); loader.close(); raise SystemExit(75)
    ordered = [complete[int(value)] for value in panel["selected_core_id"]]
    scaler_path = method_output / "target_scaler.json"
    atomic_json(scaler_path, scaler)
    archive = {
        "schema_version": "p12f-sample-archive-v1",
        "created_utc": utc_now(),
        "method": args.method,
        "phase": "ph006",
        "draws": draws,
        "panel_marker": str(panel_path.resolve()),
        "panel_sha256": sha256(panel_path),
        "checkpoint": str((Path(config["sources"]["g1_checkpoint"]) if args.low_checkpoint is None else args.low_checkpoint).resolve()),
        "checkpoint_sha256": frozen["g1_checkpoint_sha256"] if args.low_checkpoint is None else frozen["low_checkpoint_sha256"],
        "g1_checkpoint_sha256": frozen["g1_checkpoint_sha256"],
        "g1_filter_sha256": frozen["g1_filter_sha256"],
        "conditioning_contract_sha256": sha256(contract_root / "TRAINING_LOADER_READY.json"),
        "target_scaler": str(scaler_path.resolve()),
        "target_scaler_sha256": sha256(scaler_path),
        "entries": ordered,
        "truth_files_read": ["ph006"],
        "ph001_opened": False,
        "pass": True,
    }
    atomic_json(archive_path, archive)
    store.close(); loader.close()
    print(json.dumps(archive, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
