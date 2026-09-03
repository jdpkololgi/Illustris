#!/usr/bin/env python3
"""Export matched P12-F3-L2 Fourier-Gaussian and Fourier-flow ph006 samples."""
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
from workflows.sbi.p12f3_export_hybrid_archive import (
    EvaluationTargetStore,
    atomic_npz,
    core_bounds,
    freeze_subpanel,
    lowpass_numpy,
)
from workflows.sbi.p12f3_fourier_modes import (
    ConditionalFourierVelocityUNet,
    build_fourier_layout,
    sample_fourier_heun,
    unpack_fourier_components,
    unwhiten_components,
)
from workflows.sbi.p12f3_train_lowmode_flow import load_g1_model
from workflows.sbi.p12f_gaussian_controls import correlated_unit_residuals


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f3_fourier_lowmode_v1.json"
DEFAULT_SOURCE_PANEL = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f_dependency_rescue_v2/"
    "evaluation_sufficiency_seed42/panel_1024/P12F_PH006_PANEL_1024.json"
)
METHODS = ("fourier_gaussian_h24", "fourier_flow_h24")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-panel", type=Path, default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--flow-checkpoint", type=Path)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--draw-batch", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=6600.0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_flow(path: Path, config: dict, config_path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    terminal_path = path.parent / "P12F3L2_FOURIER_FLOW_TRAINED.json"
    terminal = json.loads(terminal_path.read_text())
    run = json.loads((path.parent / "run_manifest.json").read_text())
    if (
        checkpoint.get("schema_version") != "p12f3l2-fourier-flow-checkpoint-v1"
        or int(checkpoint.get("update", -1)) != int(config["training"]["science_updates"])
        or checkpoint.get("ph001_opened")
        or terminal.get("schema_version") != "p12f3l2-fourier-flow-trained-v1"
        or not terminal.get("pass")
        or terminal.get("checkpoint_sha256") != sha256(path)
        or terminal.get("frozen_digest") != checkpoint.get("frozen_digest")
        or run.get("frozen_digest") != checkpoint.get("frozen_digest")
        or run.get("frozen", {}).get("config_sha256") != sha256(config_path)
        or run.get("ph001_opened")
    ):
        raise RuntimeError("P12-F3-L2 flow checkpoint is incomplete or unsafe")
    model = ConditionalFourierVelocityUNet(
        condition_channels=int(config["model"]["condition_channels"]),
        base=int(config["model"]["unet_base"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, terminal


@torch.inference_mode()
def sample_low(
    method: str,
    model,
    condition: torch.Tensor,
    *,
    layout,
    whitening: dict,
    draws: int,
    steps: int,
    seed: int,
    batch: int,
) -> np.ndarray:
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    if method == "fourier_gaussian_h24":
        standard = torch.randn(
            (draws, layout.components),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        physical = unwhiten_components(standard, whitening, layout)
        return unpack_fourier_components(physical, layout)[:, 0].cpu().numpy().astype(np.float32)
    parts = []
    for start in range(0, draws, batch):
        count = min(batch, draws - start)
        part = sample_fourier_heun(
            model, condition, layout=layout, whitening=whitening,
            draws=count, steps=steps, generator=generator,
        )
        parts.append(part.cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P12-F3-L2 archive export requires CUDA")
    if (args.method == "fourier_flow_h24") != (args.flow_checkpoint is not None):
        raise RuntimeError("exactly the Fourier flow method requires a flow checkpoint")
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != "p12f3-fourier-lowmode-v1":
        raise RuntimeError("unsupported P12-F3-L2 config")
    if config["roles"]["validation"] != "ph006" or config["roles"]["sealed_blind_test"] != "ph001":
        raise PermissionError("P12-F3-L2 phase roles changed")
    if "ph001" in json.dumps({"sources": config["sources"], "panel": str(args.source_panel), "output": str(args.output_root)}).lower():
        raise PermissionError("ph001 path entered P12-F3-L2 evaluation")
    whitening_path = Path(config["sources"]["whitening"])
    whitening_marker = json.loads(whitening_path.read_text())
    if not whitening_marker.get("pass") or whitening_marker.get("validation_phase_used_for_fit") or whitening_marker.get("ph001_opened"):
        raise RuntimeError("unsafe P12-F3-L2 whitening marker")
    whitening = whitening_marker["whitening"]

    args.output_root.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_root / "P12F3L2_PH006_PANEL_256.json"
    panel = freeze_subpanel(args.source_panel, panel_path)
    method_output = args.output_root / args.method
    method_output.mkdir(parents=True, exist_ok=True)
    archive_path = method_output / "P12F_SAMPLE_ARCHIVE.json"
    if archive_path.exists():
        print(archive_path.read_text(), flush=True); return
    if any(method_output.iterdir()) and not args.resume:
        raise RuntimeError("non-empty Fourier archive requires --resume")

    g1_model, scaler = load_g1_model(config, args.device)
    filter_path = Path(config["sources"]["g1_filter"])
    filter_contract = json.loads(filter_path.read_text())
    flow_model = None
    if args.flow_checkpoint is not None:
        flow_model, _ = load_flow(args.flow_checkpoint, config, args.config, args.device)
    contract_root = Path(config["sources"]["conditioning_contract"])
    phase_root = Path(config["sources"]["phase_root"])
    loader = P10RandomResponseLoader(contract_root, include_blind=False)
    store = EvaluationTargetStore(phase_root)
    adapter = loader.field_adapter("ph006")
    halo = int(config["patch"]["conditioning_halo_voxels"])
    alignment = int(config["patch"]["alignment_voxels"])
    voxel = float(config["target"]["voxel_mpc_h"])
    edges = tuple(float(value) for value in config["target"]["band_edges_h_mpc"])
    maximum_k = edges[-1]
    draws = int(config["science_gate"]["evaluation_draws"])
    entries = []
    progress_path = method_output / "SAMPLE_ARCHIVE_PROGRESS.json"
    if progress_path.exists():
        entries = list(json.loads(progress_path.read_text())["entries"])
    frozen = {
        "config_sha256": sha256(args.config), "panel_sha256": sha256(panel_path),
        "source_panel_sha256": sha256(args.source_panel),
        "g1_checkpoint_sha256": sha256(Path(config["sources"]["g1_checkpoint"])),
        "g1_filter_sha256": sha256(filter_path), "whitening_sha256": sha256(whitening_path),
        "flow_checkpoint_sha256": None if args.flow_checkpoint is None else sha256(args.flow_checkpoint),
        "method": args.method, "draws": draws, "selected_core_id": panel["selected_core_id"],
        "source_hashes": {
            "exporter": sha256(Path(__file__)),
            "fourier_modes": sha256(REPO_ROOT / "workflows/sbi/p12f3_fourier_modes.py"),
            "common_evaluator": sha256(REPO_ROOT / "workflows/sbi/p12f_common_evaluator.py"),
        },
    }
    digest = canonical_digest(frozen)
    if progress_path.exists() and json.loads(progress_path.read_text()).get("frozen_digest") != digest:
        raise RuntimeError("Fourier archive resume contract changed")
    complete = {int(row["core_id"]): row for row in entries}
    run_path = method_output / "run_manifest.json"
    if run_path.exists():
        if json.loads(run_path.read_text()).get("frozen_digest") != digest:
            raise RuntimeError("Fourier archive run contract changed")
    else:
        atomic_json(run_path, {
            "schema_version":"p12f3l2-fourier-export-run-v1", "created_utc":utc_now(),
            "git_revision_at_launch":git_revision(), "frozen_digest":digest, "frozen":frozen,
            "truth_files_read":["ph006"], "ph001_opened":False,
        })
    started = time.monotonic()
    try:
        for ordinal, core_id_value in enumerate(panel["selected_core_id"]):
            core_id = int(core_id_value)
            if core_id in complete:
                if sha256(Path(complete[core_id]["path"])) != complete[core_id]["sha256"]:
                    raise RuntimeError("completed Fourier core archive changed")
                continue
            patch = adapter.extract(core_id, halo, CHANNELS, alignment_voxels=alignment)
            condition, _ = model_inputs(patch, loader.field_normalization, args.device)
            mean, log_std = g1_model(condition)
            mean_scaled = mean[0, 0].cpu().numpy().astype(np.float32)
            std_scaled = torch.exp(log_std[0, 0]).cpu().numpy().astype(np.float32)
            seed = 43_000 + core_id
            unit = correlated_unit_residuals(filter_contract, draws=draws, seed=seed, shape=tuple(mean_scaled.shape))
            residual_g1 = std_scaled[None] * unit
            high = residual_g1 - lowpass_numpy(residual_g1, voxel_mpc_h=voxel, maximum_k=maximum_k)
            layout = build_fourier_layout(mean_scaled.shape, voxel_mpc_h=voxel, band_edges_h_mpc=edges)
            learned_low = sample_low(
                args.method, flow_model, condition, layout=layout, whitening=whitening,
                draws=draws, steps=int(config["model"]["ode_steps"]),
                seed=seed+100_000_000, batch=args.draw_batch,
            )
            scaled = mean_scaled[None] + high + learned_low
            samples = (scaled*np.float32(scaler["std"])+np.float32(scaler["mean"])).astype(np.float32)
            mean_physical = (mean_scaled*np.float32(scaler["std"])+np.float32(scaler["mean"])).astype(np.float32)
            target = store.extract(patch)
            counts = np.asarray(patch.values[patch.channel_names.index("counts")], dtype=np.float32)
            path = method_output / f"core_{core_id:08d}.npz"
            atomic_npz(
                path, delta_samples=samples, delta_truth=np.asarray(target["delta"],dtype=np.float32),
                conditional_mean=mean_physical, support=np.asarray(target["support"],dtype=np.uint8),
                angular_response=np.asarray(target["angular_response"],dtype=np.float32),
                boundary_distance_mpc=np.asarray(target["boundary_distance"],dtype=np.float32),
                tracer_density=counts/np.float32(voxel**3), core_bounds=core_bounds(patch),
                galaxy_frac_index_local=np.asarray(patch.authoritative_frac_index_local,dtype=np.float32),
            )
            row = {"core_id":core_id,"path":str(path.resolve()),"sha256":sha256(path),"seed":seed,
                   "shape":list(layout.shape),"modes":layout.modes,"components":layout.components}
            entries.append(row); complete[core_id]=row
            atomic_json(progress_path,{"schema_version":"p12f3l2-sample-progress-v1","frozen_digest":digest,"entries":entries,"ph001_opened":False})
            print(json.dumps({"method":args.method,"core":ordinal+1,"total":len(panel["selected_core_id"]),"elapsed_seconds":time.monotonic()-started}),flush=True)
            if time.monotonic()-started >= args.max_wall_seconds:
                raise SystemExit(75)
        ordered = [complete[int(value)] for value in panel["selected_core_id"]]
        scaler_path = method_output / "target_scaler.json"; atomic_json(scaler_path, scaler)
        archive = {
            "schema_version":"p12f-sample-archive-v1","created_utc":utc_now(),"method":args.method,
            "phase":"ph006","draws":draws,"panel_marker":str(panel_path.resolve()),"panel_sha256":sha256(panel_path),
            "checkpoint":str((Path(config["sources"]["g1_checkpoint"]) if args.flow_checkpoint is None else args.flow_checkpoint).resolve()),
            "checkpoint_sha256":frozen["g1_checkpoint_sha256"] if args.flow_checkpoint is None else frozen["flow_checkpoint_sha256"],
            "g1_checkpoint_sha256":frozen["g1_checkpoint_sha256"],"g1_filter_sha256":frozen["g1_filter_sha256"],
            "whitening_sha256":frozen["whitening_sha256"],
            "conditioning_contract_sha256":sha256(contract_root/"TRAINING_LOADER_READY.json"),
            "target_scaler":str(scaler_path.resolve()),"target_scaler_sha256":sha256(scaler_path),
            "entries":ordered,"truth_files_read":["ph006"],"ph001_opened":False,"pass":True,
        }
        atomic_json(archive_path,archive); print(json.dumps(archive,indent=2,sort_keys=True),flush=True)
    finally:
        store.close(); loader.close()


if __name__ == "__main__":
    main()
