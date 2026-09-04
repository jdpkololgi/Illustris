#!/usr/bin/env python3
"""Fit conditional pre-whitening and train matched Fourier flow/diffusion arms."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f3_conditional_models import (
    ALL_PATCH_CHANNELS,
    ConditionalLowModeGaussianUNet,
    fourier_v_pair,
    low_mode_target,
    proxy_condition,
    sample_fourier_ddim,
    standardized_low_field,
)
from workflows.sbi.p12f3_fourier_modes import (
    ConditionalFourierVelocityUNet,
    build_fourier_layout,
    empty_whitening_accumulator,
    equal_band_flow_loss,
    finalize_whitening,
    pack_fourier_components,
    rectified_flow_pair,
    sample_fourier_heun,
    update_whitening_accumulator,
    whiten_components,
)
from workflows.sbi.p12f3_train_conditional_gaussian import (
    digest,
    git_revision,
    load_config,
    restore_rng_state,
    shuffle_seed,
    split_selected,
    utc_now,
)
from workflows.sbi.p12f3_train_fourier_lowmode_flow import _open_common
from workflows.sbi.p12f3_train_lowmode_flow import epoch_references, load_g1_model, target_tensor


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f3_conditional_calibration_v1.json"
DEFAULT_OUTPUT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_conditional_calibration_v1")
METHODS = ("flow", "diffusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("fit-whitening", "train"), required=True)
    parser.add_argument("--method", choices=METHODS, default="flow")
    parser.add_argument("--gaussian-arm", default="proxy7")
    parser.add_argument("--gaussian-run", default="seed42_v1")
    parser.add_argument("--run-name", default="seed42_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--max-wall-seconds", type=float, default=6600.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_location_scale(args: argparse.Namespace, config: dict, device: str):
    root = args.output_root / "gaussian" / args.gaussian_arm / args.gaussian_run
    marker_path = root / "P12F3_CONDITIONAL_GAUSSIAN_TRAINED.json"
    marker = json.loads(marker_path.read_text())
    checkpoint_path = Path(marker["checkpoint"])
    if (
        marker.get("schema_version") != "p12f3-conditional-gaussian-trained-v1"
        or not marker.get("pass") or marker.get("ph001_opened")
        or marker.get("checkpoint_sha256") != sha256(checkpoint_path)
    ):
        raise RuntimeError("unsafe conditional location/scale parent")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ConditionalLowModeGaussianUNet(base=int(config["gaussian_control"]["unet_base"])).to(device)
    model.load_state_dict(state["model"], strict=True); model.eval().requires_grad_(False)
    return model, marker, marker_path, checkpoint_path


def build_example(*, loader, store, g1_model, location_model, scaler, phase, core_id, config, parent, arm, device, whitening=None):
    patch = loader.field_adapter(phase).extract(
        core_id, int(parent["patch"]["conditioning_halo_voxels"]), ALL_PATCH_CHANNELS,
        alignment_voxels=int(parent["patch"]["alignment_voxels"]),
    )
    condition, g1_mean, _ = proxy_condition(
        patch, loader.field_normalization, g1_model, device=device, arm=arm,
        shuffle_seed=shuffle_seed(int(config["training"]["seed"]), phase, core_id) if arm == "proxy7_shuffled" else None,
    )
    target_data = store.extract(phase, patch)
    target = target_tensor(target_data["delta"], scaler, device)
    layout = build_fourier_layout(
        tuple(target.shape[-3:]), voxel_mpc_h=float(config["target"]["voxel_mpc_h"]),
        band_edges_h_mpc=tuple(float(value) for value in config["target"]["band_edges_h_mpc"]),
    )
    target_low = low_mode_target(target - g1_mean, layout)
    with torch.inference_mode():
        location, log_scale = location_model(condition)
        standard_field = standardized_low_field(target_low, location, log_scale, layout)
        vector = pack_fourier_components(standard_field, layout)
        if whitening is not None:
            vector = whiten_components(vector, whitening, layout)
    return condition, vector, layout, location, log_scale, patch


def frozen_contract(config, config_path, parent_path, selected, training, internal, phases, args, gaussian_marker_path, gaussian_checkpoint):
    return {
        "config": str(config_path.resolve()), "config_sha256": sha256(config_path),
        "parent_config": str(parent_path.resolve()), "parent_config_sha256": sha256(parent_path),
        "source_hashes": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(REPO_ROOT / "workflows/sbi/p12f3_conditional_models.py"): sha256(REPO_ROOT / "workflows/sbi/p12f3_conditional_models.py"),
        },
        "selected_core_ids": selected, "training_core_ids": training,
        "internal_validation_core_ids": internal, "training_phases": list(phases),
        "gaussian_arm": args.gaussian_arm,
        "gaussian_marker": str(gaussian_marker_path.resolve()), "gaussian_marker_sha256": sha256(gaussian_marker_path),
        "gaussian_checkpoint": str(gaussian_checkpoint.resolve()), "gaussian_checkpoint_sha256": sha256(gaussian_checkpoint),
        "method": args.method, "ph001_opened": False,
    }


def fit_whitening(args, config, parent, parent_path):
    _, _, phases, _, _, loader, store, selected = _open_common(parent)
    training, internal = split_selected(selected, phases, float(config["training"]["internal_validation_fraction_per_phase"]), int(config["training"]["seed"]))
    location_model, _, marker_path, checkpoint_path = load_location_scale(args, config, args.device)
    output = args.output_root / "generative" / args.gaussian_arm
    output.mkdir(parents=True, exist_ok=True)
    whitening_path = output / "conditional_whitening.json"
    if whitening_path.exists():
        print(whitening_path.read_text(), flush=True); store.close(); loader.close(); return
    frozen = frozen_contract(config, args.config, parent_path, selected, training, internal, phases, args, marker_path, checkpoint_path)
    frozen["method"] = "shared_flow_diffusion_target"
    g1_model, scaler = load_g1_model(parent, args.device)
    accumulator = empty_whitening_accumulator(4)
    refs = [(phase, int(core)) for phase in phases for core in training[phase]]
    maximum_roundtrip = 0.0
    with torch.inference_mode():
        for ordinal, (phase, core_id) in enumerate(refs):
            _, vector, layout, _, _, _ = build_example(
                loader=loader, store=store, g1_model=g1_model, location_model=location_model,
                scaler=scaler, phase=phase, core_id=core_id, config=config, parent=parent,
                arm=args.gaussian_arm, device=args.device, whitening=None,
            )
            update_whitening_accumulator(accumulator, vector, layout)
            if ordinal == 0 or (ordinal + 1) % 100 == 0 or ordinal + 1 == len(refs):
                print(json.dumps({"stage":"conditional-whitening","core":ordinal+1,"total":len(refs)}), flush=True)
    whitening = finalize_whitening(accumulator)
    passed = bool(min(whitening["count"]) >= 1000 and min(whitening["std"]) > 0 and np.all(np.isfinite(whitening["mean"] + whitening["std"])))
    payload = {
        "schema_version":"p12f3-conditional-whitening-v1","created_utc":utc_now(),"pass":passed,
        "fit_phases":list(phases),"training_core_ids":training,"internal_validation_core_ids":internal,
        "whitening":whitening,"frozen":frozen,"frozen_digest":digest(frozen),
        "truth_files_read":[f"{phase} training delta_R7" for phase in phases],
        "validation_phase_used_for_fit":False,"ph001_opened":False,
    }
    atomic_json(whitening_path, payload)
    store.close(); loader.close()
    if not passed: raise RuntimeError("conditional whitening gate failed")
    print(json.dumps(payload, indent=2), flush=True)


def checkpoint_payload(model, optimizer, update, frozen_digest, loss_sum, loss_count, method):
    return {
        "schema_version":"p12f3-conditional-generative-checkpoint-v1","method":method,
        "model":model.state_dict(),"optimizer":optimizer.state_dict(),"update":int(update),
        "frozen_digest":frozen_digest,"loss_sum":float(loss_sum),"loss_count":int(loss_count),
        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "ph001_opened":False,
    }


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


@torch.inference_mode()
def internal_loss(model, method, refs, *, loader, store, g1_model, location_model, scaler, config, parent, arm, device, whitening):
    model.eval(); values = []
    for phase, core_ids in refs.items():
        for core_id in core_ids:
            condition, target, layout, _, _, _ = build_example(
                loader=loader, store=store, g1_model=g1_model, location_model=location_model,
                scaler=scaler, phase=phase, core_id=core_id, config=config, parent=parent,
                arm=arm, device=device, whitening=whitening,
            )
            generator = torch.Generator(device=device).manual_seed(99_000 + core_id + 1000 * int(phase[2:]))
            if method == "flow": state, time_value, desired = rectified_flow_pair(target, generator=generator)
            else: state, time_value, desired = fourier_v_pair(target, generator=generator)
            predicted = model(state, time_value, condition, layout=layout, whitening=whitening)
            values.append(float(equal_band_flow_loss(predicted, desired, layout, 2).cpu()))
    return {"mean_loss":float(np.mean(values)),"cores":len(values),"standard_error":float(np.std(values,ddof=1)/np.sqrt(len(values)))}


def train(args, config, parent, parent_path):
    _, _, phases, _, _, loader, store, selected = _open_common(parent)
    training, internal = split_selected(selected, phases, float(config["training"]["internal_validation_fraction_per_phase"]), int(config["training"]["seed"]))
    location_model, _, marker_path, location_checkpoint = load_location_scale(args, config, args.device)
    whitening_path = args.output_root / "generative" / args.gaussian_arm / "conditional_whitening.json"
    whitening_marker = json.loads(whitening_path.read_text())
    if not whitening_marker.get("pass") or whitening_marker.get("validation_phase_used_for_fit") or whitening_marker.get("ph001_opened"):
        raise RuntimeError("unsafe conditional whitening parent")
    whitening = whitening_marker["whitening"]
    output = args.output_root / "generative" / args.method / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"; manifest_path = output / "run_manifest.json"
    canary_path = output / "TECHNICAL_CANARY_COMPLETE.json"; terminal_path = output / "P12F3_CONDITIONAL_GENERATIVE_TRAINED.json"
    if terminal_path.exists(): print(terminal_path.read_text(), flush=True); store.close(); loader.close(); return
    frozen = frozen_contract(config, args.config, parent_path, selected, training, internal, phases, args, marker_path, location_checkpoint)
    frozen.update({"whitening":str(whitening_path.resolve()),"whitening_sha256":sha256(whitening_path)})
    frozen_digest = digest(frozen)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()).get("frozen_digest") != frozen_digest: raise RuntimeError("generative resume contract changed")
    elif any(output.iterdir()): raise RuntimeError("non-empty generative output has no valid manifest")
    else: atomic_json(manifest_path,{"schema_version":"p12f3-conditional-generative-run-v1","created_utc":utc_now(),"git_revision_at_launch":git_revision(),"frozen":frozen,"frozen_digest":frozen_digest,"ph001_opened":False})
    torch.manual_seed(int(config["training"]["seed"])); np.random.seed(int(config["training"]["seed"]))
    g1_model, scaler = load_g1_model(parent, args.device)
    model = ConditionalFourierVelocityUNet(condition_channels=7, base=int(config["conditional_flow"]["unet_base"])).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(config["training"]["learning_rate"]),weight_decay=float(config["training"]["weight_decay"]))
    update=0; loss_sum=0.0; loss_count=0
    if checkpoint_path.exists():
        state=torch.load(checkpoint_path,map_location=args.device,weights_only=False)
        if state.get("frozen_digest")!=frozen_digest or state.get("method")!=args.method or state.get("ph001_opened"): raise RuntimeError("unsafe generative checkpoint")
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);update=int(state["update"]);loss_sum=float(state["loss_sum"]);loss_count=int(state["loss_count"])
        restore_rng_state(state)
    else: atomic_checkpoint(checkpoint_path,checkpoint_payload(model,optimizer,0,frozen_digest,0,0,args.method))
    total=int(config["training"]["science_updates"]);stop=total if args.stop_after_updates is None else int(args.stop_after_updates)
    if stop<=update or stop>total: raise ValueError("invalid conditional generative stop update")
    refs_per_epoch=sum(len(training[p]) for p in phases);trace_path=output/"loss_trace.jsonl";started=time.monotonic();last_gradient=float("nan")
    try:
        while update<stop:
            epoch=update//refs_per_epoch;ordinal=update%refs_per_epoch
            phase,core_id=epoch_references(training,phases,seed=int(config["training"]["seed"]),epoch=epoch)[ordinal]
            condition,target,layout,_,_,_=build_example(loader=loader,store=store,g1_model=g1_model,location_model=location_model,scaler=scaler,phase=phase,core_id=core_id,config=config,parent=parent,arm=args.gaussian_arm,device=args.device,whitening=whitening)
            generator=None
            if args.method=="flow": state,time_value,desired=rectified_flow_pair(target,generator=generator)
            else: state,time_value,desired=fourier_v_pair(target,generator=generator)
            model.train();predicted=model(state,time_value,condition,layout=layout,whitening=whitening)
            loss=equal_band_flow_loss(predicted,desired,layout,2)
            optimizer.zero_grad(set_to_none=True);loss.backward();last_gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["training"]["gradient_clip"])).detach().cpu());optimizer.step()
            if not torch.isfinite(loss) or not np.isfinite(last_gradient) or not all(torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError("non-finite conditional generative state")
            update+=1;value=float(loss.detach().cpu());loss_sum+=value;loss_count+=1
            if update%int(config["training"]["loss_log_every_updates"])==0 or update==stop:
                with trace_path.open("a") as stream: stream.write(json.dumps({"update":update,"epoch_equivalent":update/refs_per_epoch,"loss":value,"mean_loss":loss_sum/loss_count,"preclip_gradient_norm":last_gradient,"phase":phase,"core_id":core_id,"elapsed_seconds":time.monotonic()-started},sort_keys=True)+"\n")
            if update%int(config["training"]["checkpoint_every_updates"])==0 or update==stop: atomic_checkpoint(checkpoint_path,checkpoint_payload(model,optimizer,update,frozen_digest,loss_sum,loss_count,args.method))
            if time.monotonic()-started>=args.max_wall_seconds and update<stop:
                atomic_json(output/"PAUSED.json",{"schema_version":"p12f3-conditional-generative-pause-v1","method":args.method,"update":update,"frozen_digest":frozen_digest,"ph001_opened":False});raise SystemExit(75)
        probe_phase=phases[0];probe_core=training[probe_phase][0]
        condition,_,layout,_,_,_=build_example(loader=loader,store=store,g1_model=g1_model,location_model=location_model,scaler=scaler,phase=probe_phase,core_id=probe_core,config=config,parent=parent,arm=args.gaussian_arm,device=args.device,whitening=whitening)
        generator=torch.Generator(device=args.device).manual_seed(73001)
        model.eval()
        if args.method=="flow": probe=sample_fourier_heun(model,condition,layout=layout,whitening=whitening,draws=4,steps=int(config["conditional_flow"]["ode_steps"]),generator=generator)
        else: probe=sample_fourier_ddim(model,condition,layout=layout,whitening=whitening,draws=4,steps=int(config["diffusion"]["network_evaluations"]),generator=generator)
        passed=bool(torch.isfinite(probe).all() and float(probe.std())>1e-6)
        if stop==int(config["training"]["technical_canary_updates"]):
            marker={"schema_version":"p12f3-conditional-generative-canary-v1","pass":passed,"method":args.method,"update":update,"checkpoint":str(checkpoint_path.resolve()),"checkpoint_sha256":sha256(checkpoint_path),"frozen_digest":frozen_digest,"ph001_opened":False}
            atomic_json(canary_path,marker);print(json.dumps(marker,indent=2),flush=True);return
        validation=internal_loss(model,args.method,internal,loader=loader,store=store,g1_model=g1_model,location_model=location_model,scaler=scaler,config=config,parent=parent,arm=args.gaussian_arm,device=args.device,whitening=whitening)
        marker={"schema_version":"p12f3-conditional-generative-trained-v1","created_utc":utc_now(),"pass":passed,"method":args.method,"updates":update,"mean_training_loss":loss_sum/max(loss_count,1),"internal_validation":validation,"checkpoint":str(checkpoint_path.resolve()),"checkpoint_sha256":sha256(checkpoint_path),"frozen_digest":frozen_digest,"ph006_used_for_fit":False,"ph001_opened":False}
        atomic_json(terminal_path,marker);print(json.dumps(marker,indent=2),flush=True)
        if not passed: raise RuntimeError("conditional generative technical gate failed")
    finally: store.close();loader.close()


def main():
    args=parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("conditional generative work requires CUDA")
    config,parent,parent_path=load_config(args.config)
    if args.stage=="fit-whitening": fit_whitening(args,config,parent,parent_path)
    else: train(args,config,parent,parent_path)


if __name__=="__main__": main()
