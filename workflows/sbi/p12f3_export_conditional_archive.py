#!/usr/bin/env python3
"""Export ph006 samples for the conditional Gaussian/flow/diffusion rescue."""
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

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f3_conditional_models import (
    ALL_PATCH_CHANNELS,
    ConditionalLowModeGaussianUNet,
    proxy_condition,
    reconstruct_conditional_low,
    sample_conditional_gaussian_low,
    sample_fourier_ddim,
)
from workflows.sbi.p12f3_fourier_modes import (
    ConditionalFourierVelocityUNet,
    build_fourier_layout,
    sample_fourier_heun,
)
from workflows.sbi.p12f3_export_hybrid_archive import EvaluationTargetStore, atomic_npz, core_bounds, lowpass_numpy
from workflows.sbi.p12f3_train_conditional_gaussian import load_config, shuffle_seed
from workflows.sbi.p12f3_train_lowmode_flow import load_g1_model
from workflows.sbi.p12f_gaussian_controls import correlated_unit_residuals


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f3_conditional_calibration_v1.json"
DEFAULT_OUTPUT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_conditional_calibration_v1/evaluation")
METHODS = (
    "conditional_gaussian_base3", "conditional_gaussian_proxy7",
    "conditional_gaussian_proxy7_shuffled", "conditional_flow_proxy7",
    "conditional_diffusion_proxy7",
)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--method",choices=METHODS,required=True)
    parser.add_argument("--run-name",default="seed42_v1")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--draw-batch",type=int,default=4)
    parser.add_argument("--data-parallel",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--max-wall-seconds",type=float,default=6600)
    return parser.parse_args()


def utc_now(): return datetime.now(timezone.utc).isoformat()
def git_revision(): return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def method_parts(method: str) -> tuple[str, str]:
    if method.startswith("conditional_gaussian_"): return "gaussian", method.removeprefix("conditional_gaussian_")
    if method == "conditional_flow_proxy7": return "flow", "proxy7"
    if method == "conditional_diffusion_proxy7": return "diffusion", "proxy7"
    raise ValueError(method)


def load_location(output_root: Path, arm: str, run_name: str, config: dict, device: str):
    root=output_root.parent/"gaussian"/arm/run_name
    marker_path=root/"P12F3_CONDITIONAL_GAUSSIAN_TRAINED.json";marker=json.loads(marker_path.read_text());checkpoint=Path(marker["checkpoint"])
    filter_path=root/"conditional_residual_filter.json"
    if not marker.get("pass") or marker.get("ph001_opened") or marker.get("checkpoint_sha256")!=sha256(checkpoint): raise RuntimeError("unsafe location/scale model")
    filter_contract=json.loads(filter_path.read_text())
    if not filter_contract.get("pass") or filter_contract.get("validation_phase_used_for_fit") or filter_contract.get("ph001_opened"): raise RuntimeError("unsafe conditional covariance filter")
    model=ConditionalLowModeGaussianUNet(base=int(config["gaussian_control"]["unet_base"])).to(device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["model"]);model.eval().requires_grad_(False)
    return model,filter_contract,marker_path,checkpoint,filter_path


def load_generative(output_root: Path, method: str, run_name: str, config: dict, device: str):
    root=output_root.parent/"generative"/method/run_name
    marker_path=root/"P12F3_CONDITIONAL_GENERATIVE_TRAINED.json";marker=json.loads(marker_path.read_text());checkpoint=Path(marker["checkpoint"])
    whitening_path=output_root.parent/"generative"/"proxy7"/"conditional_whitening.json"
    whitening_marker=json.loads(whitening_path.read_text())
    if not marker.get("pass") or marker.get("method")!=method or marker.get("ph001_opened") or marker.get("checkpoint_sha256")!=sha256(checkpoint): raise RuntimeError("unsafe conditional generative model")
    if not whitening_marker.get("pass") or whitening_marker.get("validation_phase_used_for_fit") or whitening_marker.get("ph001_opened"): raise RuntimeError("unsafe conditional whitening")
    model=ConditionalFourierVelocityUNet(condition_channels=7,base=int(config["conditional_flow"]["unet_base"])).to(device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["model"]);model.eval().requires_grad_(False)
    return model,whitening_marker["whitening"],marker_path,checkpoint,whitening_path


@torch.inference_mode()
def sample_generative(kind,model,condition,layout,whitening,location,log_scale,draws,steps,seed,batch):
    generator=torch.Generator(device=condition.device).manual_seed(seed)
    pieces=[]
    for start in range(0,draws,batch):
        count=min(batch,draws-start)
        if kind=="flow": standard=sample_fourier_heun(model,condition,layout=layout,whitening=whitening,draws=count,steps=steps,generator=generator)
        else: standard=sample_fourier_ddim(model,condition,layout=layout,whitening=whitening,draws=count,steps=steps,generator=generator)
        pieces.append(reconstruct_conditional_low(standard,location,log_scale,layout).cpu().numpy().astype(np.float32))
    return np.concatenate(pieces)


def main():
    args=parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("conditional archive export requires CUDA")
    config,parent,parent_path=load_config(args.config);kind,arm=method_parts(args.method)
    if kind=="diffusion" and config["diffusion"]["run_only_if_licensed"]:
        license_path=args.output_root.parent/"DIFFUSION_LICENSE.json"
        if not license_path.is_file() or not json.loads(license_path.read_text()).get("licensed"): raise PermissionError("diffusion export is not licensed")
    location_model,filter_contract,location_marker,location_checkpoint,filter_path=load_location(args.output_root,arm,args.run_name,config,args.device)
    generative_model=whitening=generative_marker=generative_checkpoint=whitening_path=None
    if kind in ("flow","diffusion"):
        generative_model,whitening,generative_marker,generative_checkpoint,whitening_path=load_generative(args.output_root,kind,args.run_name,config,args.device)
        if args.data_parallel:
            if torch.cuda.device_count()<2: raise RuntimeError("data parallel export requires at least two GPUs")
            generative_model=torch.nn.DataParallel(generative_model)
    panel_path=Path(config["sources"]["source_panel"]);panel=json.loads(panel_path.read_text())
    if panel.get("phase")!="ph006" or panel.get("selection_uses_truth") or panel.get("ph001_opened") or len(panel.get("selected_core_id",[]))!=256: raise RuntimeError("unsafe ph006 conditional panel")
    output=args.output_root/args.method;output.mkdir(parents=True,exist_ok=True);archive_path=output/"P12F_SAMPLE_ARCHIVE.json"
    if archive_path.exists(): print(archive_path.read_text(),flush=True);return
    if any(output.iterdir()) and not args.resume: raise RuntimeError("non-empty conditional archive requires resume")
    loader=P10RandomResponseLoader(Path(config["sources"]["conditioning_contract"]),include_blind=False);store=EvaluationTargetStore(Path(config["sources"]["phase_root"]));adapter=loader.field_adapter("ph006")
    g1_model,scaler=load_g1_model(parent,args.device);g1_filter=json.loads(Path(config["sources"]["g1_filter"]).read_text())
    frozen={"config_sha256":sha256(args.config),"panel_sha256":sha256(panel_path),"method":args.method,"draws":int(config["evaluation"]["draws"]),"location_marker_sha256":sha256(location_marker),"location_checkpoint_sha256":sha256(location_checkpoint),"conditional_filter_sha256":sha256(filter_path),"generative_marker_sha256":None if generative_marker is None else sha256(generative_marker),"generative_checkpoint_sha256":None if generative_checkpoint is None else sha256(generative_checkpoint),"whitening_sha256":None if whitening_path is None else sha256(whitening_path),"source_hashes":{"exporter":sha256(Path(__file__)),"models":sha256(REPO_ROOT/"workflows/sbi/p12f3_conditional_models.py")},"ph001_opened":False}
    frozen_digest=digest(frozen);run_path=output/"run_manifest.json"
    if run_path.exists():
        if json.loads(run_path.read_text()).get("frozen_digest")!=frozen_digest: raise RuntimeError("conditional export resume contract changed")
    else: atomic_json(run_path,{"schema_version":"p12f3-conditional-export-run-v1","created_utc":utc_now(),"git_revision_at_launch":git_revision(),"frozen":frozen,"frozen_digest":frozen_digest,"truth_files_read":["ph006"],"ph001_opened":False})
    progress_path=output/"SAMPLE_ARCHIVE_PROGRESS.json";entries=[] if not progress_path.exists() else list(json.loads(progress_path.read_text())["entries"]);complete={int(row["core_id"]):row for row in entries};started=time.monotonic();draws=int(config["evaluation"]["draws"])
    try:
        for ordinal,core_value in enumerate(panel["selected_core_id"]):
            core_id=int(core_value)
            if core_id in complete:
                if sha256(Path(complete[core_id]["path"]))!=complete[core_id]["sha256"]: raise RuntimeError("conditional archive core changed")
                continue
            patch=adapter.extract(core_id,int(parent["patch"]["conditioning_halo_voxels"]),ALL_PATCH_CHANNELS,alignment_voxels=int(parent["patch"]["alignment_voxels"]))
            condition,g1_mean,g1_log_std=proxy_condition(patch,loader.field_normalization,g1_model,device=args.device,arm=arm,shuffle_seed=shuffle_seed(int(config["training"]["seed"]),"ph006",core_id) if arm=="proxy7_shuffled" else None)
            location,log_scale=location_model(condition)
            shape=tuple(g1_mean.shape[-3:]);layout=build_fourier_layout(shape,voxel_mpc_h=float(config["target"]["voxel_mpc_h"]),band_edges_h_mpc=tuple(config["target"]["band_edges_h_mpc"]));seed=43_000+core_id
            unit=correlated_unit_residuals(g1_filter,draws=draws,seed=seed,shape=shape);g1_residual=np.exp(g1_log_std[0,0].cpu().numpy())[None]*unit
            g1_low=lowpass_numpy(g1_residual,voxel_mpc_h=float(config["target"]["voxel_mpc_h"]),maximum_k=float(config["target"]["band_edges_h_mpc"][-1]));high=g1_residual-g1_low
            if kind=="gaussian": low=sample_conditional_gaussian_low(location,log_scale,layout,filter_contract,draws=draws,seed=seed+100_000_000)
            else: low=sample_generative(kind,generative_model,condition,layout,whitening,location,log_scale,draws,int(config["conditional_flow"]["ode_steps"] if kind=="flow" else config["diffusion"]["network_evaluations"]),seed+100_000_000,args.draw_batch)
            mean_scaled=g1_mean[0,0].cpu().numpy();scaled=mean_scaled[None]+high+low;samples=(scaled*np.float32(scaler["std"])+np.float32(scaler["mean"])).astype(np.float32)
            target=store.extract(patch);counts=np.asarray(patch.values[patch.channel_names.index("counts")],dtype=np.float32);path=output/f"core_{core_id:08d}.npz"
            atomic_npz(path,delta_samples=samples,delta_truth=np.asarray(target["delta"],dtype=np.float32),conditional_mean=(mean_scaled*np.float32(scaler["std"])+np.float32(scaler["mean"])).astype(np.float32),support=np.asarray(target["support"],dtype=np.uint8),angular_response=np.asarray(target["angular_response"],dtype=np.float32),boundary_distance_mpc=np.asarray(target["boundary_distance"],dtype=np.float32),tracer_density=counts/np.float32(float(config["target"]["voxel_mpc_h"])**3),core_bounds=core_bounds(patch),galaxy_frac_index_local=np.asarray(patch.authoritative_frac_index_local,dtype=np.float32))
            row={"core_id":core_id,"path":str(path.resolve()),"sha256":sha256(path),"seed":seed,"shape":list(shape),"modes":layout.modes,"components":layout.components};entries.append(row);complete[core_id]=row;atomic_json(progress_path,{"schema_version":"p12f3-conditional-sample-progress-v1","frozen_digest":frozen_digest,"entries":entries,"ph001_opened":False})
            print(json.dumps({"method":args.method,"core":ordinal+1,"total":256,"elapsed_seconds":time.monotonic()-started}),flush=True)
            if time.monotonic()-started>=args.max_wall_seconds: raise SystemExit(75)
        scaler_path=output/"target_scaler.json";atomic_json(scaler_path,scaler);ordered=[complete[int(v)] for v in panel["selected_core_id"]]
        archive={"schema_version":"p12f-sample-archive-v1","created_utc":utc_now(),"method":args.method,"phase":"ph006","draws":draws,"panel_marker":str(panel_path.resolve()),"panel_sha256":sha256(panel_path),"checkpoint":str((location_checkpoint if generative_checkpoint is None else generative_checkpoint).resolve()),"checkpoint_sha256":sha256(location_checkpoint if generative_checkpoint is None else generative_checkpoint),"conditioning_contract_sha256":sha256(Path(config["sources"]["conditioning_contract"])/"TRAINING_LOADER_READY.json"),"target_scaler":str(scaler_path.resolve()),"target_scaler_sha256":sha256(scaler_path),"entries":ordered,"truth_files_read":["ph006"],"ph001_opened":False,"pass":True}
        atomic_json(archive_path,archive);print(json.dumps(archive,indent=2),flush=True)
    finally: store.close();loader.close()


if __name__=="__main__": main()
