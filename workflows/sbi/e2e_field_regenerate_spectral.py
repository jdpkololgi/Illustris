#!/usr/bin/env python3
"""Separate spectral E2E parents, with matched cubic sampling and guarded roles.

Per-phase checkpoints are resumable only with identical source/config hashes.
Incomplete shards are never overwritten. Successful packaging is not R0 release.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
from scipy import fft

from workflows.sbi.e2e_field_build_products import (
    TENSOR_COMPONENTS, read_json, require_compute, sha256, validate_config, write_json, eigs,
)
from workflows.sbi.e2e_field_dataset import Moments, centred_crop, transformed, STANDARDIZED
from workflows.sbi.e2e_field_error_budget import native_coordinates, smoothed_spectrum, inverse_component
from workflows.sbi.e2e_field_prepare_data import REPO, safe_path

DEFAULT_CONFIG = REPO/"configs/e2e_field_spectral_v2.json"


def interpolate(field, coords, degree=3):
    """Separable local Lagrange interpolation; bounded planes, no global prefilter.

At each requested x, contract its native stencil before y and z contractions.
The result is the same tensor-product polynomial as the explicit corner sum.
"""
    if degree not in (1,3,5):
        raise ValueError("unsupported interpolation order")
    offsets = np.arange(-(degree//2),degree//2+2)
    base = [np.floor(c).astype(np.int64) for c in coords]
    weights = []
    for c,b in zip(coords,base):
        t = c-b
        weights.append(np.array([np.prod([(t-j)/(o-j) for j in offsets if j!=o],axis=0) for o in offsets]))
    n = field.shape[0]
    ix = [(b[None,:]+offsets[:,None]) % n for b in base]
    yz = np.ix_(ix[1].ravel(),ix[2].ravel())
    result = np.empty(tuple(len(c) for c in coords),dtype=np.float64)
    for x in range(len(coords[0])):
        plane = np.zeros((ix[1].size,ix[2].size),dtype=np.float64)
        for j in range(degree+1):
            plane += weights[0][j,x]*field[ix[0][j,x]][yz]
        plane = plane.reshape(degree+1,len(coords[1]),degree+1,len(coords[2]))
        result[x] = np.einsum("iyjz,iy,jz->yz",plane,weights[1],weights[2],optimize=True)
    return result


def source_hashes():
    names = [Path(__file__), *[Path(__file__).with_name(x) for x in
             ("e2e_field_build_products.py","e2e_field_dataset.py","e2e_field_error_budget.py","e2e_field_prepare_data.py")]]
    return {str(p.relative_to(REPO)):sha256(p) for p in names}


def inputs(path):
    c = read_json(path)
    bpath = REPO/c["build_config"]
    b = read_json(bpath)
    validate_config(b)
    if (c["schema_version"]!="e2e-spectral-products-v2" or c["phase_roles"]!=b["phase_roles"]
        or c["parent_sides"]!=[64,96] or c["smoothing_mpc_h"]!=7 or c["training_ready"]
        or c["r0_physics_pass"] or c["particle_redeposition"]):
        raise ValueError("unexpected spectral regeneration contract")
    root = Path(b["output_root"])
    index_path = root/"parent_arrays/DATASET_INDEX.json"
    index = read_json(index_path)
    if index["config_sha256"]!=sha256(bpath):
        raise ValueError("source dataset contract drift")
    native = read_json(root/"native_truth/NATIVE_TRUTH_COMPLETE.json")
    if native["dataset_index_sha256"]!=sha256(index_path):
        raise ValueError("source density provenance does not match dataset")
    out = safe_path(Path(c["output_root"])).resolve()
    allowed = Path("/pscratch/sd/d/dkololgi/abacus/e2e_field_v2").resolve()
    if not out.is_relative_to(allowed) or out==allowed:
        raise ValueError("outputs must be an E2E v2 child")
    return c,b,index,native,out,index_path


def phase_build(phase,c,b,index,native,out,registration):
    started = time.monotonic()
    receipt_path = out/f"{phase}_COMPLETE.json"
    if receipt_path.exists():
        r = read_json(receipt_path)
        if r["registration"]!=registration or any(sha256(x["path"])!=x["sha256"] for x in r["shards"]):
            raise ValueError("completed phase is not an exact resumable product")
        print(f"{phase}: verified completed phase, resume skips recomputation",flush=True)
        return r
    rows = [r for r in index["parents"] if r["phase"]==phase]
    if len(rows)!=32 or any(r["role"]!=c["phase_roles"][phase] for r in rows):
        raise ValueError("phase panel/role mismatch")
    source = next(p for p in native["phases"] if p["phase"]==phase)
    if sha256(source["density_path"])!=source["density_sha256"]:
        raise ValueError("source density hash mismatch")
    old_shards = {r["shard"] for r in rows}
    for old in old_shards:
        if sha256(old)!=next(s["sha256"] for s in index["shards"] if s["path"]==old):
            raise ValueError("observation/source parent hash mismatch")
    paths = {cap:out/f"{phase}_{cap}.h5" for cap in ("NGC","SGC")}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("partial phase preserved; choose a reviewed recovery path, never overwrite")
    print(f"{phase}: verified inputs; generating spectral Gaussian density",flush=True)
    spec,mean = smoothed_spectrum(source["density_path"],b["box_mpc_h"],c["smoothing_mpc_h"])
    controls = []
    prior_path = Path(b["output_root"]).parent/"error_budget_20260908"/f"{phase}_spectral_reference.h5"
    with ExitStack() as stack:
        old = {p:stack.enter_context(h5py.File(p,"r")) for p in old_shards}
        saved = {cap:stack.enter_context(h5py.File(p,"x")) for cap,p in paths.items()}
        prior = None
        if c["phase_roles"][phase]=="train":
            report = read_json(prior_path.with_name(f"{phase}_ERROR_BUDGET.json"))
            if sha256(prior_path)!=report["reference_sha256"]:
                raise ValueError("prior training reference drift")
            prior = stack.enter_context(h5py.File(prior_path,"r"))
        for cap,f in saved.items():
            f.attrs.update({"schema_version":c["schema_version"],"phase":phase,"cap":cap,
                "role":c["phase_roles"][phase],"training_ready":False,"target_semantics":"Gaussian R7 density, spectral tensor, cubic sampling",
                "config_sha256":registration["config_sha256"],"density_source_sha256":source["density_sha256"]})
        for row in rows:
            src = old[row["shard"]][row["group"]]
            g = saved[row["cap"]].create_group(row["group"])
            for key,value in src.attrs.items():
                g.attrs[key] = value
            src.file.copy(src["condition_raw"],g,name="condition_raw")
            src.file.copy(src["masks"],g,name="masks")
            g.create_dataset("delta_r7_gaussian",shape=(96,)*3,dtype="f4",chunks=(16,)*3,compression="lzf")
            g.create_dataset("tensor_spectral",shape=(96,96,96,6),dtype="f4",chunks=(16,16,16,1),compression="lzf")
        work = spec.copy()
        field = fft.irfftn(work,s=(b["native_ngrid"],)*3,workers=32,overwrite_x=True)
        del work
        for row in rows:
            x = interpolate(field,native_coordinates(row,96,b))
            saved[row["cap"]][row["group"]]["delta_r7_gaussian"][:] = x.astype(np.float32)
        del field
        gc.collect()
        print(f"{phase}: cubic-sampled Gaussian density complete",flush=True)
        for column,component in enumerate(TENSOR_COMPONENTS):
            field = inverse_component(spec,b["box_mpc_h"],component)
            for row in rows:
                x = interpolate(field,native_coordinates(row,96,b))
                saved[row["cap"]][row["group"]]["tensor_spectral"][...,column] = x.astype(np.float32)
                if prior is not None:
                    core = centred_crop(x,32)
                    reference = prior[row["anchor_id"]]["tensor_local_cubic_lagrange"][...,column]
                    parity = float(np.max(np.abs(core-reference)))
                    quintic = interpolate(field,native_coordinates(row,32,b),degree=5)
                    scale = max(float(np.std(quintic)),1e-12)
                    difference = float(np.sqrt(np.mean((core-quintic)**2))/scale)
                    if parity>c["technical_prior_cubic_reference_max_abs"] or difference>c["technical_cubic_quintic_relative_l2"]:
                        raise ValueError(f"interpolation parity/convergence failed {row['anchor_id']}: {parity}, {difference}")
                    controls.append({"anchor_id":row["anchor_id"],"component":column,
                                     "prior_cubic_max_abs":parity,"cubic_quintic_rmse_over_scatter":difference})
            del field
            gc.collect()
            print(f"{phase}: spectral component {component} complete",flush=True)
        del spec
        gc.collect()
        checks = []
        for row in rows:
            g = saved[row["cap"]][row["group"]]
            tensor = g["tensor_spectral"][:]
            delta = g["delta_r7_gaussian"][:]
            eigen = eigs(tensor)
            trace_error = float(np.max(np.abs(tensor[...,[0,3,5]].sum(axis=-1)-delta)))
            if not np.isfinite(tensor).all() or not np.isfinite(delta).all() or trace_error>c["technical_trace_max_abs"]:
                raise ValueError("nonfinite product or tensor/density trace mismatch")
            g.create_dataset("eigenvalues_spectral",data=eigen.astype(np.float32),chunks=(16,16,16,3),compression="lzf")
            src = old[row["shard"]][row["group"]]
            for group in ("condition_raw","masks"):
                for name in src[group]:
                    if not np.array_equal(src[group][name][:],g[group][name][:]):
                        raise ValueError("observation/mask copy changed")
            checks.append({"anchor_id":row["anchor_id"],"trace_max_abs":trace_error,"condition_and_masks_exact":True})
    shards = [{"path":str(p),"sha256":sha256(p),"bytes":p.stat().st_size} for p in paths.values()]
    receipt = {"phase":phase,"role":c["phase_roles"][phase],"registration":registration,
               "density_source":source["density_path"],"density_sha256":source["density_sha256"],"density_mean":mean,
               "shards":shards,"parents":32,"checks":checks,"interpolation_controls":controls,
               "science_summaries_evaluated":False,"technical_pass":True,"training_ready":False,
               "job_id":os.environ.get("SLURM_JOB_ID"),"elapsed_seconds":time.monotonic()-started}
    write_json(receipt_path,receipt)
    print(f"{phase}: COMPLETE {receipt['elapsed_seconds']:.1f}s",flush=True)
    return receipt


class SpectralParentDataset:
    def __init__(self,index_path,role,*,allow_unreleased=False,allow_confirmation=False):
        self.index = read_json(index_path)
        if self.index["schema_version"]!="e2e-spectral-products-v2":
            raise ValueError("not a spectral v2 dataset")
        if not self.index["training_ready"] and not allow_unreleased:
            raise PermissionError("science-training release is closed")
        if role=="internal_confirmation" and not allow_confirmation:
            raise PermissionError("confirmation access needs separate authority")
        if role not in set(self.index["phase_roles"].values()):
            raise ValueError("unknown role")
        self.parents = [p for p in self.index["parents"] if p["role"]==role]
        if sha256(self.index["normalization"])!=self.index["normalization_sha256"]:
            raise ValueError("normalization drift")
        self.normalization = read_json(self.index["normalization"])

    def parent(self,position,side=96,normalize=True):
        if side not in (64,96):
            raise ValueError("unsupported geometry")
        row = self.parents[position]
        with h5py.File(safe_path(Path(row["shard"])),"r") as f:
            g = f[row["group"]]
            delta = centred_crop(g["delta_r7_gaussian"][:],side)
            channels = []
            for name in self.index["condition_channels"]:
                x = centred_crop(g["condition_raw"][name][:],side+32)
                if normalize:
                    x = transformed(name,x)
                    if name in self.normalization["channels"]:
                        stat = self.normalization["channels"][name]
                        x = (x-stat["mean"])/stat["std"]
                channels.append(x)
            masks = {name:centred_crop(ds[:],side).astype(bool) for name,ds in g["masks"].items()}
        if normalize:
            stat = self.normalization["target"]
            delta = (delta-stat["mean"])/stat["std"]
        return {"anchor_id":row["anchor_id"],"delta":np.asarray(delta,dtype=np.float32),
                "condition":np.stack(channels),"masks":masks,"children":row["children_by_side"][str(side)]}


def finalize(c,index,out,receipts,registration):
    rows = [{**row,"shard":str(out/f"{row['phase']}_{row['cap']}.h5")} for row in index["parents"]]
    moments = Moments()
    channel_moments = {name:Moments() for name in STANDARDIZED}
    for row in rows:
        if row["role"]!="train":
            continue
        with h5py.File(row["shard"],"r") as f:
            g = f[row["group"]]
            moments.add(g["delta_r7_gaussian"][:])
            for name,moment in channel_moments.items():
                moment.add(transformed(name,g["condition_raw"][name][:]))
    original = read_json(index["normalization"])
    if sha256(index["normalization"])!=index["normalization_sha256"]:
        raise ValueError("original scaler changed")
    channels = {name:value.report() for name,value in channel_moments.items()}
    if channels!=original["channels"]:
        raise ValueError("unchanged observations do not reproduce training-only normalization")
    norm = {**original,"target":moments.report(),"channels":channels,"target_semantics":c["target_dataset"],
            "config_sha256":registration["config_sha256"]}
    write_json(out/"TRAIN_NORMALIZATION.json",norm)
    result = {**index,"schema_version":c["schema_version"],"config_sha256":registration["config_sha256"],
              "builder_sha256":sha256(__file__),
              "registration":registration,"parents":rows,"target_dataset":c["target_dataset"],
              "normalization":str(out/"TRAIN_NORMALIZATION.json"),"normalization_sha256":sha256(out/"TRAIN_NORMALIZATION.json"),
              "shards":[s for r in receipts for s in r["shards"]],"training_ready":False,
              "r0_physics_pass":False,"regeneration_complete":True,"source_index_semantics":"v1 geometry/observations only; target replaced",
              "remaining_gates":["R0-PHYSICS cubic target domain and science-functional tolerances",
                                 "matched model/transform and diagnostic-power contracts","explicit fit authorization"]}
    result["inherited_geometry_native_reference_sha256"] = result.pop("native_reference_sha256",None)
    write_json(out/"DATASET_INDEX.json",result)
    for role in ("train","internal_confirmation"):
        try:
            SpectralParentDataset(out/"DATASET_INDEX.json",role,allow_unreleased=role=="internal_confirmation")
        except PermissionError:
            pass
        else:
            raise ValueError("dataset release guard failed")
    dataset = SpectralParentDataset(out/"DATASET_INDEX.json","train",allow_unreleased=True)
    wide,narrow = dataset.parent(0,96),dataset.parent(0,64)
    if not np.isfinite(wide["condition"]).all() or not np.isfinite(wide["delta"]).all():
        raise ValueError("nonfinite normalized products")
    if not np.array_equal(centred_crop(wide["delta"],64),narrow["delta"]):
        raise ValueError("nested normalized targets differ")
    for a,b in zip(wide["condition"],narrow["condition"]):
        if not np.array_equal(centred_crop(a,96),b):
            raise ValueError("nested normalized observations differ")
    for shard in result["shards"]:
        if sha256(shard["path"])!=shard["sha256"]:
            raise ValueError("final shard hash drift")
    write_json(out/"REGENERATION_COMPLETE.json",{"schema_version":c["schema_version"],"registration":registration,
         "dataset_index_sha256":sha256(out/"DATASET_INDEX.json"),"phases":[r["phase"] for r in receipts],
         "parents":len(rows),"geometry_views":2*len(rows),"bytes":sum(s["bytes"] for s in result["shards"]),
         "technical_pass":True,"training_ready":False,"r0_physics_pass":False,"job_id":os.environ.get("SLURM_JOB_ID"),
         "all_shard_hashes_verified":True,"readiness_and_confirmation_guards_pass":True,"nested_reader_smoke_pass":True})
    print("REGENERATION COMPLETE; technical pass, science-training gate remains closed",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args = p.parse_args()
    require_compute()
    c,b,index,native,out,index_path = inputs(args.config)
    out.mkdir(parents=True,exist_ok=True)
    registration = {"config_sha256":sha256(args.config),"source_index_sha256":sha256(index_path),"source_sha256":source_hashes()}
    manifest = out/"REGENERATION_STARTED.json"
    if manifest.exists():
        if read_json(manifest)["registration"]!=registration:
            raise ValueError("resume source or config mismatch")
    else:
        write_json(manifest,{"registration":registration,"config":c,"job_id":os.environ["SLURM_JOB_ID"],
            "base_git":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),"training_ready":False})
    if (out/"REGENERATION_COMPLETE.json").exists():
        raise FileExistsError("regeneration already complete; do not rerun")
    receipts = [phase_build(phase,c,b,index,native,out,registration) for phase in c["phase_roles"]]
    if source_hashes()!=registration["source_sha256"]:
        raise ValueError("source changed during regeneration")
    finalize(c,index,out,receipts,registration)


if __name__ == "__main__":
    main()
