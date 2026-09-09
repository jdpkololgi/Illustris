#!/usr/bin/env python3
"""Training-only finite-parent screen on regenerated spectral products.

Numerical completion and a deterministic screen are distinct from R0-PHYSICS.
No new smoothing, learned fits, holdout payloads or deployable oracle correction.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import time

import h5py
import numpy as np

from workflows.sbi.e2e_field_build_products import (
    context_slice, eigs, read_json, require_compute, sha256, tensor_from_delta,
    validate_config, write_json,
)
from workflows.sbi.e2e_field_dataset import centred_crop
from workflows.sbi.e2e_field_error_budget import science, tensor_metrics
from workflows.sbi.e2e_field_prepare_data import REPO, safe_path

DEFAULT_CONFIG = REPO / "configs/e2e_field_domain_gate_v2.json"


def gate_values(metrics, candidate, reference):
    if any(p["value"] is None for s in (candidate, reference) for p in s["pair"]):
        raise ValueError("undefined primary pair statistic; cannot qualify")
    return {
        "eigen_rmse_max_abs": float(max(metrics["eigen_rmse"])),
        "eigen_bias_max_abs": float(max(abs(x) for x in metrics["eigen_bias"])),
        "eigen_rmse_max_over_truth_std": float(max(metrics["eigen_rmse_over_truth_std"])),
        "four_class_disagreement_max": metrics["four_class_disagreement"],
        "filling_fraction_max_abs_change": abs(candidate["filling_fraction"]-reference["filling_fraction"]),
        "pair_probability_max_abs_change": max(abs(a["value"]-b["value"]) for a,b in zip(candidate["pair"],reference["pair"])),
        "largest_void_fraction_max_abs_change": abs(candidate["largest_void_fraction"]-reference["largest_void_fraction"]),
        "changed_connection_axes_max": sum(a!=b for a,b in zip(candidate["connections_xyz"],reference["connections_xyz"])),
    }


def failures(values, tolerances, multiplier=1.):
    if set(values)!=set(tolerances) or not np.isfinite(list(values.values())).all():
        raise ValueError("missing or nonfinite gate values")
    return [name for name,limit in tolerances.items() if values[name]>limit*multiplier]


def traceless_constant(residual, mask):
    mean = residual[mask].mean(axis=0)
    mean[[0,3,5]] -= mean[[0,3,5]].sum()/3
    return mean


def describe(rows, methods, cfg):
    result = {}
    for method in methods:
        result[method] = {}
        for mask in cfg["masks"]:
            records = [r["masks"][mask][method] for r in rows]
            result[method][mask] = {
                "parents": len(rows),
                "failed_parents": sum(bool(r["failed_gates"]) for r in records),
                "gate_failure_counts": {k:sum(k in r["failed_gates"] for r in records) for k in cfg["tolerances"]},
                "max_gate_values": {k:max(r["gate_values"][k] for r in records) for k in cfg["tolerances"]},
                "median_eigen_rmse_over_truth_std": np.median([r["metrics"]["eigen_rmse_over_truth_std"] for r in records],axis=0).tolist(),
                "median_eigen_rmse": np.median([r["metrics"]["eigen_rmse"] for r in records],axis=0).tolist(),
                "median_constant_residual_energy_fraction": float(np.median([r["metrics"]["constant_tensor_residual_energy_fraction"] for r in records])),
                "failed_anchor_ids": [r["anchor_id"] for r,rec in zip(rows,records) if rec["failed_gates"]],
            }
    return result


def validate_inputs(config_path):
    cfg = read_json(config_path)
    if (cfg["schema_version"]!="e2e-domain-gate-v2" or cfg["phases"]!=["ph000","ph002","ph003"]
        or cfg["parent_sides"]!=[64,96] or cfg["science_side"]!=32 or cfg["threshold"]!=.2
        or cfg["training_ready"] or cfg["r0_physics_pass"] or cfg["posterior_uncertainty_available"]):
        raise ValueError("unregistered phase, geometry or release")
    build = read_json(REPO/cfg["build_config"])
    validate_config(build)
    root = safe_path(Path(cfg["dataset_root"]))
    done = read_json(root/"REGENERATION_COMPLETE.json")
    index = read_json(root/"DATASET_INDEX.json")
    if not done["technical_pass"] or sha256(root/"DATASET_INDEX.json")!=done["dataset_index_sha256"]:
        raise ValueError("spectral products are not verified complete")
    if index["schema_version"]!="e2e-spectral-products-v2" or index["phase_roles"]!=build["phase_roles"]:
        raise ValueError("target or phase-role mismatch")
    rows = [r for r in index["parents"] if r["phase"] in cfg["phases"]]
    if len(rows)!=96 or len({r["anchor_id"] for r in rows})!=96 or any(r["role"]!="train" for r in rows):
        raise ValueError("only the 96 unique registered training anchors are permitted")
    if any(sum(r["phase"]==p for r in rows)!=32 for p in cfg["phases"]):
        raise ValueError("phase count mismatch")
    out = safe_path(Path(cfg["output_root"])).resolve()
    allowed = Path("/pscratch/sd/d/dkololgi/abacus/e2e_field_v2").resolve()
    if not out.is_relative_to(allowed) or out==allowed or out==root.resolve():
        raise ValueError("audit must use a separate child output")
    return cfg,build,index,done,rows,out


def run(config_path):
    require_compute()
    cfg,build,index,done,rows,out = validate_inputs(config_path)
    out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    source_paths = [Path(__file__),*[Path(__file__).with_name(n) for n in (
        "e2e_field_build_products.py","e2e_field_dataset.py","e2e_field_error_budget.py","e2e_field_prepare_data.py")]]
    registration = {"config_sha256":sha256(config_path),"dataset_index_sha256":done["dataset_index_sha256"],
                    "source_sha256":{str(p.relative_to(REPO)):sha256(p) for p in source_paths}}
    write_json(out/"DOMAIN_GATE_STARTED.json",{"registration":registration,"config":cfg,
        "job_id":os.environ["SLURM_JOB_ID"],"node":socket.gethostname(),
        "base_git":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()})
    # Hash only training shards. Never open held-out payloads even for checksums.
    paths = sorted({r["shard"] for r in rows})
    for path in paths:
        expected = next(s["sha256"] for s in index["shards"] if s["path"]==path)
        if sha256(path)!=expected:
            raise ValueError("training shard hash mismatch")
    print("Verified six training shards; no holdout payloads read",flush=True)
    states = {p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in paths}
    methods = [f"parent_{s}" for s in cfg["parent_sides"]]+cfg["controls"]
    cell = build["cell_mpc"]*build["coordinate_h"]
    results,technical = [],[]
    for row in rows:
        core = context_slice([48]*3,32)
        with h5py.File(row["shard"],"r") as f:
            g = f[row["group"]]
            delta = g["delta_r7_gaussian"][:].astype(np.float64)
            truth = g["tensor_spectral"][core].astype(np.float64)
            support = g["masks/observed_parent"][core].astype(bool)
        tensors = {}
        for side in cfg["parent_sides"]:
            local = centred_crop(delta,side)
            tensor = tensor_from_delta(local,cell)
            residual = float(np.max(np.abs(tensor[...,[0,3,5]].sum(axis=-1)-local)))
            if residual>cfg["technical_trace_max_abs"] or not np.isfinite(tensor).all():
                raise ValueError("parent tensor numerical failure")
            technical.append(residual)
            tensors[f"parent_{side}"] = tensor[context_slice([side//2]*3,32)]
        tensors["96_no_parent_mean"] = tensors["parent_96"].copy()
        tensors["96_no_parent_mean"][...,[0,3,5]] -= float(delta.mean())/3
        true_eigen = eigs(truth)
        item = {k:row[k] for k in ("anchor_id","phase","cap","shell","support_stratum")}
        item["masks"] = {}
        for mask_name,mask in (("all_science_voxels",np.ones((32,)*3,dtype=bool)),("observed_science_voxels",support)):
            if not mask.any():
                raise ValueError("empty science support")
            reference = science(true_eigen,mask,cell,cfg)
            oracle = traceless_constant(tensors["parent_96"]-truth,mask)
            tensors["96_oracle_constant_traceless_residual_removed"] = tensors["parent_96"]-oracle
            values = {}
            for name in methods:
                candidate = science(eigs(tensors[name]),mask,cell,cfg)
                metrics = tensor_metrics(tensors[name],truth,mask)
                gates = gate_values(metrics,candidate,reference)
                values[name] = {"metrics":metrics,"science":candidate,"gate_values":gates,
                                "failed_gates":failures(gates,cfg["tolerances"])}
            item["masks"][mask_name] = {**values,"reference":reference,"oracle_removed_tensor":oracle.tolist()}
        results.append(item)
        print(f"checked {len(results)}/96 {row['anchor_id']}",flush=True)
    report = {"registration":registration,"parents":results,"summary":describe(results,methods,cfg),
        "by_phase":{p:describe([r for r in results if r["phase"]==p],methods,cfg) for p in cfg["phases"]},
        "by_shell_support":{f"{shell}/{support}":describe([r for r in results if r["shell"]==shell and r["support_stratum"]==support],methods,cfg)
                            for shell,support in sorted({(r["shell"],r["support_stratum"]) for r in results})},
        "experimental_unit":"three cosmological phases; anchors/caps/voxels correlated; no IID voxel errors or significance"}
    eligible = [s for s in cfg["parent_sides"] if all(report["summary"][f"parent_{s}"][m]["failed_parents"]==0 for m in cfg["masks"])]
    sensitivity = {str(scale):{str(side):sum(any(failures(r["masks"][m][f"parent_{side}"]["gate_values"],cfg["tolerances"],scale)
                      for m in cfg["masks"]) for r in results) for side in cfg["parent_sides"]} for scale in cfg["sensitivity_multiplier_grid"]}
    if {str(p.relative_to(REPO)):sha256(p) for p in source_paths}!=registration["source_sha256"] or sha256(config_path)!=registration["config_sha256"]:
        raise ValueError("code/config changed during audit")
    if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=state for p,state in states.items()):
        raise ValueError("training payload changed during audit")
    write_json(out/"DOMAIN_GATE_REPORT.json",report)
    verdict = {"registration":registration,"technical_pass":True,"report_sha256":sha256(out/"DOMAIN_GATE_REPORT.json"),
        "eligible_parent_sides":eligible,"selected_screen_candidate":min(eligible) if eligible else None,
        "domain_decision":"NO_REGISTERED_DOMAIN_QUALIFIES" if not eligible else "SCREEN_PASS_NOT_SCIENCE_RELEASE",
        "failed_anchor_counts_by_tolerance_multiplier":sensitivity,
        "max_numerical_trace_residual":max(technical),"job_id":os.environ["SLURM_JOB_ID"],
        "node":socket.gethostname(),"elapsed_seconds":time.monotonic()-started,
        "training_ready":False,"r0_physics_pass":False,"posterior_scaled_budget_evaluated":False,
        "holdout_payloads_read":False,"learned_training_performed":False,
        "remaining":"model/transform release conditional on R0; realistic diagnostic power and posterior-scaled budget unresolved"}
    write_json(out/"DOMAIN_GATE_COMPLETE.json",verdict)
    print(f"DOMAIN AUDIT COMPLETE: {verdict['domain_decision']}; training remains closed",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    run(parser.parse_args().config)
