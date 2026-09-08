#!/usr/bin/env python3
"""Bounded metadata summaries and provenance archive for the E2E physics audit."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/pscratch/sd/d/dkololgi/abacus/e2e_field_v1/error_budget_20260908")


def read(path):
    if path.suffix != ".json" or path.stat().st_size > 32*1024*1024:
        raise ValueError("summary only reads bounded JSON metadata")
    return json.loads(path.read_text())


def digest(path):
    if path.stat().st_size > 32*1024*1024:
        raise ValueError("no payload hashing on login node")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    x = np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("empty/nonfinite summary")
    return {"mean":np.mean(x,axis=0).tolist(),"median":np.median(x,axis=0).tolist(),
            "min":np.min(x,axis=0).tolist(),"max":np.max(x,axis=0).tolist(),
            "abs_p95":np.quantile(np.abs(x),.95,axis=0).tolist(),
            "abs_max":np.max(np.abs(x),axis=0).tolist()}


def mappings():
    result = {"fd2_only":("native_fd2","spectral_legacy_floor"),
              "fd4_only":("fd4","spectral_legacy_floor"),
              "fd8_only":("fd8","spectral_legacy_floor"),
              "floor_vs_linear":("spectral_legacy_floor","spectral_linear"),
              "round_vs_linear":("spectral_nearest_round","spectral_linear"),
              "linear_vs_cubic":("spectral_linear","spectral_local_cubic_lagrange"),
              "clean_linear_no_dc_96":("clean_linear_96_no_dc","spectral_linear")}
    for side in (64,96,128):
        for rule,label in (("legacy_floor","floor"),("linear","linear")):
            result[f"clean_{label}_parent_{side}"] = (f"clean_{rule}_{side}",f"spectral_{rule}")
        if side<=96:
            result[f"inherited_parent_{side}"] = (f"inherited_{side}","native_fd2")
    return result


def summarize(parents,mask):
    output = {}
    for comparison,(candidate,reference) in mappings().items():
        rows = [r["masks"][mask] for r in parents]
        numerical = {key:distribution([r["comparisons"][comparison][key] for r in rows])
                     for key in rows[0]["comparisons"][comparison]}
        p = [r["science"][candidate] for r in rows]
        t = [r["science"][reference] for r in rows]
        science = {key+"_difference":distribution([a[key]-b[key] for a,b in zip(p,t)])
                   for key in ("filling_fraction","largest_void_fraction")}
        pairs = [[a["pair"][i]["value"]-b["pair"][i]["value"] for i in range(3)] for a,b in zip(p,t)]
        science["pair_statistic_difference"] = distribution(pairs)
        changed = [[a["connections_xyz"][i]!=b["connections_xyz"][i] for i in range(3)] for a,b in zip(p,t)]
        science["connection_changed_parent_count"] = int(np.any(changed,axis=1).sum())
        science["connection_changed_axis_event_count"] = int(np.sum(changed))
        science["connection_axis_event_count"] = len(parents)*3
        science["supported_ordered_pairs"] = distribution([[a["ordered_supported_pairs"] for a in b["pair"]] for b in t])
        output[comparison] = {"parents":len(parents),"numerics":numerical,"science":science}
    output["error_gram"] = distribution([r["masks"][mask]["tensor_error_gram"] for r in parents])
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preview-phase",choices=["ph000","ph002","ph003"])
    p.add_argument("--allocation",type=int)
    args = p.parse_args()
    if args.preview_phase:
        phase = read(DATA/f"{args.preview_phase}_ERROR_BUDGET.json")
        result = summarize(phase["parents"],"observed_science_voxels")
        compact = {key:{"eigen_relative_rmse_median":value["numerics"]["eigen_rmse_over_truth_std"]["median"],
                        "filling_absmax":value["science"]["filling_fraction_difference"]["abs_max"],
                        "pair_absmax":value["science"]["pair_statistic_difference"]["abs_max"],
                        "connection_changed_parents":value["science"]["connection_changed_parent_count"]}
                   for key,value in result.items() if key!="error_gram"}
        print(json.dumps(compact,indent=2))
        return
    if not args.allocation:
        p.error("--allocation required for terminal archival")
    complete = read(DATA/"ERROR_BUDGET_COMPLETE.json")
    registration = read(DATA/"FROZEN_AUDIT.json")
    config_path = REPO/"configs/e2e_field_error_budget_v1.json"
    if complete["config_sha256"]!=digest(config_path) or complete["code_sha256"]!=digest(REPO/"workflows/sbi/e2e_field_error_budget.py"):
        raise ValueError("audit source/config drift")
    if [r["phase"] for r in complete["phases"]] != ["ph000","ph002","ph003"]:
        raise ValueError("audit phases incomplete")
    for name,expected in registration["dependency_sha256"].items():
        if digest(REPO/"workflows/sbi"/name)!=expected:
            raise ValueError("audit dependency drift")
    receipt = subprocess.check_output(["sacct","-j",str(args.allocation),"--noheader","--parsable2",
                 "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,Timelimit,MaxRSS,NodeList"],text=True,timeout=30)
    lines = [line.split("|") for line in receipt.splitlines()]
    job = next(x for x in lines if x[0]==str(args.allocation))
    if job[2]!="COMPLETED" or job[3]!="0:0":
        raise ValueError("allocation not terminal-successful")
    report = {"schema_version":"e2e-physical-error-budget-summary-v1",
              "config":read(config_path),"phase_summaries":{},"stratum_summaries":{},
              "units":"eigenvalues dimensionless; science differences absolute fractions, not percent",
              "inference":"descriptive training-only sensitivity; no IID inference or posterior-scaled pass",
              "r0_physics_pass":False,"training_ready":False,
              "software_versions":{x:version(x) for x in ("numpy","scipy","h5py")},
              "source_reports_sha256":{},"scheduler_receipt":receipt}
    all_parents = []
    for phase in complete["phases"]:
        name = phase["phase"]
        all_parents.extend(phase["parents"])
        report["phase_summaries"][name] = {mask:summarize(phase["parents"],mask) for mask in
                                         ("all_science_voxels","observed_science_voxels")}
        report["source_reports_sha256"][name] = digest(DATA/f"{name}_ERROR_BUDGET.json")
    for shell in range(4):
        for support in ("interior","boundary"):
            parents = [r for r in all_parents if r["shell"]==shell and r["support_stratum"]==support]
            report["stratum_summaries"][f"shell{shell}_{support}"] = summarize(parents,"observed_science_voxels")
    out = REPO/"docs/evidence/e2e_field_v1/error_budget_20260908"
    out.mkdir(exist_ok=False)
    def save(name,data):
        with (out/name).open("x") as f:
            json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    save("SUMMARY.json",report)
    save("FROZEN_AUDIT.json",registration)
    for phase in complete["phases"]:
        save(f"{phase['phase']}_ERROR_BUDGET.json",phase)
    save("ARCHIVE_MANIFEST.json",{"complete_report_sha256":digest(DATA/"ERROR_BUDGET_COMPLETE.json"),
         "source_root":str(DATA),"summary_builder_sha256":digest(Path(__file__)),
         "archived_files_sha256":{x.name:digest(x) for x in out.iterdir()},
         "r0_physics_pass":False,"training_ready":False})
    print(json.dumps({"summary":str(out/"SUMMARY.json"),"phases":3,"anchors":len(all_parents)},indent=2))


if __name__ == "__main__":
    main()
