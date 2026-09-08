#!/usr/bin/env python3
"""Archive small E2E preparation receipts; never read/copy array payloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]


def read_metadata(path):
    path = Path(path)
    if path.suffix != ".json" or path.stat().st_size > 32*1024*1024:
        raise ValueError("archive reads bounded JSON metadata only")
    return json.loads(path.read_text())


def hash_small(path):
    path = Path(path)
    if path.stat().st_size > 32*1024*1024:
        raise ValueError("archive cannot hash array payloads")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=REPO/"configs/e2e_field_build_v1.json")
    p.add_argument("--output",type=Path,default=REPO/"docs/evidence/e2e_field_v1/build_20260908")
    p.add_argument("--allocations",type=int,nargs="+",required=True)
    args = p.parse_args()
    c = read_metadata(args.config)
    root = Path(c["output_root"])
    index = read_metadata(root/"parent_arrays/DATASET_INDEX.json")
    smoke = read_metadata(root/"DATA_ENGINEERING_SMOKE.json")
    native = read_metadata(root/"native_truth/NATIVE_TRUTH_COMPLETE.json")
    if not smoke["data_engineering_pass"] or not native["independent_truth_arrays_ready"]:
        raise ValueError("cannot archive unfinished data preparation as ready")
    if smoke["dataset_index_sha256"] != hash_small(root/"parent_arrays/DATASET_INDEX.json"):
        raise ValueError("index drift after data validation")
    if smoke["native_truth_manifest_sha256"] != hash_small(root/"native_truth/NATIVE_TRUTH_COMPLETE.json"):
        raise ValueError("native reference drift after validation")
    if smoke["config_sha256"] != hash_small(args.config):
        raise ValueError("configuration drift")
    out = args.output.resolve()
    allowed = (REPO/"docs/evidence/e2e_field_v1").resolve()
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError("archive must be a new E2E evidence child")
    out.mkdir(parents=True,exist_ok=False)
    def save(name,data):
        with (out/name).open("x") as stream:
            json.dump(data,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write("\n")
    sources = [
        ("SCREENED_PARENTS.json",root/"SCREENED_PARENTS.json"),
        ("NATIVE_REFERENCE_AUDIT.json",root/"native_audit/NATIVE_REFERENCE_AUDIT.json"),
        ("NATIVE_TRUTH_COMPLETE.json",root/"native_truth/NATIVE_TRUTH_COMPLETE.json"),
        ("DATA_ENGINEERING_SMOKE.json",root/"DATA_ENGINEERING_SMOKE.json"),
        ("TRAIN_NORMALIZATION.json",root/"parent_arrays/TRAIN_NORMALIZATION.json"),
    ]
    for name,source in sources:
        save(name,read_metadata(source))
    scheduler = subprocess.run(
        ["sacct","-j",",".join(map(str,args.allocations)),"--noheader","--parsable2",
         "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,Timelimit,MaxRSS,NodeList"],
        check=True,capture_output=True,text=True,timeout=30)
    save("SCHEDULER_RECEIPT.json",{"allocation_ids":args.allocations,"sacct":scheduler.stdout,
                                   "initial_interruption":"allocation owner shell auto-logout; active step terminated",
                                   "partial_preserved":str(root/"native_truth/ph002_partial_58068916.h5"),
                                   "d2_scheduler_mutations":0})
    source_files = [args.config,*sorted((REPO/"workflows/sbi").glob("e2e_field_*.py")),
                    REPO/"workflows/sbi/run_e2e_data_completion.sh"]
    report = {"schema_version":"e2e-data-build-report-v1",
              "created_utc":datetime.now(timezone.utc).isoformat(),
              "source_snapshot_basis":"base Git revision plus exact source hashes",
              "base_git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
              "source_sha256":{str(x.relative_to(REPO)):hash_small(x) for x in source_files},
              "software_versions":{x:version(x) for x in ("numpy","scipy","h5py","astropy","torch")},
              "data_root":str(root),"dataset_index":str(root/"parent_arrays/DATASET_INDEX.json"),
              "dataset_index_sha256":hash_small(root/"parent_arrays/DATASET_INDEX.json"),
              "anchor_count":len(index["parents"]),"geometry_view_count":2*len(index["parents"]),
              "phase_roles":c["phase_roles"],"array_shards":index["shards"],
              "parent_array_bytes":sum(x["bytes"] for x in index["shards"]),
              "native_truth_bytes":sum(x["bytes"] for x in native["phases"]),
              "data_engineering_ready":True,"independent_truth_ready":True,
              "training_ready":False,"r0_physics_pass":False,"external_confirmation":None,
              "scientific_release_blockers":index["remaining_gates"],
              "final_science_training_population_frozen":False,"learned_models_trained":0,
              "ph001_or_ph006_payloads_accessed":False,
              "archived_metadata":{name:hash_small(out/name) for name,_ in sources}}
    save("BUILD_REPORT.json",report)
    print(json.dumps({"report":str(out/"BUILD_REPORT.json"),"anchors":report["anchor_count"],
                      "parent_array_bytes":report["parent_array_bytes"],"native_truth_bytes":report["native_truth_bytes"],
                      "data_engineering_ready":True,"training_ready":False},indent=2))


if __name__ == "__main__":
    main()
