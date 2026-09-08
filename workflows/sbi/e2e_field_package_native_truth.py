#!/usr/bin/env python3
"""Package independent native tensor labels for every prepared E2E parent.

Only numerical/source parity is checked on selection and confirmation phases.
No learned posterior, domain ranking or science functional is evaluated there.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import time

import h5py
import numpy as np

from workflows.sbi.e2e_field_build_products import (
    DEFAULT_CONFIG, eigs, native_indices, read_json, require_compute, sha256,
    validate_config, write_json,
)
from workflows.sbi.e2e_field_native_reference import (
    potential_from_counts, reference_eigenvalues, sample_native_tensor,
)


def package(c, config_path):
    root = Path(c["output_root"])
    index_path = root/"parent_arrays/DATASET_INDEX.json"
    index = read_json(index_path)
    if index["config_sha256"] != sha256(config_path):
        raise ValueError("parent contract differs from native-truth contract")
    screened = read_json(root/"SCREENED_PARENTS.json")
    out = root/"native_truth"
    out.mkdir(exist_ok=True)
    index_hash = sha256(index_path)
    all_phases = []
    side = index["stored_target_side"]
    for phase, role in c["phase_roles"].items():
        receipt_path = out/f"{phase}_COMPLETE.json"
        if receipt_path.exists():
            existing = read_json(receipt_path)
            if (existing["config_sha256"] != sha256(config_path) or
                existing["dataset_index_sha256"] != index_hash or
                existing["builder_sha256"] != sha256(__file__) or
                sha256(existing["shard"]) != existing["shard_sha256"]):
                raise ValueError("existing native truth is not an exact resumable product")
            all_phases.append(existing)
            continue
        started = time.monotonic()
        shard = out/f"{phase}.h5"
        if shard.exists():
            raise FileExistsError("partial native truth preserved; do not silently overwrite")
        sources = [s for s in screened["sources"] if s["phase"] == phase]
        marker = read_json(sources[0]["target_metadata"]["path"])
        slabs = marker["inputs"]["tweb_rank_files"]
        density_dir = Path(slabs[0]["path"]).parents[2]/"density"
        density_path = density_dir/f"AbacusSummit_base_c000_{phase}_z0.200_ngrid2048_ab10_tsc_counts.npy"
        manifest = read_json(density_path.with_suffix(".manifest.json"))
        if manifest["phase"] != phase:
            raise ValueError("native density phase mismatch")
        density_hash = sha256(density_path)
        expected = manifest.get("legacy_source",{}).get("sha256")
        if expected and density_hash != expected:
            raise ValueError("native density checksum mismatch")
        potential, mean = potential_from_counts(density_path,c["box_mpc_h"],c["native_smoothing_mpc_h"])
        rows = [a for a in index["parents"] if a["phase"] == phase]
        worst_eigen, worst_trace = 0., 0.
        with h5py.File(shard,"x") as output:
            output.attrs.update({"phase":phase,"role":role,"training_ready":False,
                                 "density_sha256":density_hash,"tensor_order":"xx,xy,xz,yy,yz,zz",
                                 "independent_native_reference":True,"dataset_index_sha256":index_hash,
                                 "no_posterior_or_science_summary_evaluated":True})
            for ordinal, row in enumerate(rows):
                start = [v-side//2 for v in row["center"]]
                axes = native_indices(row["grid"],start,side,c)
                tensor = sample_native_tensor(potential,axes,c["box_mpc_h"]/c["native_ngrid"])
                stored, _ = reference_eigenvalues(slabs,axes)
                error = float(np.max(np.abs(eigs(tensor)-stored)))
                if error > c["technical_tolerances"]["native_eigenvalue_max_abs"]:
                    raise ValueError(f"native eigenvalue parity failure: {phase} {row['anchor_id']} {error}")
                with h5py.File(row["shard"],"r") as packed:
                    delta = packed[row["group"]]["delta_r7_trace"][:]
                trace_error = float(np.max(np.abs(delta-((stored[...,0]+stored[...,1])+stored[...,2]))))
                if trace_error > c["technical_tolerances"]["source_trace_max_abs"]:
                    raise ValueError("native source trace/coordinate parity failure")
                worst_eigen, worst_trace = max(worst_eigen,error),max(worst_trace,trace_error)
                g = output.create_group(row["anchor_id"])
                g.create_dataset("tensor_native",data=tensor.astype(np.float32),compression="lzf",chunks=(16,16,16,6))
                g.create_dataset("eigenvalues_native",data=stored,compression="lzf",chunks=(16,16,16,3))
                g.attrs["global_voxel_start"] = start
                if ordinal%8 == 7:
                    print(f"native labels {phase}: {ordinal+1}/{len(rows)}",flush=True)
        del potential
        gc.collect()
        receipt = {"schema_version":"e2e-native-parent-truth-v1","phase":phase,"role":role,
                   "config_sha256":sha256(config_path),"dataset_index_sha256":index_hash,
                   "builder_sha256":sha256(__file__),"reference_builder_sha256":sha256(Path(__file__).with_name("e2e_field_native_reference.py")),
                   "job_id":os.environ.get("SLURM_JOB_ID"),"parents":len(rows),
                   "density_path":str(density_path),"density_sha256":density_hash,"density_mean":mean,
                   "density_manifest_sha256":sha256(density_path.with_suffix(".manifest.json")),
                   "native_max_eigenvalue_abs_difference":worst_eigen,"source_trace_max_abs_difference":worst_trace,
                   "numerical_parity_pass":True,"science_domain_adequacy_pass":False,
                   "no_posterior_or_science_summary_evaluated":True,
                   "shard":str(shard),"shard_sha256":sha256(shard),"bytes":shard.stat().st_size,
                   "elapsed_seconds":time.monotonic()-started}
        write_json(receipt_path,receipt)
        all_phases.append(receipt)
        print(f"completed native truth {phase}: {receipt['elapsed_seconds']:.1f}s",flush=True)
    write_json(out/"NATIVE_TRUTH_COMPLETE.json",{"phases":all_phases,"independent_truth_arrays_ready":True,
                                               "training_ready":False,"dataset_index_sha256":index_hash})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args = p.parse_args()
    c = read_json(args.config)
    validate_config(c)
    require_compute()
    package(c,args.config)


if __name__ == "__main__":
    main()
