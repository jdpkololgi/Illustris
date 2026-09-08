#!/usr/bin/env python3
"""Validate E2E packaged data and full-size numerical fixtures on a compute node."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import h5py
import numpy as np

from workflows.sbi.e2e_field_build_products import (
    DEFAULT_CONFIG, read_json, require_compute, sha256, tensor_from_delta,
    validate_config, write_json,
)
from workflows.sbi.e2e_field_dataset import ParentDataset, centred_crop
from workflows.sbi.e2e_field_numerics import (
    AUDIT_LOW_CYCLES_PER_VOXEL, coordinate_noise, evolve_high, haar, low_projection,
)


def relative_l2(a,b):
    return float(np.linalg.norm((a-b).ravel()) / max(np.linalg.norm(b.ravel()),1e-30))


def validate(c,config_path):
    started = time.monotonic()
    root = Path(c["output_root"])
    index_path = root/"parent_arrays/DATASET_INDEX.json"
    index = read_json(index_path)
    if index["config_sha256"] != sha256(config_path):
        raise ValueError("dataset config mismatch")
    native_path = root/"native_truth/NATIVE_TRUTH_COMPLETE.json"
    native = read_json(native_path)
    if native["dataset_index_sha256"] != sha256(index_path):
        raise ValueError("independent truth refers to another dataset")
    density_hashes = [p["density_sha256"] for p in native["phases"]]
    if len(set(density_hashes)) != len(c["phase_roles"]):
        raise ValueError("different phase roles share a density source")
    for shard in index["shards"]:
        if sha256(shard["path"]) != shard["sha256"]:
            raise ValueError("parent shard hash mismatch")
    for phase in native["phases"]:
        if not phase["numerical_parity_pass"] or sha256(phase["shard"]) != phase["shard_sha256"]:
            raise ValueError("native truth hash/parity mismatch")
    try:
        ParentDataset(index_path,"train")
    except PermissionError:
        readiness_guard = True
    else:
        raise ValueError("unreleased data unexpectedly available to training")
    try:
        ParentDataset(index_path,"internal_confirmation",allow_unreleased=True)
    except PermissionError:
        confirmation_guard = True
    else:
        raise ValueError("confirmation guard failed")
    dataset = ParentDataset(index_path,"train",allow_unreleased=True)
    samples = []
    # One source per training phase/cap, selected by manifest order, not truth.
    seen = set()
    for i,row in enumerate(dataset.parents):
        key = (row["phase"],row["cap"])
        if key not in seen:
            seen.add(key)
            samples.append(i)
    checks = []
    for position in samples:
        wide = dataset.parent(position,side=96,normalize=False)
        narrow = dataset.parent(position,side=64,normalize=False)
        if not np.array_equal(narrow["delta"],centred_crop(wide["delta"],64)):
            raise ValueError("nested target mismatch")
        for a,b in zip(narrow["condition"],wide["condition"]):
            if not np.array_equal(a,centred_crop(b,96)):
                raise ValueError("nested observation mismatch")
        normed = dataset.parent(position,side=96)
        if not np.isfinite(normed["condition"]).all() or not np.isfinite(normed["delta"]).all():
            raise ValueError("nonfinite normalized dataset")
        stat = dataset.normalization["target"]
        err = relative_l2(normed["delta"]*stat["std"]+stat["mean"],wide["delta"])
        if err > c["technical_tolerances"]["transform_relative_l2"]:
            raise ValueError("target scaler roundtrip failed")
        checks.append({"anchor_id":wide["anchor_id"],"nested_target_exact":True,
                       "nested_condition_exact":True,"scaler_roundtrip_relative_l2":err})
    numerics = []
    raw = dataset.parent(samples[0],side=96,normalize=False)["delta"].astype(np.float64)
    for side in (64,96):
        x = centred_crop(raw,side)
        low = low_projection(x)
        high = x-low
        tensor = tensor_from_delta(x,c["cell_mpc"]*c["coordinate_h"])
        trace_error = float(np.max(np.abs(tensor[...,[0,3,5]].sum(axis=-1)-x)))
        result = {"side":side,"roundtrip_relative_l2":relative_l2(low+high,x),
                  "low_high_leakage_relative_l2":float(np.linalg.norm(low_projection(high).ravel())/np.linalg.norm(x.ravel())),
                  "parent_mean_abs_error":abs(float(low.mean()-x.mean())),
                  "tensor_trace_max_abs":trace_error,"haar":{}}
        for depth in (1,2):
            wave = haar(high,depth)
            restored = haar(wave,depth,inverse=True)
            result["haar"][str(depth)] = {"relative_l2":relative_l2(restored,high),
                                           "relative_energy_error":abs(float(np.sum(wave**2)/np.sum(high**2)-1))}
        noise = coordinate_noise([-8,40,72],[side]*3,"e2e-engineering:sample0")
        overlap = coordinate_noise([0,48,80],[32]*3,"e2e-engineering:sample0")
        if not np.array_equal(noise[8:40,8:40,8:40],overlap):
            raise ValueError("global noise identity failed")
        direct = evolve_high(noise)
        for offset in (0,16):
            tiled = evolve_high(noise,tiled=True,offset=offset)
            result[f"heun_tiling_offset{offset}_max_abs"] = float(np.max(np.abs(direct-tiled)))
            if not np.array_equal(direct,tiled):
                raise ValueError("synchronous reference sampler is not tile invariant")
        result["coordinate_noise_exact"] = True
        limit = c["technical_tolerances"]["transform_relative_l2"]
        if (result["roundtrip_relative_l2"] > limit or result["low_high_leakage_relative_l2"] > limit or
            trace_error > c["technical_tolerances"]["tensor_trace_max_abs"] or
            any(v["relative_l2"] > limit for v in result["haar"].values())):
            raise ValueError("full-size numerical fixture failed")
        numerics.append(result)
        print(f"full-size data/transform/RNG/Heun checks pass: {side}^3",flush=True)
    report = {"schema_version":"e2e-data-engineering-smoke-v1","config_sha256":sha256(config_path),
              "validator_sha256":sha256(__file__),"job_id":os.environ.get("SLURM_JOB_ID"),
              "dataset_index_sha256":sha256(index_path),"native_truth_manifest_sha256":sha256(native_path),
              "source_phase_density_hashes_unique":True,"all_parent_and_truth_hashes_verified":True,
              "training_normalization_phases":dataset.normalization["fit_phases"],
              "readiness_guard_pass":readiness_guard,"confirmation_guard_pass":confirmation_guard,
              "parent_count":len(index["parents"]),"parent_geometry_view_count":2*len(index["parents"]),
              "distinct_parent_anchors_by_role":{r:sum(p["role"]==r for p in index["parents"])
                                                for r in set(c["phase_roles"].values())},
              "dataset_checks":checks,"numerical_checks":numerics,"data_engineering_pass":True,
              "numerical_split_cycles_per_voxel":AUDIT_LOW_CYCLES_PER_VOXEL,
              "numerical_split_h_mpc":2*np.pi*AUDIT_LOW_CYCLES_PER_VOXEL/(c["cell_mpc"]*c["coordinate_h"]),
              "wavelet_fixture":"orthonormal Haar local pairs, depths 1 and 2; no added package dependency",
              "science_training_cutoff_frozen":False,"actual_neural_sampler_tested":False,
              "r0_physics_pass":False,"diagnostic_power_study_complete":False,"training_ready":False,
              "elapsed_seconds":time.monotonic()-started}
    write_json(root/"DATA_ENGINEERING_SMOKE.json",report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args = p.parse_args()
    c = read_json(args.config)
    validate_config(c)
    require_compute()
    validate(c,args.config)


if __name__ == "__main__":
    main()
