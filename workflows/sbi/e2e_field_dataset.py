#!/usr/bin/env python3
"""Pack and load source-verified E2E parent arrays, without promoting science.

One maximum-size target/context per anchor supplies both nested geometries.
Readiness is layered: successful packaging never substitutes for R0-PHYSICS,
diagnostic-power or the final model/training contract.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
from astropy.cosmology import Planck18

from workflows.sbi.e2e_field_build_products import (
    DEFAULT_CONFIG, context_slice, read_json, require_compute, sha256,
    validate_config, write_json,
)
from workflows.sbi.e2e_field_prepare_data import safe_path

CHANNEL_TRANSFORMS = {"counts": "log1p", "expected_counts_random": "log1p",
                      "ntilde_mpc3": "log_floor_1e-10"}
STANDARDIZED = {"counts", "expected_counts_random", "log_count_ratio_random",
                "ntilde_mpc3", "distance_to_support_boundary", "observer_redshift"}


def transformed(name, data):
    x = np.asarray(data, dtype=np.float32)
    if name in ("counts", "expected_counts_random"):
        if np.min(x) < 0:
            raise ValueError("negative count/expectation")
        return np.log1p(x)
    if name == "ntilde_mpc3":
        if np.min(x) < 0:
            raise ValueError("negative number density")
        return np.log(np.maximum(x, 1e-10))
    return x


class Moments:
    """Pooled voxel moments; identical volume per parent gives equal parent weight."""
    def __init__(self):
        self.n, self.total, self.square = 0, 0.0, 0.0

    def add(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.n += x.size
        self.total += float(x.sum())
        self.square += float(np.square(x).sum())

    def report(self):
        if self.n == 0:
            raise ValueError("normalization has no training voxels")
        mean = self.total/self.n
        variance = max(0.0, self.square/self.n-mean*mean)
        return {"mean": mean, "std": max(float(np.sqrt(variance)),1e-6), "voxel_presentations": self.n}


def centred_crop(x, side):
    if x.ndim != 3 or any(n < side or (n-side)%2 for n in x.shape):
        raise ValueError("incompatible nested parent crop")
    return x[tuple(slice((n-side)//2,(n+side)//2) for n in x.shape)]


def child_layout(center, side, core=32, halo=8):
    start = np.asarray(center)-side//2
    if side % core:
        raise ValueError("child cores do not tile parent")
    children = []
    for index in np.ndindex(*([side//core]*3)):
        lo = start + np.asarray(index)*core
        hi = lo+core
        children.append({"core_start": lo.tolist(), "core_stop": hi.tolist(),
                         "halo_start": np.maximum(start,lo-halo).tolist(),
                         "halo_stop": np.minimum(start+side,hi+halo).tolist()})
    return children


def verify_sources(source):
    verified = []
    for key in ("target_file", "response_file"):
        item = source[key]
        actual = sha256(item["path"])
        if actual != item["recorded_sha256"]:
            raise ValueError(f"source payload checksum mismatch: {key} {source['phase']}/{source['cap']}")
        verified.append({"path": item["path"], "sha256": actual, "kind": key})
    marker = read_json(source["target_metadata"]["path"])
    p3_path = marker["inputs"]["p3_manifest"]
    if sha256(p3_path) != marker["inputs"]["p3_manifest_sha256"]:
        raise ValueError("P3 base manifest hash mismatch")
    p3 = read_json(p3_path)
    known = {str(Path(v["file"]).resolve()): v["file_sha256"] for v in p3["components"].values()}
    for path in sorted({v["path"] for v in source["virtual_sources"]}):
        resolved = str(safe_path(Path(path)).resolve())
        expected = known.get(resolved)
        # ph000 imports byte-identical legacy cap fields, while its immutable
        # manifest retains their pre-import paths. Accept only the declared cap
        # checksum, never merely a matching filename or file size.
        legacy_alias = False
        if expected is None and source["phase"] == "ph000":
            expected = p3["components"][source["cap"]]["file_sha256"]
            legacy_alias = True
        if expected is None:
            raise ValueError("virtual dataset has no registered P3 source")
        actual = sha256(path)
        if actual != expected:
            raise ValueError("virtual dataset source checksum mismatch")
        verified.append({"path": resolved, "sha256": actual, "kind": "virtual_dataset_source",
                         "byte_identical_legacy_import": legacy_alias})
    return verified


def pack(c, config_path):
    root = Path(c["output_root"])
    screened_path = root / "SCREENED_PARENTS.json"
    screened = read_json(screened_path)
    if screened["config_sha256"] != sha256(config_path):
        raise ValueError("screened contract hash mismatch")
    # This is only a numerical-reference requirement, not the domain adequacy gate.
    native_path = root / "native_audit/NATIVE_REFERENCE_AUDIT.json"
    native = read_json(native_path)
    if not native.get("native_tensor_replay_pass") or not native.get("source_trace_coordinate_pass"):
        raise ValueError("independent native reference is not validated")
    destination = root / "parent_arrays"
    destination.mkdir(exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("refusing to mix with existing parent products")
    started = time.monotonic()
    side = max(c["parent_sides"])
    context_side = side+2*c["observation_halo"]
    channels = c["condition_channels"] + ["observer_redshift"]
    moments = {name: Moments() for name in channels if name in STANDARDIZED}
    target_moments = Moments()
    z_grid = np.linspace(0,0.8,4001)
    r_grid = Planck18.comoving_distance(z_grid).value
    rows, shards, provenance = [], [], []
    for source in screened["sources"]:
        phase, cap = source["phase"], source["cap"]
        if phase not in c["phase_roles"]:
            raise ValueError("phase outside frozen roles")
        verified = verify_sources(source)
        provenance.extend(verified)
        before = {v["path"]: (Path(v["path"]).stat().st_size,Path(v["path"]).stat().st_mtime_ns)
                  for v in verified}
        shard = destination/f"{phase}_{cap}.h5"
        anchors = [a for a in screened["anchors"] if a["phase"] == phase and a["cap"] == cap]
        with h5py.File(source["target_file"]["path"],"r") as target, \
             h5py.File(source["response_file"]["path"],"r") as response, h5py.File(shard,"x") as out:
            out.attrs.update({"phase": phase, "role": source["role"], "cap": cap,
                              "config_sha256": sha256(config_path), "training_ready": False,
                              "target_semantics": "frozen nearest-native-cell trace of R7 finite-difference tidal tensor"})
            for anchor in anchors:
                g = out.create_group(anchor["anchor_id"])
                g.attrs["center"] = anchor["center"]
                g.attrs["origin_mpc"] = anchor["grid"]["origin_mpc"]
                g.attrs["cell_mpc"] = c["cell_mpc"]
                g.attrs["coordinate_h"] = c["coordinate_h"]
                g.attrs["role"] = anchor["role"]
                parent_slice = context_slice(anchor["center"],side)
                wide_slice = context_slice(anchor["center"],context_side)
                delta = np.asarray(target["delta_r7"][parent_slice], dtype=np.float32)
                if delta.shape != (side,)*3 or not np.isfinite(delta).all():
                    raise ValueError("invalid complete latent target")
                g.create_dataset("delta_r7_trace",data=delta,compression="lzf",chunks=(32,32,32))
                if source["role"] == "train":
                    target_moments.add(delta)
                condition = g.create_group("condition_raw")
                for name in c["condition_channels"]:
                    x = np.asarray(response[name][wide_slice], dtype=np.float32)
                    if x.shape != (context_side,)*3 or not np.isfinite(x).all():
                        raise ValueError(f"invalid condition {name}")
                    condition.create_dataset(name,data=x,compression="lzf",chunks=(32,32,32))
                    tx = transformed(name,x)
                    if name in moments and source["role"] == "train":
                        moments[name].add(tx)
                xyz = [anchor["grid"]["origin_mpc"][a] +
                       (np.arange(wide_slice[a].start,wide_slice[a].stop)+.5)*c["cell_mpc"] for a in range(3)]
                radius = np.sqrt(xyz[0][:,None,None]**2+xyz[1][None,:,None]**2+xyz[2][None,None,:]**2)
                redshift = np.interp(radius,r_grid,z_grid).astype(np.float32)
                condition.create_dataset("observer_redshift",data=redshift,compression="lzf",chunks=(32,32,32))
                if source["role"] == "train":
                    moments["observer_redshift"].add(redshift)
                masks = g.create_group("masks")
                for name in ("latent_domain","truth_available","loss_domain"):
                    masks.create_dataset(name,data=np.ones((side,)*3,dtype=np.uint8),compression="lzf")
                core = np.zeros((side,)*3,dtype=np.uint8)
                core[context_slice([side//2]*3,c["science_core_side"])] = 1
                observed = np.asarray(response["support_random"][parent_slice],dtype=np.uint8)
                masks.create_dataset("science_core_geometry",data=core,compression="lzf")
                masks.create_dataset("observed_parent",data=observed,compression="lzf")
                masks.create_dataset("science_supported",data=core*observed,compression="lzf")
                rows.append({**anchor,"shard": str(shard),"group": anchor["anchor_id"],
                             "available_parent_sides": c["parent_sides"],
                             "children_by_side": {str(n): child_layout(anchor["center"],n,c["child_core_side"],c["child_halo"])
                                                  for n in c["parent_sides"]}})
        for p,state in before.items():
            if (Path(p).stat().st_size,Path(p).stat().st_mtime_ns) != state:
                raise ValueError("source changed during extraction")
        shards.append({"path": str(shard), "bytes": shard.stat().st_size,"sha256": sha256(shard)})
        print(f"packed {phase}/{cap}: {len(anchors)} anchors, {shard.stat().st_size/1e9:.3f} GB",flush=True)
    normalization = {"fit_roles": ["train"],"fit_phases": [p for p,r in c["phase_roles"].items() if r=="train"],
                     "fit_population": "equal-volume largest parent targets / wide conditions; equal weight per anchor",
                     "target": target_moments.report(), "channels": {k:v.report() for k,v in moments.items()},
                     "transforms": CHANNEL_TRANSFORMS,"other_channels": "identity",
                     "target_scaling": "one affine field-space transform; preserve and generate parent mean"}
    write_json(destination/"TRAIN_NORMALIZATION.json",normalization)
    report = {"schema_version": "e2e-parent-arrays-v1","config_sha256": sha256(config_path),
              "builder_sha256": sha256(__file__),"screened_manifest_sha256": sha256(screened_path),
              "native_reference_sha256": sha256(native_path),"job_id": os.environ.get("SLURM_JOB_ID"),
              "phase_roles": c["phase_roles"],"condition_channels": channels,
              "stored_target_side": side,"stored_context_side": context_side,
              "observation_halo": c["observation_halo"],"normalization": str(destination/"TRAIN_NORMALIZATION.json"),
              "normalization_sha256": sha256(destination/"TRAIN_NORMALIZATION.json"),
              "sources_verified": provenance,"parents": rows,"shards": shards,
              "arrays_packaged": True,"training_ready": False,"external_confirmation": None,
              "remaining_gates": ["R0-PHYSICS scientific error budget and domain decision",
                                  "frozen multiscale transform and diagnostic-power contract",
                                  "model/training contract and explicit launch authorization"],
              "elapsed_seconds": time.monotonic()-started}
    write_json(destination/"DATASET_INDEX.json",report)
    return report


class ParentDataset:
    def __init__(self,index_path,role,*,allow_unreleased=False,allow_confirmation=False):
        self.index = read_json(index_path)
        if not self.index["training_ready"] and not allow_unreleased:
            raise PermissionError("E2E science-training gates are not released")
        if role == "internal_confirmation" and not allow_confirmation:
            raise PermissionError("confirmation data requires a separate explicit opening")
        if role not in set(self.index["phase_roles"].values()):
            raise ValueError("unknown data role")
        self.parents = [r for r in self.index["parents"] if r["role"]==role]
        if sha256(self.index["normalization"]) != self.index["normalization_sha256"]:
            raise ValueError("normalization drift")
        self.normalization = read_json(self.index["normalization"])

    def __len__(self):
        return len(self.parents)

    def parent(self,position,side=96,normalize=True):
        row = self.parents[position]
        if side not in row["available_parent_sides"]:
            raise ValueError("unregistered parent geometry")
        ncontext = side+2*self.index["observation_halo"]
        with h5py.File(safe_path(Path(row["shard"])),"r") as f:
            g = f[row["group"]]
            delta = centred_crop(g["delta_r7_trace"][:],side)
            values = []
            for name in self.index["condition_channels"]:
                x = centred_crop(g["condition_raw"][name][:],ncontext)
                if normalize:
                    x = transformed(name,x)
                    if name in self.normalization["channels"]:
                        stat = self.normalization["channels"][name]
                        x = (x-stat["mean"])/stat["std"]
                values.append(x)
            masks = {name: centred_crop(g["masks"][name][:],side).astype(bool) for name in g["masks"]}
        if normalize:
            stat = self.normalization["target"]
            delta = (delta-stat["mean"])/stat["std"]
        return {"anchor_id": row["anchor_id"],"delta": np.asarray(delta,dtype=np.float32),
                "condition": np.stack(values).astype(np.float32),"masks": masks,
                "children": row["children_by_side"][str(side)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args = p.parse_args()
    c = read_json(args.config)
    validate_config(c)
    require_compute()
    pack(c,args.config)


if __name__ == "__main__":
    main()
