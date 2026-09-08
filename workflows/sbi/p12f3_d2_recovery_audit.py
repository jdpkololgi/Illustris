#!/usr/bin/env python3
"""Read-only D2 support/coordinate and export-resume audit; compute node only.

Writes only a new audit JSON. Never changes masks, rows, archives or contracts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


def sha(path):
    path = Path(path)
    if "ph001" in str(path):
        raise PermissionError("D2 audit refuses ph001")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("compute allocation required")
    phase_root = args.root.parents[1]
    base = args.root / "evaluation/seed42_v1"
    archive_path = base / "d2_modern_base4_nfe50/P12F_SAMPLE_ARCHIVE.json"
    archive = json.loads(archive_path.read_text())
    if not archive["pass"] or archive["phase"] != "ph006" or archive["ph001_opened"]:
        raise RuntimeError("unsafe archive")
    adapter = phase_root / "training_contract_r1_random/adapters/ph006/field"
    manifest = json.loads((adapter / "adapter_manifest.json").read_text())
    arrays = {name: np.load(adapter / (name+".npy"), mmap_mode="r") for name in
              ("core_voxel_start", "core_active_offsets", "core_active_parent", "core_active_frac_index", "core_cap")}
    handles = {}
    rows = []
    for entry in archive["entries"]:
        path = Path(entry["path"])
        if sha(path) != entry["sha256"]:
            raise RuntimeError("archive shard hash mismatch")
        core = int(entry["core_id"])
        with np.load(path, allow_pickle=False) as data:
            support = data["support"].astype(bool)
            coordinates = data["galaxy_frac_index_local"].astype(np.float64)
            bounds = data["core_bounds"]
        start, stop = arrays["core_active_offsets"][core:core+2]
        parents = arrays["core_active_parent"][int(start):int(stop)]
        global_coordinates = arrays["core_active_frac_index"][int(start):int(stop)]
        context_start = arrays["core_voxel_start"][core] - bounds[0]
        canonical = global_coordinates-context_start
        if not np.array_equal(canonical.astype(np.float32), coordinates.astype(np.float32)):
            raise RuntimeError(f"archive coordinate identity mismatch at core {core}")
        cap = "NGC" if arrays["core_cap"][core] == 1 else "SGC"
        if cap not in handles:
            handles[cap] = h5py.File(manifest["caps"][cap]["field_path"], "r")
        selection = tuple(slice(int(a), int(a+b)) for a,b in zip(context_start,support.shape))
        original_support = np.asarray(handles[cap]["support_random"][selection], dtype=bool)
        if not np.array_equal(support, original_support):
            raise RuntimeError(f"archive support differs from canonical support at core {core}")
        masks = {}
        for name, coords, rule in (("archive_rint",coordinates,np.rint),
                                  ("canonical_rint",canonical,np.rint),
                                  ("canonical_cell",canonical,lambda x:np.floor(x+.5))):
            index = rule(coords).astype(np.int64)
            if np.any(index<0) or np.any(index>=np.asarray(support.shape)):
                raise RuntimeError(f"galaxy outside patch at core {core}")
            masks[name] = support[tuple(index.T)]
        bad = np.flatnonzero(~masks["archive_rint"])
        row = dict(core=core, cap=cap, galaxies=len(coordinates),
                   unsupported={name:int(np.count_nonzero(~mask)) for name,mask in masks.items()},
                   float32_changes=int(np.count_nonzero(masks["archive_rint"]!=masks["canonical_rint"])),
                   cell_rule_changes=int(np.count_nonzero(masks["archive_rint"]!=masks["canonical_cell"])),
                   examples=[dict(parent=int(parents[i]), coordinate=coordinates[i].tolist(),
                                  canonical_coordinate=canonical[i].tolist()) for i in bad[:3]])
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!="examples"}),flush=True)
    for handle in handles.values():
        handle.close()
    progress_path = base / "d2_modern_base4_nfe100/SAMPLE_ARCHIVE_PROGRESS.json"
    progress = json.loads(progress_path.read_text())
    panel = json.loads(Path(archive["panel_marker"]).read_text())
    ids = [int(e["core_id"]) for e in progress["entries"]]
    if ids != panel["selected_core_id"][:len(ids)] or len(set(ids))!=len(ids):
        raise RuntimeError("resume panel prefix invalid")
    for entry in progress["entries"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError("saved 100-step export shard changed")
    output = dict(schema_version="d2-recovery-readonly-audit-v1",created_utc=datetime.now(timezone.utc).isoformat(),
        allocation=os.environ["SLURM_JOB_ID"],source_sha256=sha(__file__),archive_sha256=sha(archive_path),
        progress_sha256=sha(progress_path),support_exact=True,coordinate_identity_exact_float32=True,
        galaxies=sum(r["galaxies"] for r in rows),
        unsupported={name:sum(r["unsupported"][name] for r in rows) for name in masks},
        cores_with_unsupported=sum(r["unsupported"]["archive_rint"]>0 for r in rows),
        float32_changes=sum(r["float32_changes"] for r in rows),
        cell_rule_changes=sum(r["cell_rule_changes"] for r in rows),
        resume_verified_cores=len(ids),remaining_cores=len(panel["selected_core_id"])-len(ids),
        ph001_access=False,records=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output,stream,indent=2)
        stream.write("\n")
    print(json.dumps({k:v for k,v in output.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    main()
