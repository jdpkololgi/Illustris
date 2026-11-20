"""Rebuild one phase's Delaunay-plus-radius union after validating reusable geometry."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from workflows.abacus_tweb.lowz_point_loader import ROOT, read, array, roles, require


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("index_root",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--phase",default="ph002")
    args=parser.parse_args()
    require("SLURM_STEP_ID" in os.environ,"compute step required")
    phase=args.phase
    require(phase in roles(ROOT) and read(args.index_root/"INDEX_READY.json")["pass"],"loader not ready")
    r=read(ROOT.parent/"registry.json")
    rec=read(Path(r["reuse_source_root"])/phase/"p2_graph/P2_COMPLETE.json")
    require(rec["pass"] and rec["phase"]==phase,"no valid parent graph")
    src=rec["inputs"]
    for key in ("p1","gnn_metadata","graph","union"):
        require(digest(src[key])==rec["hashes"][key],key+" binding mismatch")
    gm=Path(src["graph_metadata"])
    oldp=read(Path(src["p1"]))
    p=read(ROOT.parent/phase/"p1_canonical/manifest.json")
    new=array(Path(p["points"]))
    old=array(gm.parent/read(gm)["files"]["points"])
    # Same N is insufficient: exact observed Cartesian XYZ, cap, row order and TARGETID.
    require(new.shape==old.shape,"parent shape")
    for i in range(0,len(new),250000):
        np.testing.assert_array_equal(old[i:i+250000],new[i:i+250000])
    with np.load(oldp["index"]) as oi, np.load(p["index"]) as ni:
        for k in ("targetid","parent_node_id","cap"):
            np.testing.assert_array_equal(oi[k],ni[k])
        ncontext=int(ni["context"].sum())
    gn=read(Path(src["gnn_metadata"]))
    require(digest(gn["outputs"]["gnn_arrays_npz"])==rec["hashes"]["gnn_arrays"],"graph payload changed")
    args.output.mkdir(exist_ok=False)
    subprocess.run([sys.executable,"-m","workflows.abacus_tweb.p2b_build_full_radius_union",
        "--graph-dir",str(gm.parent),"--prefix",gm.name.removesuffix("_metadata.json"),
        "--canonical-index",p["index"],"--radius-mpc","14.78","--out-dir",str(args.output)],check=True)
    out=read(args.output/"p2b_union_manifest.json")
    require(out["counts"]["context_nodes"]==ncontext and out["canonical_index"]==p["index"],"context mismatch")
    (args.output/"CORRECTED_UNION_READY.json").write_text(json.dumps(
        dict(phase=phase,context=ncontext,source=src,output_sha256=digest(args.output/"p2b_union_manifest.json"))))


if __name__=="__main__":
    main()
