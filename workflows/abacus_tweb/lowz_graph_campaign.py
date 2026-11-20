"""Materialize and qualify all corrected graph views after component gates pass."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from workflows.abacus_tweb.lowz_point_loader import DEV, ROOT, array, roles, read, require
from workflows.abacus_tweb.lowz_graph_union_pilot import digest


def context_boundary(phase, union):
    # Filtering parent geometry is not re-tessellation. Quantify the direct
    # boundary contact separately; never interpret this as a matched-channel
    # or an edge/feature-equivalence proof for a context-only re-tessellation.
    src = read(union/"CORRECTED_UNION_READY.json")["source"]
    gm = Path(src["graph_metadata"])
    edges = array(gm.parent/read(gm)["files"]["edges"])
    p1 = read(ROOT/"phases"/phase/"phase_contract.json")["inputs"]["p1_manifest"]
    with np.load(read(Path(p1))["index"]) as idx:
        ctx = idx["context"].astype(bool)
    touch = np.zeros(len(ctx),bool)
    crossing = 0
    for start in range(0,len(edges),500_000):
        u,v = edges[start:start+500_000].T
        left,right = ctx[u],ctx[v]
        touch[u[left & ~right]] = True
        touch[v[right & ~left]] = True
        crossing += int((left != right).sum())
    owned = array(ROOT/"adapters"/phase/"field/core_active_parent.npy")
    return dict(excluded_parent_rows=int((~ctx).sum()),crossing_parent_edges=crossing,
                directly_adjacent_context=int(touch.sum()),directly_adjacent_authoritative=int(touch[owned].sum()))


def run(output, unions, pilot):
    require("SLURM_STEP_ID" in os.environ,"compute step required")
    all_union = read(unions)
    visible = roles(ROOT)
    require(all_union["passed"] and set(all_union["phases"])==set(visible),"incomplete corrected union set")
    p0 = read(pilot/"PATCH_GEOMETRY_READY.json")
    require(p0["passed"] and p0["phase"]=="ph002" and len(p0["checks"])==2
            and p0["union_sha256"]==all_union["phases"]["ph002"]["manifest_sha256"],"bad ph002 pilot")
    require(output.is_dir() and not (output/"graph_index.json").exists(),"expected new job directory")
    def execute(mod,*args):
        subprocess.run([sys.executable,"-u","-m","workflows.abacus_tweb."+mod,*map(str,args)],check=True)
    index, boundary = {}, {}
    for p in visible:
        u=all_union["phases"][p]
        d=Path(u["directory"])
        require(digest(d/"p2b_union_manifest.json")==u["manifest_sha256"],"union changed")
        phase_out = pilot if p=="ph002" else output/p
        if p!="ph002":
            execute("lowz_graph_patch_index",p,d,phase_out)
        marker=read(phase_out/"PATCH_GEOMETRY_READY.json")
        require(marker["passed"] and marker["phase"]==p and marker["union_sha256"]==u["manifest_sha256"]
                and {s["cap"] for s in marker["checks"]}=={0,1}, f"{p}: unverified patches")
        index[p]=str(phase_out.resolve())
        boundary[p]=context_boundary(p,d)
        print(p,"graph patches qualified",flush=True)
    path=output/"graph_index.json"
    path.write_text(json.dumps(index,indent=2))
    transform=output/"graph_features"
    execute("lowz_graph_normalize",path,transform)
    overlay=output/"corrected_graph_view"
    execute("lowz_gpatch_train","bind",path,transform,overlay)
    from workflows.abacus_tweb.lowz_gpatch_train import Loader
    from workflows.abacus_tweb.p8_train_graph_patch import transformed_patch
    edge=read(transform/"graph_transform.json")["edge"]
    measured={}
    with Loader(overlay,include_blind=False) as loader:
        for p in visible:
            ids=array(ROOT/"phases"/p/("validation_core_id.npy" if p in DEV else "training_core_id.npy"))
            adapter=loader.graph_adapter(p)
            measured[p]=[]
            for c in (0,1):
                eligible=ids[adapter.core_cap[ids]==c]
                cid=int(eligible[len(eligible)//2])
                patch=adapter.extract(cid,2)
                n,e,s,r=transformed_patch(patch,edge,c,"cpu")
                require(n.shape==(patch.n_node,8) and e.shape==(patch.n_edge,5)
                        and bool(n.isfinite().all()) and bool(e.isfinite().all()),"invalid transformed patch")
                np.testing.assert_array_equal(patch.parent_node_id[patch.loss_mask],adapter.core_nodes(cid)[0])
                measured[p].append(dict(cap=c,nodes=patch.n_node,edges=patch.n_edge,owned=int(patch.loss_mask.sum())))
    result=dict(passed=True,base=str(ROOT),overlay=str(overlay),graph_view_sha256=digest(overlay/"GRAPH_VIEW_READY.json"),
                transform_sha256=digest(transform/"graph_transform.json"),patches=measured,
                parent_delaunay_context_boundary=boundary)
    (output/"GRAPH_LOADER_CHECKED.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("output","unions","pilot"): ap.add_argument(key,type=Path)
    a=ap.parse_args()
    run(a.output,a.unions,a.pilot)
