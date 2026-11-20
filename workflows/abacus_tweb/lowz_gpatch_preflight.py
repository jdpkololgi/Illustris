"""Price the corrected graph arm with actual Arm-A updates and real grouped inference.

No subset R², tuning, early stopping, or blind phases in this gate. Stage
only after the CPU campaign's transformed loader checks have passed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from workflows.abacus_tweb import lowz_gpatch_train as gp
from workflows.abacus_tweb import p10_train_arm_a as trainer
from workflows.abacus_tweb import p8_train_graph_patch as graph
from workflows.abacus_tweb.lowz_point_loader import ROOT, DEV, array, read, require
from workflows.abacus_tweb.lowz_graph_union_pilot import digest


def probe_groups(cores, caps, size=4, per_cap=4):
    """Sample actual contiguous validation groups, not arbitrary far-apart cores."""
    groups = []
    for cap in (0,1):
        ids=np.sort(cores[caps[cores]==cap])
        starts=np.arange(0,len(ids),size)
        require(len(starts)>=per_cap,"too few groups to sample")
        sampled=np.linspace(0,len(starts)-1,per_cap,dtype=int)
        groups.extend((cap,ids[starts[i]:starts[i]+size]) for i in sampled)
    return groups


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("campaign",type=Path)
    ap.add_argument("output",type=Path)
    a=ap.parse_args()
    require("SLURM_STEP_ID" in os.environ and torch.cuda.is_available(),"GPU compute step required")
    receipt=read(a.campaign/"GRAPH_LOADER_CHECKED.json")
    view=Path(receipt["overlay"])
    require(receipt["passed"] and receipt["graph_view_sha256"]==digest(view/"GRAPH_VIEW_READY.json")
            and receipt["transform_sha256"]==digest(view/"transforms/graph/graph_transform.json"),
            "CPU prerequisites not qualified")
    a.output.mkdir(exist_ok=False)
    # Read the clock at logged, completed updates. The trainer synchronizes the
    # loss and gradient values; the first 64 include cold adapter startup.
    marks={}
    original=trainer.append_jsonl
    def timed(path,row):
        original(path,row)
        if path.name=="loss_trace.jsonl" and row["global_step"] in (64,256):
            marks[row["global_step"]]=time.monotonic()
    trainer.append_jsonl=timed
    name="lowz_gpatch_preflight"
    sys.argv=["gpatch","--model","graph","--seed","42","--epochs","4","--min-epochs","4",
        "--lr","0.002","--disable-early-stopping","--validation-group-cores","4",
        "--checkpoint-every","256","--loss-log-every","16","--stop-after-updates","256",
        "--contract-root",str(view),"--output-root",str(a.output),"--run-name",name]
    gp.main()
    d=a.output/name/"graph/seed_42"
    gate=read(d/"TECHNICAL_CANARY_COMPLETE.json")
    require(gate["pass"] and gate["global_step"]==gate["cursor"]==256 and gate["epoch"]==1,
            "no matched 256-update checkpoint")
    elapsed=marks[256]-marks[64]
    require(elapsed>0,"invalid timing")
    trace=[json.loads(line) for line in (d/"loss_trace.jsonl").read_text().splitlines()]
    require(all(np.isfinite(r["optimizer_objective_window_mean"]) and
                np.isfinite(r["preclip_gradient_norm_window_mean"]) for r in trace),"nonfinite training")
    ck=torch.load(d/"arm_a_checkpoint.pt",map_location="cpu",weights_only=False)
    require(ck["global_step"]==ck["resume"]["cursor"]==ck["scheduler_state"]["last_epoch"]==256,
            "inconsistent checkpoint cursor")
    m=graph.GraphPatchNet(latent_size=80,heads=8,dropout=0.1).cuda()
    m.load_state_dict(ck["model_state"])
    edge=read(view/"transforms/graph/graph_transform.json")["edge"]
    checks=[]
    with gp.Loader(view,include_blind=False) as loader:
        for p in DEV:
            adapter=loader.graph_adapter(p)
            cores=array(ROOT/"phases"/p/"validation_core_id.npy")
            for cap,group in probe_groups(cores,adapter.core_cap):
                start=time.monotonic()
                rows,y,fail,n,e=graph.predict_fold(m,adapter,group,edge,"cuda",4)
                duration=time.monotonic()-start
                # Independent P12-A ownership, not the graph's own masked row count.
                field=ROOT/"adapters"/p/"field"
                off=array(field/"core_active_offsets.npy")
                parent=array(field/"core_active_parent.npy")
                expected=np.concatenate([parent[off[i]:off[i+1]] for i in group])
                np.testing.assert_array_equal(np.sort(rows),np.sort(expected))
                require(fail==0 and y.shape==(len(rows),3) and len(np.unique(rows))==len(rows)
                        and np.isfinite(y).all(),"inference group failed")
                checks.append(dict(phase=p,cap=cap,cores=group.tolist(),rows=len(rows),seconds=duration,
                                   nodes=n,edges=e))
    result=dict(passed=True,source_step=256,seed=42,training_window=[64,256],seconds=elapsed,
                updates_per_second=192/elapsed,training_cores_per_epoch=read(d/"run_manifest.json")["training_cores_per_epoch"],
                validation_samples=checks,peak_training_torch_bytes=ck["maximum_memory"],
                peak_torch_bytes_including_inference=torch.cuda.max_memory_allocated(),
                checkpoint_sha256=digest(d/"arm_a_checkpoint.pt"))
    (a.output/"GPATCH_GPU_GATE.json").write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
