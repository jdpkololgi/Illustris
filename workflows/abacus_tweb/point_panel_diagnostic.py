"""Matched 2048-update diagnostic; scores only the registered 24 development groups."""
import sys, os, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.append("/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/lib/python3.11/site-packages")
import numpy as np
import torch
from workflows.abacus_tweb import p10_train_arm_a as t

arm, out = sys.argv[1], Path(sys.argv[2])
graph = arm == "graph"
r = Path("/pscratch/sd/d/dkololgi/abacus")
root = r / ("lowz_graph_adapters_61668072/corrected_graph_view" if graph else "p12a_lowz_20261029_v1/training_contract_v1")
name = "lowz_gpatch_diag" if graph else "litept_s_diag"
model = "graph" if graph else "unet"
if graph:
    from workflows.abacus_tweb import lowz_gpatch_train as impl
else:
    from workflows.abacus_tweb import lowz_litept_train as impl
sys.argv = [arm,"--model",model,"--seed","42","--epochs","4","--min-epochs","4","--disable-early-stopping","--checkpoint-every","256","--loss-log-every","32","--stop-after-updates","2048","--contract-root",str(root),"--output-root",str(out),"--run-name",name] + (["--lr","0.002","--validation-group-cores","4"] if graph else ["--ordered-prefetch"])
if graph: impl.main(host_cone=True)
else: impl.main()
d = out/name/model/"seed_42"
c = torch.load(d/"arm_a_checkpoint.pt", map_location="cpu", weights_only=False)
assert c["global_step"] == c["resume"]["cursor"] == 2048 and c["resume"]["epoch"] == 1
m = (t.graph_impl.GraphPatchNet(latent_size=80, heads=8, dropout=0.1) if graph else impl.PackedRegressor(impl.UP)).cuda()
m.load_state_dict(c["model_state"])
m.eval()
gate = json.loads((r/"lowz_gpatch_gpu_61683377/measurement/GPATCH_GPU_GATE.json").read_text())
assert gate["passed"] and len(gate["validation_samples"]) == 24
results = {}
with t.P10PhaseBalancedLoader(root,include_blind=False) as loader, torch.no_grad():
    for phase in ("ph006","ph012","ph013"):
        truth = loader.targets_by_parent(phase)
        with np.load(loader.phase_records[phase]["inputs"]["assignment"]) as a:
            shell = np.full(len(truth),-1,dtype=np.int8)
            shell[a["parent_node_id"]]=a["shell"]
        ids, vals = [], []
        f=r/"p12a_lowz_20261029_v1/training_contract_v1/adapters"/phase/"field"
        off=np.load(f/"core_active_offsets.npy",mmap_mode="r")
        parent=np.load(f/"core_active_parent.npy",mmap_mode="r")
        for group in gate["validation_samples"]:
            if group["phase"] != phase: continue
            cores=np.array(group["cores"])
            if graph:
                edge=json.loads((root/"transforms/graph/graph_transform.json").read_text())["edge"]
                ix,y,fail,*_=t.graph_impl.predict_fold(m,loader.graph_adapter(phase),cores,edge,"cuda",4)
            else:
                ix,y,fail=impl.predict(m,loader.field_adapter(phase),cores,loader.field_normalization,"cuda")
            expected=np.concatenate([parent[off[i]:off[i+1]] for i in cores])
            np.testing.assert_array_equal(np.sort(ix),np.sort(expected))
            assert not fail and len(ix)==group["rows"]
            ids.extend(ix); vals.append(t.increments_to_eigenvalues(t.unscale_increments(y,loader.target_scaler)))
        order=np.argsort(ids); ix=np.asarray(ids)[order]; y=np.concatenate(vals)[order].astype(float)
        assert len(np.unique(ix))==len(ix) and np.all(shell[ix]>=0) and np.isfinite(y).all()
        rows={}
        for s in range(4):
            mask=shell[ix]==s; obs=np.asarray(truth[ix[mask]],dtype=float)
            denom=((obs-obs.mean(0))**2).sum(0) if len(obs)>1 else np.zeros(3)
            rows[s]=dict(n=len(obs),r2=(1-((obs-y[mask])**2).sum(0)/denom).tolist() if np.all(denom>0) else None)
        results[phase]=dict(parent_sha256=hashlib.sha256(ix.astype("<i8").tobytes()).hexdigest(),shells=rows)
assert sum(s["n"] for p in results.values() for s in p["shells"].values())==217200
if not graph:
    baseline=json.loads((out/"graph.json").read_text())
    assert all(baseline[p]["parent_sha256"]==v["parent_sha256"] for p,v in results.items())
(out/(arm+".json")).write_text(json.dumps(results,indent=2,allow_nan=False))
