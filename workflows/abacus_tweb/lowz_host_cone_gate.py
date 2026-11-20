"""Compare real host-cone patches against the frozen full-graph GPU gate."""
import json
from pathlib import Path

import numpy as np
import torch

from workflows.abacus_tweb.lowz_graph_host_cone import owned_patch
from workflows.abacus_tweb.lowz_graph_patch_index import CorrectedPatchAdapter
from workflows.abacus_tweb.lowz_graph_streaming import StreamingGraphPatchNet, owned_patch as tensors
from workflows.abacus_tweb.lowz_graph_union_pilot import digest
from workflows.abacus_tweb.lowz_point_loader import DEV, read, require
from workflows.abacus_tweb.p8_train_graph_patch import grouped_patch

BASE = Path("/pscratch/sd/d/dkololgi/abacus/lowz_gpatch_gpu_61675946/measurement")
GATE_HASH = "8793b55f90ae9cbd9c454304c251acd78221b031407a0d68d4c1a694c70620ee"


class ConeAdapter(CorrectedPatchAdapter):
    def extract(self, core_id, num_passes, *, dependency_hops_per_pass=2, loss_policy="authoritative"):
        require((num_passes,dependency_hops_per_pass,loss_policy)==(2,2,"authoritative"),"wrong host cone task")
        return owned_patch(self,[core_id])


def compare_arrays(ref, alt):
    for k in ("parent_node_id","node_features","loss_mask","authoritative_core_mask","strict_support_mask"):
        np.testing.assert_array_equal(getattr(ref,k),getattr(alt,k))
    # Independent oracle on the already-materialized full reference patch:
    # three simultaneous expansions by its *directed* local edges.
    within = ref.authoritative_core_mask.copy()
    for _ in range(3):
        within[ref.receivers[within[ref.senders]]] = True
    keep = within[ref.receivers]
    for k in ("union_edge_id","senders","receivers","edge_features"):
        np.testing.assert_array_equal(getattr(ref,k)[keep],getattr(alt,k))


def comparison(result, baseline):
    for k in ("seed","source_step","training_window","training_cores_per_epoch"):
        require(result[k]==baseline[k],f"unmatched {k}")
    def identity(g):
        return (g["phase"],g["cap"],tuple(g["cores"]),g["rows"],g["nodes"],g["largest_owned"])
    new, old=result["validation_samples"],baseline["validation_samples"]
    require(len(new)==len(old)==24 and list(map(identity,new))==list(map(identity,old)),
            "different inference workload")
    require(all(0<=a["edges"]<=b["edges"] for a,b in zip(new,old)), "unexpected extra directed edges")
    return dict(baseline_gate_sha256=GATE_HASH,
                training_speed_ratio=result["updates_per_second"]/baseline["updates_per_second"],
                sampled_inference_speed_ratio=sum(g["seconds"] for g in old)/sum(g["seconds"] for g in new),
                retained_sample_edge_fraction=sum(g["edges"] for g in new)/sum(g["edges"] for g in old))


def qualify(view, campaign, out, loader_type):
    gate_path=BASE/"GPATCH_GPU_GATE.json"
    require(digest(gate_path)==GATE_HASH,"unmatched baseline gate")
    baseline=read(gate_path)
    ck=BASE/"lowz_gpatch_preflight/graph/seed_42/arm_a_checkpoint.pt"
    require(baseline["passed"] and digest(ck)==baseline["checkpoint_sha256"],"baseline checkpoint changed")
    model=StreamingGraphPatchNet().cuda().eval()
    model.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["model_state"],strict=True)
    spec=read(view/"transforms/graph/graph_transform.json")["edge"]
    fixtures=[("ph000",int(c["cap"]),[int(c["core"])],True) for c in
              read(campaign/"ph000/PATCH_GEOMETRY_READY.json")["checks"]]
    for p in DEV:
        for cap in (0,1):
            # Exercise the largest non-stress group of each phase/cap; avoid
            # keeping two simultaneous ~90M-edge patches in host memory.
            q=max((x for x in baseline["validation_samples"] if x["phase"]==p and
                   x["cap"]==cap and not x["largest_owned"]),key=lambda x:x["edges"])
            fixtures.append((p,cap,q["cores"],False))
    require(len(fixtures)==8 and {c for p,c,_,one in fixtures if one}=={0,1},"invalid oracle fixtures")
    records=[]
    with loader_type(view,include_blind=False) as loader, torch.inference_mode():
        for p,cap,cores,one in fixtures:
            a=loader.graph_adapter(p)
            require(np.all(a.core_cap[cores]==cap),"oracle cap mismatch")
            ref=a.extract(cores[0],2) if one else grouped_patch(a,cores)
            alt=owned_patch(a,cores)
            compare_arrays(ref,alt)
            def pred(patch):
                return model(*tensors(patch,spec,cap,"cuda"))[patch.loss_mask].cpu()
            y,z=pred(ref),pred(alt)
            torch.testing.assert_close(y,z,rtol=2e-4,atol=3e-5)
            records.append(dict(phase=p,cap=cap,cores=cores,owned=len(y),reference_edges=ref.n_edge,
                                cone_edges=alt.n_edge,max_abs=float((z-y).abs().max())))
            del ref,alt
    marker=dict(passed=True,baseline_gate_sha256=GATE_HASH,checkpoint_sha256=digest(ck),checks=records)
    (out/"HOST_CONE_REAL_PATCH_GATE.json").write_text(json.dumps(marker,indent=2,allow_nan=False))
    return marker
