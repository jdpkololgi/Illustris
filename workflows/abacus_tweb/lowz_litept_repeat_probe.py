"""Compare native and deterministic-attention kernels at fixed weights, inputs and RNG.

Launch each mode in a separate Python process, on the same GPU. No optimizer
updates: this localizes numerical scatter rather than testing convergence.
"""
import argparse
import json
import os
import random
import numpy as np
import torch
from workflows.abacus_tweb.lowz_litept_train import INDEX, UP
from workflows.abacus_tweb.lowz_litept import LitePTRegressor, obs_patch, voxels
from workflows.abacus_tweb.lowz_point_loader import ROOT, PointPhase, array
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
from workflows.abacus_tweb.p8_deterministic_common import linear_increments, scale_increments


def difference(a, b):
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    return (float((a-b).abs().max()), int((~torch.isclose(a,b,rtol=1e-5,atol=1e-7)).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("native","deterministic_attention"))
    args = ap.parse_args()
    if "SLURM_STEP_ID" not in os.environ:
        raise RuntimeError("GPU compute step required")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model = LitePTRegressor(UP, deterministic_attention=(args.mode=="deterministic_attention")).cuda().train()
    points = PointPhase(ROOT, INDEX, "ph002")
    with P10PhaseBalancedLoader(ROOT) as loader:
        cores = array(ROOT / "phases/ph002/training_core_id.npy")
        truth = loader.targets_by_parent("ph002")
        for cid in cores[np.linspace(0,len(cores)-1,4,dtype=int)]:
            rows, x, feat, ix = obs_patch(points, loader, "ph002", int(cid))
            voxel, inv = voxels(x, feat)
            data = {k:torch.as_tensor(v,device="cuda") for k,v in voxel.items()}
            ix_t = torch.as_tensor(inv[ix],device="cuda")
            y = scale_increments(linear_increments(truth[rows[ix]]),loader.target_scaler)
            target = torch.as_tensor(y,device="cuda",dtype=torch.float32)
            buffers = {k:v.detach().clone() for k,v in model.named_buffers()}
            rng = (torch.get_rng_state(),torch.cuda.get_rng_state(),np.random.get_state(),random.getstate())
            def trial():
                with torch.no_grad():
                    for k,v in model.named_buffers():
                        v.copy_(buffers[k])
                torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
                np.random.set_state(rng[2]); random.setstate(rng[3])
                model.zero_grad(set_to_none=True)
                pred = model({k:v.clone() for k,v in data.items()}, ix_t)
                (pred-target).square().mean().backward()
                grads = {k:v.grad.detach().cpu().clone() for k,v in model.named_parameters() if v.grad is not None}
                return pred.detach().cpu(), grads
            p, g = trial()
            q, h = trial()
            if g.keys() != h.keys():
                raise ValueError("different active parameter sets")
            d = {k:difference(g[k],h[k]) for k in g}
            worst = max(d, key=lambda k:d[k][0])
            print(json.dumps(dict(mode=args.mode,core=int(cid),rows=len(ix),forward=difference(p,q),
                worst_gradient=worst,max_gradient_abs=d[worst][0],
                gradient_elements_outside_tolerance=sum(b for a,b in d.values()),
                incl_backward_peak_bytes=torch.cuda.max_memory_allocated())),flush=True)


if __name__ == "__main__":
    main()
