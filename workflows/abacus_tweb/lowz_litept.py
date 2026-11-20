"""Official LitePT-S, with the P12-A scaled-increment task in place of segmentation."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn

from workflows.abacus_tweb import p8_train_unet_patch as grid
from workflows.abacus_tweb.p6_field_patch_utils import trilinear_sample

UPSTREAM = "436d04801c8151faebe66a1b2d368a9711e7e6aa"


def obs_patch(phase_points, field_loader, phase, core):
    """Exact observed positions + the three production channels interpolated to each point."""
    field = field_loader.field_adapter(phase).extract(
        core, grid.HALO_VOXELS, grid.CHANNELS, alignment_voxels=grid.ALIGNMENT_VOXELS)
    return observed_inputs(phase_points, field, field_loader.field_normalization)


def observed_inputs(phase_points, field, normalization):
    rows, xyz, loss_ix = phase_points.patch(field.core_id, halo=grid.HALO_VOXELS)
    np.testing.assert_array_equal(rows[loss_ix], field.authoritative_parent_id)
    cap = ("SGC", "NGC")[field.cap]
    g = phase_points.m["caps"][cap]
    frac = (xyz - g["origin_mpc"])/g["cell_mpc"] - .5 - field.context_start
    channels, _ = grid.model_inputs(field, normalization, "cpu")
    feature = trilinear_sample(channels.numpy()[0], frac)
    if not np.isfinite(feature).all():
        raise ValueError("nonfinite observation channel")
    # Common cap origin, not an arbitrary per-patch rotation or centering.
    return rows, xyz-g["origin_mpc"], feature, loss_ix


def voxels(xyz, features, cell=1.):
    """Average all points in each 1 Mpc voxel; inverse maps every galaxy back."""
    key = np.floor(xyz/cell).astype(np.int64)
    cells, inverse, counts = np.unique(key,axis=0,return_inverse=True,return_counts=True)
    inverse = inverse.reshape(-1)
    def mean(x):
        acc=np.zeros((len(cells),x.shape[1]), dtype=np.float64)
        np.add.at(acc,inverse,x)
        return (acc/counts[:,None]).astype(np.float32)
    return dict(coord=mean(xyz)-xyz.min(0).astype(np.float32),
                grid_coord=(cells-cells.min(0)).astype(np.int32),
                feat=mean(features),offset=np.array([len(cells)],np.int64)), inverse


class LitePTRegressor(nn.Module):
    def __init__(self, src, *, deterministic_attention=False):
        super().__init__()
        src=Path(src).resolve()
        if subprocess.check_output(["git","-C",str(src),"rev-parse","HEAD"],text=True).strip()!=UPSTREAM:
            raise ValueError("unregistered LitePT source")
        sys.path.insert(0,str(src))
        from litept import model
        if Path(model.__file__).resolve()!=src/"litept/model.py":
            raise ImportError("unexpected LitePT implementation")
        if deterministic_attention:
            # PyTorch's deterministic-algorithms switch does not set this
            # third-party function argument. Limit the override to the LitePT
            # module in this dedicated worker; do not rewrite upstream code.
            from functools import partial
            from types import SimpleNamespace
            import flash_attn
            model.flash_attn = SimpleNamespace(flash_attn_varlen_qkvpacked_func=
                partial(flash_attn.flash_attn_varlen_qkvpacked_func, deterministic=True))
        self.backbone=model.LitePT(in_channels=3,shuffle_orders=False)
        self.head=nn.Sequential(nn.Linear(72,128),nn.SiLU(),nn.Linear(128,128),
                                nn.SiLU(),nn.Linear(128,3))

    def forward(self, voxel_input, galaxy_voxel):
        return self.head(self.backbone(voxel_input).feat[galaxy_voxel])


def smoke():
    """Real GPU forward/backward with only the ph002 training labels; no evaluation."""
    import argparse
    import json
    import os
    from workflows.abacus_tweb.lowz_point_loader import ROOT, PointPhase, array, read
    from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
    from workflows.abacus_tweb.p8_deterministic_common import linear_increments,scale_increments
    parser=argparse.ArgumentParser()
    parser.add_argument("index_root",type=Path)
    parser.add_argument("upstream",type=Path)
    args=parser.parse_args()
    if "SLURM_STEP_ID" not in os.environ or not read(args.index_root/"INDEX_READY.json")["pass"]:
        raise RuntimeError("verified point index and GPU step required")
    torch.manual_seed(42)
    net=LitePTRegressor(args.upstream).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.0002)
    phase="ph002"
    points=PointPhase(ROOT,args.index_root,phase)
    with P10PhaseBalancedLoader(ROOT) as loader:
        ids=array(ROOT/"phases"/phase/"training_core_id.npy")
        for cid in ids[np.linspace(0,len(ids)-1,8,dtype=int)]:
            rows,x,feat,ix=obs_patch(points,loader,phase,int(cid))
            d,inv=voxels(x,feat)
            y=scale_increments(linear_increments(loader.targets_by_parent(phase)[rows[ix]]),loader.target_scaler)
            pred=net({k:torch.as_tensor(v,device="cuda") for k,v in d.items()},
                     torch.as_tensor(inv[ix],device="cuda"))
            if pred.shape!=y.shape:
                raise RuntimeError("point row shape mismatch")
            loss=(pred-torch.as_tensor(y,dtype=pred.dtype,device="cuda")).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
            opt.step()
            print(json.dumps(dict(core=int(cid),n=len(ix),context=len(rows),loss=float(loss),
                                  grad_norm=float(norm),peak_bytes=torch.cuda.max_memory_allocated())),flush=True)


if __name__=="__main__":
    smoke()
