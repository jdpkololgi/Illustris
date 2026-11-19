"""Bounded three-patch solver and draw-count pilot. Run with frozen production imports."""
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
from workflows.sbi import lowz_coarse_fit as m
from workflows.sbi.e2e_wholefield_pilot_evaluate import CORE, LocalWideTidal
from workflows.sbi.e2e_vdm_context_models import block_mean

NORM = 'ce4e185228982341b52ab560c15090be2c3d1606c661d1add788402656dc46da'
CASES = [('ph006','NGC_s0_boundary_00'), ('ph012','SGC_s0_interior_00'),
         ('ph013','NGC_s3_interior_00')]


def features(operator, rho, wide):
    t = operator.tensor(rho, wide).reshape(-1, 6)
    eig = np.linalg.eigvalsh(t[:, [0,1,2,1,3,4,2,4,5]].reshape(-1,3,3))[:, ::-1]
    return np.column_stack((rho[CORE].ravel(), eig))


def score(x, y):
    q, r = np.quantile(x, [.05,.95], axis=0)
    spread = sum(np.abs(a-b) for i,a in enumerate(x) for b in x[:i])/len(x)**2
    return dict(crps=(abs(x-y).mean(0)-spread).mean(0).tolist(),
                bias=(x.mean(0)-y).mean(0).tolist(),
                coverage90=((y>=q)&(y<=r)).mean(0).tolist(), width90=(r-q).mean(0).tolist())


@torch.inference_mode()
def main():
    p = m.pilot
    m.fine.t.setup(17)
    root = p.c.ROOT.parent
    dest = root/('lowz_patch_pilot_'+os.environ['SLURM_JOB_ID'])
    dest.mkdir(exist_ok=False)
    chart = p.views.load_chart(NORM)
    coarse = p.CoupledBackbone('coarse','wide',base=24,levels=3)
    fine = m.fine.t.FineModel('unet','x')
    for model, path, digest in [(coarse,root/'lowz_coarse_velocity_seed17_v1/LATEST.pt','b79083d1df2a4129dc9511d1aa7a9d29ededa61d6b139656f66086520f75e6dd'),(fine,root/'lowz_unet_endpoint_20261118_seed17/LATEST.pt','90a6f257c013160c330f75a7e2effbfa441ac39b025b241dd2f50907c96259c7')]:
        assert p.c.sha256(path)==digest
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['ema'])
        model.cuda().eval()
    op = LocalWideTidal()
    ten = lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float32)[None]
    for k,(phase,suffix) in enumerate(CASES):
        address = phase+'_'+suffix
        directory = m.fine.t.ROOT/'data'/phase/address
        obs = p.views.load_observations(phase,address,NORM)
        cond = p.FieldCondition(ten(obs['joint']),ten(obs['wide']),ten(obs['joint_center_from_wide_mpc_h']),'wide')
        with np.load(directory/'OBSERVATIONS.npz') as f:
            wide = p.views.scale_channels(f['wide'],chart['wide'])
            np.testing.assert_allclose(wide,obs['wide'],atol=1e-5,rtol=0)
            joint = ten(p.views.scale_channels(f['joint'],chart['joint']))
        levels = {}
        for steps,nfe,n in [(8,16,4),(16,32,8),(32,64,4)]:
            draws = []
            start = time.monotonic()
            error = 0.
            for i in range(n):
                seed = 87000+100*k+i
                z = m.coarse_models.sample(coarse,cond,'cfm',steps,[seed])
                noise = torch.randn((1,1,*m.SHAPE),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed+10000))
                u = m.fine_models.sample(fine,m.fine_models.Condition(joint,ten(wide),z,'sampled'),noise,nfe=nfe)
                cs = chart['coarse_logrho']
                w = torch.exp(z.double()*cs['std'][0]+cs['mean'][0])
                mass = w[(...,*m.COARSE_CROP)]
                rho = m.decode_density(mass,u.double()*chart['fine_residual']['std'][0])
                assert torch.isfinite(rho).all() and rho.min()>0
                error = max(error,((block_mean(rho)-mass)/mass).abs().max().item())
                assert error < 1e-8
                draws.append(features(op,rho[0,0].cpu().numpy(),w[0,0].cpu().numpy()))
            levels[nfe] = np.stack(draws)
            with np.load(directory/'TARGETS.npz') as f:
                truth = features(op,f['rho'].astype(float),f['coarse_wide'].astype(float))
            report = dict(address=address,host=socket.gethostname(),coarse_steps=steps,fine_nfe=nfe,n=n,seconds=time.monotonic()-start,max_mass_error=error,score=score(levels[nfe],truth),half=score(levels[nfe][:n//2],truth))
            if nfe>16:
                report['paired_mean_abs_change'] = abs(levels[nfe][:4]-levels[nfe//2][:4]).mean((0,1)).tolist()
            p.c.atomic_json(dest/f'{address}_{nfe}.json',report)
            print(address,nfe,report['seconds'],flush=True)
    p.c.atomic_json(dest/'COMPLETE.json',dict(cases=CASES,scope='patch convergence pilot only'))

if __name__=='__main__':
    main()
