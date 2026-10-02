"""Multi-axis shared-state canvas; engineering extension, not calibration."""
from dataclasses import dataclass
import numpy as np
import torch
from workflows.sbi.e2e_vdm_context_models import project

@dataclass(frozen=True)
class Geometry:
    starts: tuple
    shape: tuple
    origin: tuple
    window: tuple

    @classmethod
    def make(cls, midpoints, stage):
        mid=np.asarray(midpoints,dtype=int)
        if mid.ndim!=2 or mid.shape[1]!=3 or len(mid)==0 or np.any(mid%8):
            raise ValueError('Nonempty raw-cell midpoints aligned to eight required')
        if len(np.unique(mid,axis=0))!=len(mid):raise ValueError('Duplicate windows')
        if stage=='coarse':starts=(mid-192)//8;window=np.array([48,48,48])
        elif stage=='fine':starts=(mid-[64,48,48])//2;window=np.array([64,48,48])
        else:raise ValueError(stage)
        origin=starts.min(0);relative=starts-origin;shape=(relative+window).max(0)
        if stage=='fine' and (np.any(relative%4) or np.any(shape%4)):
            raise ValueError('Fine windows must share one block-mass lattice')
        return cls(tuple(map(tuple,relative)),tuple(shape),tuple(origin),tuple(window))

    def crop(self,i):
        return (slice(None),slice(None),*(slice(int(a),int(a+n)) for a,n in zip(self.starts[i],self.window)))

    def weights(self,device):
        w=torch.zeros((1,1,*self.shape),device=device)
        for i in range(len(self.starts)):w[self.crop(i)]+=1
        return w

@torch.no_grad()
def integrate(model, stage, conditions, state, geometry, nfe=128, on_step=None):
    if nfe<2 or nfe%2 or len(conditions)!=len(geometry.starts):
        raise ValueError('Even positive NFE and matching windows required')
    if tuple(state.shape[2:])!=geometry.shape:raise ValueError('Canvas shape mismatch')
    projector=project if stage=='fine' else lambda x:x
    weights=geometry.weights(state.device);denominator=weights.clamp_min(1)
    state=state.clone()
    def velocity(z,t):
        total=torch.zeros_like(z)
        for i,condition in enumerate(conditions):
            sl=geometry.crop(i);x=z[sl]
            total[sl]+=model(x,model.schedule(x.new_full((len(x),),t)),condition)
        return projector(total/denominator)
    for step in range(nfe//2):
        dt=2/nfe;t=step*dt
        first=velocity(state,t);proposal=projector(state+dt*first)
        state=projector(state+.5*dt*(first+velocity(proposal,t+dt)))
        if on_step is not None:on_step(step+1,state)
    if not torch.isfinite(state).all():raise FloatingPointError('Nonfinite canvas')
    return state,weights
