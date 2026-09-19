"""Full-shape engineering prototypes for the later coupled-model cost proposal.

No fit entrypoint or scientific training authority. I/J share parameters and
full observation information; only the fine latent domain differs. VDM/CFM use
the identical state-residual output network with different regression targets.
Existing scientific models/checkpoints remain untouched.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from workflows.sbi.e2e_direct_vdm import ConditionalVDM
from workflows.sbi.e2e_vdm_context_models import project,block_mean

FINE_CELL=6.766
WIDE_CELL=27.064
WIDE_SPAN=48*WIDE_CELL
SHAPES={'wide':(48,48,48),'joint':(64,48,48),'left':(48,48,48),'right':(48,48,48)}


def parent_crop(value,region):
    if region=='joint': return value
    if region not in ('left','right'): raise ValueError('unknown fine region')
    start=0 if region=='left' else 16
    return value[:,:,start:start+48,:,:]


def physical_positions(shape,cell,stride,offset):
    """XYZ feature centres after kernel3/pad1/stride2 reductions, in Mpc/h.

    Keep per-axis physical extents. The first feature stays at the FIRST input
    voxel centre; it is not centred halfway across its nominal stride block.
    """
    axes=[(torch.arange(n,device=offset.device,dtype=offset.dtype)*stride+.5-n*stride/2)*cell
          for n in shape]
    points=torch.stack(torch.meshgrid(*axes,indexing='ij'),dim=-1).reshape(-1,3)
    return points[None]+offset[:,None]


def position_encoding(points,width):
    count=math.ceil(width/6)
    frequency=2*math.pi*2**torch.linspace(0,4,count,device=points.device,dtype=points.dtype)
    phases=points[...,None]/WIDE_SPAN*frequency
    return torch.cat((phases.sin(),phases.cos()),dim=-1).flatten(-2)[...,:width]


@dataclass(frozen=True)
class FieldCondition:
    joint: torch.Tensor
    wide: torch.Tensor
    joint_center_from_wide_mpc_h: torch.Tensor
    region: str
    coarse_joint: torch.Tensor | None = None
    coarse_wide: torch.Tensor | None = None
    coarse_source: str | None = None

    def validate(self,z,stage,domain,training):
        expected='wide' if stage=='coarse' else domain
        if (self.region not in SHAPES or
                (expected=='parent' and self.region not in ('left','right')) or
                (expected!='parent' and self.region!=expected)):
            raise ValueError('condition region does not match factor domain')
        if (z.shape!=(len(z),1,*SHAPES[self.region]) or
                self.joint.shape!=(len(z),12,64,48,48) or
                self.wide.shape!=(len(z),12,48,48,48) or
                self.joint_center_from_wide_mpc_h.shape!=(len(z),3)):
            raise ValueError('registered full-shape condition required')
        if stage=='coarse':
            if any(x is not None for x in (self.coarse_joint,self.coarse_wide,self.coarse_source)):
                raise PermissionError('coarse factor must not receive matter-derived inputs')
        else:
            if (self.coarse_joint is None or self.coarse_joint.shape!=(len(z),1,64,48,48)
                    or self.coarse_wide is None or self.coarse_wide.shape!=(len(z),1,48,48,48)):
                raise ValueError('fine factor needs one aligned shared coarse field')
            allowed=('training_truth',) if training else ('sampled','fixed_mean','oracle_diagnostic')
            if self.coarse_source not in allowed:
                raise PermissionError('fine coarse-source provenance is not allowed in this mode')

    def query_offset(self):
        if self.region=='wide': return torch.zeros_like(self.joint_center_from_wide_mpc_h)
        offset=self.joint_center_from_wide_mpc_h
        shift={'left':-8*FINE_CELL,'right':8*FINE_CELL,'joint':0.}[self.region]
        return offset+offset.new_tensor([shift,0.,0.])


class CoupledBackbone(ConditionalVDM):
    """Shared rectangular-aware U-Net; matched I/J and VDM/CFM parameters.

    Coarse inference consumes spatial joint-local AND wide observations. Fine
    parent inference retains the full joint observation through cross-attention;
    it does not condition solely on a cropped local observation. I parents still
    require independent latent/noise paths; sharing features does not share noise.
    """
    def __init__(self,stage,domain,base=24,levels=3):
        if (stage,domain) not in (('coarse','wide'),('fine','parent'),('fine','joint')):
            raise ValueError('unregistered benchmark factor/domain')
        super().__init__(condition_channels=25,base=base,levels=levels,learned=False)
        self.stage,self.domain=stage,domain
        width=base*2**levels
        def encoder():
            return nn.Sequential(nn.Conv3d(13,8,3,2,1),nn.SiLU(),
                nn.Conv3d(8,16,3,2,1),nn.SiLU(),nn.Conv3d(16,32,3,2,1),nn.SiLU(),
                nn.Conv3d(32,width,1))
        self.joint_encoder=encoder(); self.wide_encoder=encoder()
        self.context_norm=nn.LayerNorm(width)
        self.context_attention=nn.MultiheadAttention(width,1,batch_first=True)

    def forward(self,z,clock,condition):
        condition.validate(z,self.stage,self.domain,self.training)
        if any(n%2**self.levels for n in z.shape[2:]): raise ValueError('nondivisible U-Net shape')
        if self.stage=='coarse':
            local=condition.wide; summary=condition.joint.mean((2,3,4),keepdim=True)
            joint_coarse=torch.zeros_like(condition.joint[:,:1])
            wide_coarse=torch.zeros_like(z); local_coarse=wide_coarse
        else:
            local=parent_crop(condition.joint,condition.region)
            summary=condition.wide.mean((2,3,4),keepdim=True)
            joint_coarse=condition.coarse_joint; wide_coarse=condition.coarse_wide
            local_coarse=parent_crop(joint_coarse,condition.region)
        phase=((clock+13.3)/26.6*1000)[:,None]*self.frequency[None]
        embedded=self.time(torch.cat((phase.sin(),phase.cos()),1))
        local_inputs=torch.cat((local,summary.expand_as(local),local_coarse),1)
        h=self.input(torch.cat((z,local_inputs),1)); skips=[]
        for block,down in zip(self.enc,self.down):
            h=block(h,embedded); skips.append(h); h=down(h)
        h=self.mid1(h,embedded); shape=h.shape
        tokens=h.flatten(2).transpose(1,2); q=self.attn_norm(tokens)
        tokens=tokens+self.attn(q,q,q,need_weights=False)[0]
        joint=self.joint_encoder(torch.cat((condition.joint,joint_coarse),1))
        wide=self.wide_encoder(torch.cat((condition.wide,wide_coarse),1))
        width=h.shape[1]
        qpos=physical_positions(h.shape[2:],WIDE_CELL if self.stage=='coarse' else FINE_CELL,
                               2**self.levels,condition.query_offset())
        jpos=physical_positions(joint.shape[2:],FINE_CELL,8,condition.joint_center_from_wide_mpc_h)
        wpos=physical_positions(wide.shape[2:],WIDE_CELL,8,torch.zeros_like(condition.query_offset()))
        values=torch.cat((joint.flatten(2).transpose(1,2),wide.flatten(2).transpose(1,2)),1)
        key_positions=position_encoding(torch.cat((jpos,wpos),1),width)
        queries=self.context_norm(tokens)+position_encoding(qpos,width)
        tokens=tokens+self.context_attention(queries,values+key_positions,values,need_weights=False)[0]
        h=self.mid2(tokens.transpose(1,2).reshape(shape),embedded)
        for up,block,skip in zip(self.up,self.dec,reversed(skips)):
            h=block(torch.cat((up(h),skip),1),embedded)
        output=z+self.output(F.silu(h))
        return project(output) if self.stage=='fine' else output


def objective(model,x,condition,generator,kind,decoder_std=.001):
    """VDM VLB bits/DOF or linear independent-base CFM velocity MSE/DOF.

    CFM uses z(t)=(1-t)eps+t*x and target velocity x-eps, including t=0/1
    in the sampler. Both objectives use exactly the same clock embedding and
    state-residual prediction network. These technical updates are not fits to
    the scientific data or evidence of posterior performance.
    """
    if kind not in ('vdm','cfm'): raise ValueError('unregistered objective')
    fine=model.stage=='fine'; projector=project if fine else lambda v:v
    if fine and block_mean(x).abs().max()>2e-6: raise ValueError('fine target outside residual subspace')
    t=(torch.arange(len(x),device=x.device)+torch.rand((),device=x.device,generator=generator))/len(x)
    eps=projector(torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator))
    expand=lambda v:v[:,None,None,None,None]
    fraction=63/64 if fine else 1.
    clock=model.schedule(t)
    if kind=='cfm':
        z=(1-expand(t))*eps+expand(t)*x
        error=projector(model(z,clock,condition)-(x-eps))
        value=error.square().mean()/fraction
        return value,dict(velocity_mse=value,independent_dimensions=x[0].numel()*fraction)
    a,s=model.schedule.coefficients(clock)
    z=expand(a)*x+expand(s)*eps
    error=projector(model(z,clock,condition)-eps).square().mean()/fraction
    diffusion=.5*model.schedule.slope.abs()*error/math.log(2)
    g0=model.schedule(x.new_zeros(1))[0]; g1=model.schedule(x.new_ones(1))[0]
    a1,s1=model.schedule.coefficients(g1)
    prior=.5*(a1.square()*x.square().mean()/fraction+s1.square()-1-torch.log(s1.square()))/math.log(2)
    decoder=(.5*math.log(2*math.pi*decoder_std**2)+.5*torch.exp(g0)/decoder_std**2)/math.log(2)
    return diffusion+prior+decoder,dict(diffusion=diffusion,prior=prior,decoder=decoder,
                                      independent_dimensions=x[0].numel()*fraction)


@torch.no_grad()
def sample(model,condition,kind,steps,seeds,noise_grid=1024):
    """Technical timing sampler; VDM ancestral or CFM Heun, no clipping.

    A later scientific sampler protocol must freeze its own convergence and
    cross-arm common-random-number controls. This function makes no such claim.
    """
    if (kind not in ('vdm','cfm') or isinstance(steps,bool) or not isinstance(steps,int) or steps<1
            or (kind=='vdm' and noise_grid%steps)):
        raise ValueError('invalid objective/step/noise-grid request')
    if len(seeds)!=len(condition.joint) or len(set(seeds))!=len(seeds):
        raise ValueError('one distinct addressed seed per draw required')
    before=model.training; model.eval()
    projector=project if model.stage=='fine' else lambda v:v
    generators=[torch.Generator(device=condition.joint.device).manual_seed(int(seed)) for seed in seeds]
    def noise():
        return projector(torch.cat([torch.randn((1,1,*SHAPES[condition.region]),
            device=condition.joint.device,dtype=condition.joint.dtype,generator=g) for g in generators]))
    def velocity(z,t):
        return model(z,model.schedule(z.new_full((len(z),),t)),condition)
    try:
        z=noise(); condition.validate(z,model.stage,model.domain,False)
        for i in range(steps):
            if kind=='cfm':
                t=i/steps; dt=1/steps
                first=velocity(z,t); proposal=projector(z+dt*first)
                z=projector(z+.5*dt*(first+velocity(proposal,(i+1)/steps)))
            else:
                gt=model.schedule(z.new_full((len(z),),1-i/steps))
                gs=model.schedule(z.new_full((len(z),),1-(i+1)/steps))
                at,st=model.schedule.coefficients(gt); ass,ss=model.schedule.coefficients(gs)
                coefficient=-torch.expm1(gs-gt); b=lambda v:v[:,None,None,None,None]
                mean=b(ass/at)*(z-b(coefficient*st)*model(z,gt,condition))
                increment=noise(); factor=noise_grid//steps
                for _ in range(factor-1): increment=increment+noise()
                z=projector(mean+b(ss*coefficient.sqrt())*increment/math.sqrt(factor))
            if not torch.isfinite(z).all(): raise FloatingPointError('nonfinite technical sampling path')
        return z
    finally:
        model.train(before)
