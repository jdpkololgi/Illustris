"""Original 3-D backbone with identity-initialized noise/capacity adapters.

EDM equations: Karras et al. 2022, arXiv:2206.00364, Table 1.
Independent implementation of equations; no upstream implementation copied.
Our normalized targets have sigma_data=1. EDM F=-VP-v is an exact change of
coordinates, not a new objective. The changed sampling/adapters are explicit.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from workflows.sbi.e2e_wide_models import _downsample_mean, _validate_pair


def edm_coefficients(sigma, sigma_data=1.):
    variance = sigma.square() + sigma_data**2
    return sigma_data**2/variance, sigma*sigma_data/variance.sqrt(), variance.rsqrt()


class NoiseAdaptedNet(nn.Module):
    def __init__(self, base, film=False, receptive=False):
        super().__init__()
        self.base = base
        widths = [block.layers[0].out_channels for block in base.encoders]
        self.widths = widths + widths[-2::-1]
        self.film = None
        if film:
            self.register_buffer('frequencies', 2.**torch.arange(8))
            self.film = nn.Sequential(nn.Linear(17,32),nn.SiLU(),nn.Linear(32,2*sum(self.widths)))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        self.receptive = None
        if receptive:
            w=widths[-1]
            self.receptive = nn.Sequential(nn.Conv3d(w,w,3,padding=2,dilation=2),nn.GroupNorm(1,w),nn.SiLU(),
                                           nn.Conv3d(w,w,3,padding=4,dilation=4),nn.GroupNorm(1,w),nn.SiLU(),
                                           nn.Conv3d(w,w,1))
            nn.init.zeros_(self.receptive[-1].weight); nn.init.zeros_(self.receptive[-1].bias)

    def forward(self,state,time,condition,wide_condition=None):
        if self.film is None and self.receptive is None:
            return self.base(state,time,condition,wide_condition=wide_condition)
        _validate_pair(state,condition)
        if time.shape!=(state.shape[0],) or condition.shape[1]!=self.base.condition_channels:
            raise ValueError('invalid time/condition')
        # Explicit finite endpoint proxy for log-noise features; VP state and
        # original time channels are NOT clamped. Double avoids float32 tan(pi/2).
        sigma=torch.tan(time.double()*math.pi/2).clamp(.002,80).to(state)
        code=torch.log(sigma)/4
        if self.film is not None:
            angles=code[:,None]*self.frequencies[None,:]
            parameters=self.film(torch.cat((code[:,None],angles.sin(),angles.cos()),dim=1))
            chunks=parameters.split([2*w for w in self.widths],dim=1)
        def adapt(x,index):
            if self.film is None:
                return x
            scale,bias=chunks[index].chunk(2,dim=1)
            return x*(1+scale[:,:,None,None,None])+bias[:,:,None,None,None]
        t=time.to(state).view(-1,1,1,1,1)
        features=torch.cat((t,torch.sin(math.pi*t),torch.cos(math.pi*t)),dim=1)
        x=torch.cat((state,condition,features.expand(-1,-1,*state.shape[2:])),dim=1)
        skips=[]
        for i,block in enumerate(self.base.encoders):
            if i:
                x=_downsample_mean(x)
            x=adapt(block(x),i);skips.append(x)
        if self.base.wide_encoder is not None:
            x=x+self.base.wide_encoder(wide_condition)[:,:,None,None,None]
        if self.receptive is not None:
            x=x+self.receptive(x)
        for i,(block,skip) in enumerate(zip(self.base.decoders,reversed(skips[:-1]))):
            x=F.interpolate(x,size=skip.shape[2:],mode='nearest')
            x=adapt(block(torch.cat((x,skip),dim=1)),len(skips)+i)
        return self.base.output(x)

    def denoise(self,noisy,sigma,condition,wide_condition=None):
        sigma=torch.as_tensor(sigma,device=noisy.device,dtype=noisy.dtype)
        if sigma.ndim==0:
            sigma=sigma.expand(noisy.shape[0])
        if sigma.shape!=(noisy.shape[0],) or bool((sigma<=0).any()):
            raise ValueError('positive sigma per batch required')
        skip,out,scale=edm_coefficients(sigma)
        expand=lambda v:v[:,None,None,None,None]
        t=2*torch.atan(sigma)/math.pi
        raw=-self(expand(scale)*noisy,t,condition,wide_condition=wide_condition)
        return expand(skip)*noisy+expand(out)*raw


def edm_loss(model,target,noise,sigma,condition,wide):
    """Weighted D-MSE evaluated as stable unit-variance F-target MSE.

lambda=1/c_out^2. For sigma_data=1 the target is -VP-v and no new weighting
relative to existing v-MSE is claimed. Avoid subtracting nearly equal fields.
"""
    sigma=float(sigma)
    if not sigma>0 or not math.isfinite(sigma):
        raise ValueError('positive finite noise required')
    alpha=1/math.sqrt(1+sigma*sigma)
    t=2*math.atan(sigma)/math.pi
    state=alpha*(target+sigma*noise)
    raw=-model(state,target.new_tensor([t]),condition,wide_condition=wide)
    target_f=alpha*(sigma*target-noise)
    return torch.mean((raw-target_f)**2)
