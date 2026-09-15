"""Fixed-noise capability controls; never a full diffusion sampler interface."""
import math
import numpy as np
from scipy.fft import dctn, idctn
import torch
from torch import nn


def vp_coefficients(time):
    a = torch.cos(time * math.pi/2)[:, None, None, None, None]
    b = torch.sin(time * math.pi/2)[:, None, None, None, None]
    if bool(((a <= 0) | (b <= 0)).any()):
        raise ValueError('strictly interior VP times required; not a sampler')
    return a, b


class ResidualAdapter(nn.Module):
    """Explicit clean skip D=z+r, returned in the audit's v chart.

    Predict the correction in clean-field units, not epsilon amplified by 1/s.
    """
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, state, time, condition, wide_condition=None):
        a, b = vp_coefficients(time)
        correction = self.net(state, time, condition, wide_condition=wide_condition)
        return -b*state/a-correction/b


class HighResolutionNet(nn.Module):
    """Full-resolution residual blocks, no activation normalization/downsampling.

    This is a combined architecture contrast, not an isolated GroupNorm test.
    Both local and wide conditions are retained. Zero output starts D at z.
    """
    def __init__(self, base, width=24, blocks=4):
        super().__init__()
        self.stem = nn.Conv3d(1+base.condition_channels, width, 3, padding=1)
        self.wide_encoder = base.wide_encoder
        wide_width = self.wide_encoder[-1].out_features
        self.wide_projection = nn.Linear(wide_width, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv3d(width,width,3,padding=1),nn.SiLU(),
                                                   nn.Conv3d(width,width,3,padding=1)) for _ in range(blocks)])
        self.output = nn.Conv3d(width, 1, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, state, time, condition, wide_condition=None):
        x = self.stem(torch.cat((state,condition),dim=1))
        x = x+self.wide_projection(self.wide_encoder(wide_condition))[:,:,None,None,None]
        for block in self.blocks:
            x = x+.1*block(x)
        return self.output(x)


class LinearReference(nn.Module):
    """Predeclared even-boundary DCT low-pass, or noisy identity. No truth fitting."""
    def __init__(self, n=96, cell=3.383, low=.32, high=.40, identity=False):
        super().__init__(); self.identity = identity
        if not 0 < low < high:
            raise ValueError('ordered positive cutoff frequencies required')
        k = np.arange(n)*math.pi/(n*cell)
        radius = np.sqrt(k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2)
        taper = np.clip((radius-low)/(high-low),0,1)
        self.response = .5*(1+np.cos(math.pi*taper))

    def clean(self, noisy):
        if self.identity:
            return noisy
        if noisy.shape[-3:] != self.response.shape:
            raise ValueError('reference grid mismatch')
        values = noisy.detach().cpu().numpy()
        filtered = idctn(dctn(values, axes=(-3,-2,-1), norm='ortho')*self.response,
                         axes=(-3,-2,-1), norm='ortho')
        return torch.as_tensor(filtered,device=noisy.device,dtype=noisy.dtype)

    def forward(self, state, time, condition, wide_condition=None):
        a,b = vp_coefficients(time)
        return (a*state-self.clean(state/a))/b


def fixed_loss(model, target, noise, ratio, condition, wide, residual=False):
    a = 1/math.sqrt(1+ratio**2); b = ratio*a
    state = a*target+b*noise
    t = target.new_full((target.shape[0],),2*math.atan(ratio)/math.pi)
    if residual:
        prediction = model.net(state,t,condition,wide_condition=wide)
        expected = -ratio*noise
        return torch.mean(((prediction-expected)/ratio)**2)
    else:
        prediction = model(state,t,condition,wide_condition=wide)
        expected = a*noise-b*target
    return torch.mean((prediction-expected)**2)
