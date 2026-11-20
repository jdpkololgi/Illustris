"""Isolated grid challenger entrypoint; leaves the running P12-A trainer unchanged.

Usage: python -m workflows.abacus_tweb.p12a_encoder_sweep ARCH [Arm A flags]
All arms use UPatch's point head, normalization, core sampler and validation.
These are controlled architectural ablations, not pretrained paper replicas.
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from workflows.abacus_tweb import p8_train_unet_patch as grid


class LargeKernel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(c, c, 5, padding=2, groups=c),
                                   grid.ChannelLayerNorm3d(c), nn.Conv3d(c, 2*c, 1),
                                   nn.GELU(), nn.Conv3d(2*c, c, 1))

    def forward(self, x):
        return x + self.block(x)


class Window(nn.Module):
    """Non-wrapping shifted windows at the coarsest scale; mask padded keys."""
    def __init__(self, c, shift):
        super().__init__()
        self.shift = shift
        self.pos = nn.Conv3d(c, c, 3, padding=1, groups=c)
        self.layer = nn.TransformerEncoderLayer(c, 4, 2*c, dropout=0.,
                                                activation="gelu", batch_first=True,
                                                norm_first=True)

    def forward(self, x):
        x = x + self.pos(x)
        b,c,d,h,w = x.shape
        s = self.shift
        pad = tuple(t for v in (w,h,d) for t in (s, (-(v+s)) % 4))
        y = F.pad(x, pad)
        valid = F.pad(torch.ones_like(x[:, :1], dtype=torch.bool), pad)
        a,e,f = (v//4 for v in y.shape[2:])
        def pack(t):
            return t.reshape(b,-1,a,4,e,4,f,4).permute(0,2,4,6,3,5,7,1).reshape(b*a*e*f,64,-1)
        z = self.layer(pack(y), src_key_padding_mask=~pack(valid).squeeze(-1))
        z = z.reshape(b,a,e,f,4,4,4,c).permute(0,7,1,4,2,5,3,6).reshape_as(y)
        return z[:,:,s:s+d,s:s+h,s:s+w]


Base = grid.UPatch


def build(arch, **kwargs):
    model = Base(**kwargs)
    if arch == "reference":
        return model
    b = kwargs.get("base",24)
    if arch == "large_kernel":
        for name,c in (("enc0",b),("enc1",2*b),("enc2",4*b)):
            setattr(model.unet,name,nn.Sequential(getattr(model.unet,name),LargeKernel(c)))
    elif arch == "window_attention":
        model.unet.bottleneck = nn.Sequential(Window(4*b,0), Window(4*b,2))
    else:
        raise ValueError(arch)
    return model


def main():
    from workflows.abacus_tweb import p10_train_arm_a as trainer
    arch = sys.argv.pop(1)
    if arch not in ("reference","large_kernel","window_attention"):
        raise SystemExit("unknown grid challenger")
    args = trainer.parse_args()
    if (args.model != "unet" or args.p11_dense_view or not args.run_name.startswith("sweep_"+arch+"_")
            or args.contract_root.resolve() != Path("/pscratch/sd/d/dkololgi/abacus/p12a_lowz_20261029_v1/training_contract_v1")):
        raise SystemExit("require registered low-z U-Net contract and architecture-specific run name")
    # Process-local substitution only: pin both entrypoint and architecture in restart checks.
    original = trainer.source_contract
    trainer.source_contract = lambda: {**original(), "workflows/abacus_tweb/p12a_encoder_sweep.py":
                                       trainer.sha256(Path(__file__))}
    trainer.unet_impl.UPatch = lambda **kw: build(arch, **kw)
    trainer.main()


if __name__ == "__main__":
    main()
