"""Read-only, full-field gradient diagnostics on registered fitting fields."""
import math
import torch
from workflows.sbi.e2e_multinoise_models import coefficients
from workflows.sbi.e2e_preservation_loss import residual_terms


def components(model,item,noise,aux,sigma):
    y=item['target'];t=y.new_tensor([2*math.atan(sigma)/math.pi])
    a,b,_=coefficients(t,model.tau)
    predict=lambda x:model(x,t,item['condition'],wide_condition=item['wide'])
    noisy=predict(a*y+b*noise)
    v0=predict(a*y);vp=predict(a*y+b*.25*aux);vm=predict(a*y-b*.25*aux)
    return {'denoising':(noisy-(a*noise-b*y)).square().mean(),
            **residual_terms(v0,vp,vm,y,aux,a,b,.25)}


def gradient_vectors(losses,parameters):
    parameters=list(parameters);out={}
    for i,(name,loss) in enumerate(losses.items()):
        grads=torch.autograd.grad(loss,parameters,retain_graph=i<len(losses)-1,allow_unused=True)
        out[name]=torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                              for p,g in zip(parameters,grads)]).double()
    return out


def comparison(a,b):
    aa=float(a.square().sum());bb=float(b.square().sum());dot=float((a*b).sum())
    return dict(norm_a=math.sqrt(aa),norm_b=math.sqrt(bb),dot=dot,
                cosine=dot/math.sqrt(aa*bb) if aa*bb>0 else None)


def remove_conflicting_component(primary,auxiliary):
    """One-sided Euclidean projection diagnostic; NOT an Adam safety guarantee."""
    dot=(primary*auxiliary).sum();square=primary.square().sum()
    if float(square)==0:
        return auxiliary.clone()
    return auxiliary-torch.minimum(dot,torch.zeros_like(dot))/square*primary
