"""Restricted Gaussian-bridge absorption predictions; not a nonlinear-CFM law."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from workflows.sbi.e2e_conditional_reference import atomic_json
from workflows.sbi.e2e_conditional_reference_math import problem
from workflows.sbi.e2e_conditional_reference_continue import digest


def absorbed_variance(lam,e):
    lam=np.asarray(lam,dtype=float);e=np.maximum(np.asarray(e,dtype=float),0)
    t,w=np.polynomial.legendre.leggauss(96);t=(t[:,None]+1)/2;w=w[:,None]/2
    s2=t*t*lam+(1-t)**2;a=(t*lam-(1-t))/s2
    def risk(log_l):
        l=np.exp(log_l);aa=(t*l-(1-t))/(t*t*l+(1-t)**2)
        return np.sum(w*(e*(1-t*aa)**2+(aa-a)**2*s2),axis=0)
    lo=np.log(lam);hi=np.log(lam*101+100*e)
    ratio=(np.sqrt(5)-1)/2
    for _ in range(64):
        x=hi-ratio*(hi-lo);y=lo+ratio*(hi-lo);left=risk(x)<risk(y)
        hi=np.where(left,y,hi);lo=np.where(left,lo,x)
    return np.exp((lo+hi)/2)


def shell_power(values,vectors,radius):
    n=len(radius);f=(vectors*np.sqrt(values)).T.reshape(-1,n,n,n)
    p=np.abs(np.fft.fftn(f,axes=(1,2,3),norm='ortho'))**2
    p=p.sum(0);shell=np.floor(radius).astype(int)
    return np.array([p[shell==i].mean() for i in np.unique(shell)])


def draw_diagnostic(draws,case,radius):
    z=(np.asarray(draws,dtype=float)-case['mu'])@case['vectors']
    raw=z.mean(0)**2;noise=z.var(0,ddof=1)/len(z);corrected=raw-noise
    predicted=absorbed_variance(case['values'],np.maximum(corrected,0))
    reference=shell_power(case['values'],case['vectors'],radius)
    return dict(kind='nonlinear_draw_approximation_not_causal_identity',draws=len(z),
        raw_mean_squared=raw.tolist(),mean_monte_carlo_variance=noise.tolist(),
        corrected_mean_squared=corrected.tolist(),negative_corrected_modes=int((corrected<0).sum()),
        predicted_shell_ratio=(shell_power(predicted,case['vectors'],radius)/reference).tolist(),
        raw_predicted_shell_ratio=(shell_power(absorbed_variance(case['values'],raw),case['vectors'],radius)/reference).tolist())


def affine_diagnostic(state,case,prior_cov,template,radius):
    lam=case['values'];u=case['vectors'];gain=case['sigma']*(case['mask']/case['std']**2)[None,:]
    learned=state['mean_map'][template].cpu().numpy().astype(float)
    delta=u.T@(learned-gain)
    cy=prior_cov*case['mask'][:,None]*case['mask'][None,:]+np.diag(case['mask']*case['std']**2)
    e=np.sum((delta@cy)*delta,axis=1)
    predicted=absorbed_variance(lam,e)
    fitted=state['log_values'][template].cpu().numpy().astype(float)
    fitted=np.exp(fitted)
    reference=shell_power(lam,u,radius)
    return dict(mean_error_per_mode=e.tolist(),learned_relative_inflation=(fitted/lam-1).tolist(),
        predicted_relative_inflation=(predicted/lam-1).tolist(),
        relative_inflation_correlation=float(np.corrcoef(fitted/lam-1,predicted/lam-1)[0,1]),
        absolute_variance_correlation=float(np.corrcoef(fitted,predicted)[0,1]),
        learned_shell_ratio=(shell_power(fitted,u,radius)/reference).tolist(),
        predicted_shell_ratio=(shell_power(predicted,u,radius)/reference).tolist())


def main(a):
    prior,_,_,_,cases,_,radius=problem(8,4)
    root=Path(a.affine);m=json.loads((root/'manifest.json').read_text());rows=[];inputs={}
    for item in m['items']:
        checkpoint=root/'results'/item['name']/'checkpoint_65536.pt'
        inputs[str(checkpoint)]=digest(checkpoint)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
        for which in [0,1]:
            r=affine_diagnostic(state,cases[which],prior,which,radius)
            rows.append(item|dict(template=which,**r))
            print(item['name'],which,'learned/predicted',r['learned_shell_ratio'][-1],r['predicted_shell_ratio'][-1],
                  'inflation correlation',r['relative_inflation_correlation'],flush=True)
    atomic_json(a.output,dict(affine=rows,inputs=inputs,source_sha256=digest(Path(__file__)),claim='restricted-family equilibrium prediction; no universal CFM claim'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--affine',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
