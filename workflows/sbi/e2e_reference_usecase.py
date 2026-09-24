"""Independent development-only propagation of errors into toy use-case summaries.

R is in grid cells, not Mpc/h. Tidal DC is isotropic (delta_mean/3).
Exact Gaussian linear marginals and independent oracle ensembles are controls.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from workflows.sbi.e2e_conditional_reference import atomic_json
from workflows.sbi.e2e_conditional_reference_math import problem
from workflows.sbi.e2e_conditional_reference_continue import digest
from workflows.sbi.e2e_partial_whitening import context


def smooth(x,r,n=8):
    k=np.fft.fftfreq(n)*2*np.pi
    k2=sum(a*a for a in np.meshgrid(k,k,k,indexing='ij'))
    return np.fft.ifftn(np.fft.fftn(x.reshape(-1,n,n,n),axes=(1,2,3))*np.exp(-.5*r*r*k2),axes=(1,2,3)).real.reshape(-1,n**3)


def tidal(x,r,n=8):
    y=smooth(x,r,n).reshape(-1,n,n,n);freq=np.fft.fftn(y,axes=(1,2,3))
    k=np.meshgrid(np.fft.fftfreq(n)*2*np.pi,np.fft.fftfreq(n)*2*np.pi,np.fft.fftfreq(n)*2*np.pi,indexing='ij')
    k2=sum(a*a for a in k);den=np.where(k2>0,k2,1)
    tensor=np.empty(y.shape+(3,3))
    for i in range(3):
        for j in range(i,3):
            factor=k[i]*k[j]/den;factor[0,0,0]=1/3 if i==j else 0
            tensor[...,i,j]=np.fft.ifftn(freq*factor,axes=(1,2,3)).real
            tensor[...,j,i]=tensor[...,i,j]
    return tensor


def class_prob(x,r):
    counts=np.zeros((512,4))
    for start in range(0,len(x),64):
        eigen=np.linalg.eigvalsh(tidal(x[start:start+64],r)).reshape(-1,512,3)
        labels=(eigen>0).sum(-1)
        for label in range(4):counts[:,label]+=(labels==label).sum(0)
    return counts/len(x)


def regions(n=8):
    result={}
    for width in [2,4,8]:
        masks=[]
        for i in range(0,n,width):
            for j in range(0,n,width):
                for k in range(0,n,width):
                    mask=np.zeros((n,n,n));mask[i:i+width,j:j+width,k:k+width]=1/width**3;masks.append(mask.ravel())
        result[width]=np.stack(masks,axis=1)
    return result


def linear_summary(draws,case,q):
    refmean=case['mu']@q;refvar=np.sum(q*(case['sigma']@q),axis=0)
    y=draws@q;sd=np.sqrt(refvar);lo,hi=np.quantile(y,[.05,.95],axis=0)
    coverage=ndtr((hi-refmean)/sd)-ndtr((lo-refmean)/sd)
    ratio=y.var(0,ddof=1)/refvar
    return dict(mean_rms_sd=float(np.sqrt(np.mean(((y.mean(0)-refmean)/sd)**2))),
        variance_ratio_mean=float(ratio.mean()),variance_ratio_min=float(ratio.min()),variance_ratio_max=float(ratio.max()),
        exact_reference_interval_coverage_mean=float(coverage.mean()),
        exact_reference_interval_coverage_max_error=float(max(abs(coverage-.9))))


def distort(draws,case,radius,mode):
    residual=draws-case['mu']
    if mode=='top13':
        shell=np.floor(radius).astype(int);factor=np.where(shell==shell.max(),np.sqrt(1.13),1)
        residual=np.fft.ifftn(np.fft.fftn(residual.reshape(-1,8,8,8),axes=(1,2,3))*factor,axes=(1,2,3)).real.reshape(-1,512)
    elif mode=='all13':residual=residual*np.sqrt(1.13)
    elif mode!='identity':raise ValueError(mode)
    return case['mu']+residual


def run(a):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('compute allocation required')
    root=Path(a.output);m=context(root);parent=Path(m['parent']);pm=context(parent)
    _,_,_,_,cases,_,radius=problem(8,4);rows=[];qregion=regions()
    for index,case in enumerate(cases):
        rng=np.random.default_rng(97001+index)
        oracle=rng.normal(size=(2048,512))@case['chol'].T+case['mu']
        reference=rng.normal(size=(2048,512))@case['chol'].T+case['mu']
        arms=[('oracle_null',oracle),('top13',distort(oracle,case,radius,'top13')),
              ('all13',distort(oracle,case,radius,'all13')),
              ('erased_covariance',rng.normal(size=(2048,512))*np.sqrt(np.diag(case['sigma']))+case['mu'])]
        draw_hashes={}
        for item in pm['items']:
            if item['lr_mode']!='decay':continue
            path=parent/'results'/item['name']/'ema'/'checkpoint_98304'/f'draws_98304_{index}_256.npy'
            draws=np.load(path)
            if draws.shape!=(2048,512) or not np.isfinite(draws).all():raise ValueError('invalid saved ensemble')
            arms.append((item['name'],draws));draw_hashes[str(path)]=digest(path)
        region_rows={name:{str(w):linear_summary(x,case,q) for w,q in qregion.items()} for name,x in arms}
        for r in [0.,.5,1.,2.]:
            q=smooth(np.eye(512),r).T
            pref=class_prob(reference,r);pnull=class_prob(oracle,r)
            null_rms=float(np.sqrt(np.mean((pnull-pref)**2)))
            for name,x in arms:
                p=class_prob(x,r);diff=p-pref
                rows.append(dict(case=index,arm=name,r_cells=r,smoothed=linear_summary(x,case,q),
                    regional=region_rows[name],class_probability_rms=float(np.sqrt(np.mean(diff**2))),
                    class_probability_max_abs=float(np.max(abs(diff))),oracle_null_rms=null_rms,
                    paired_control_probability_rms=float(np.sqrt(np.mean((p-pnull)**2))) if name in ['top13','all13','oracle_null'] else None))
        atomic_json(root/f'USECASE_case{index}.json',dict(rows=[v for v in rows if v['case']==index],input_hashes=draw_hashes))
    atomic_json(root/'USECASE_COMPLETE.json',dict(rows=rows,sources=m['sources'],scope='development-only dimensionless sensitivity, not DESI validation',
        controls='same oracle residuals with top-shell-only or all-mode variance multiplied by 1.13; separate diagonal-covariance negative control',
        tidal_convention='eigenvalues of smoothed inverse-Laplacian Hessian; isotropic DC, threshold zero; classes=count of positive eigenvalues',
        reference='exact Gaussian linear marginals; independent 2048-draw oracle for nonlinear class probabilities'))
    lines=['# Toy downstream sensitivity (not a DESI requirement validation)','',
        'R is in cells, with no assigned physical box size. Classes use threshold zero.',
        'Linear coverage integrates the exact Gaussian reference; nonlinear class probabilities have finite Monte Carlo error.',
        '', '| Arm | R/cell | Smoothed variance ratio | Class probability RMS error | Oracle-null RMS |',
        '|---|---:|---:|---:|---:|']
    for name,_ in arms:
        for r in [0.,.5,1.,2.]:
            rr=[v for v in rows if v['arm']==name and v['r_cells']==r]
            lines.append(f'| {name} | {r} | {np.mean([v["smoothed"]["variance_ratio_mean"] for v in rr]):.5f} | {np.mean([v["class_probability_rms"] for v in rr]):.5f} | {np.mean([v["oracle_null_rms"] for v in rr]):.5f} |')
    lines+=['','All original unsmoothed gates remain unchanged. This assay cannot certify 7 Mpc/h or DESI T-Web accuracy.']
    (root/'USECASE_SUMMARY.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args())
