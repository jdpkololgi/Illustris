"""Post-run absorption diagnostic in each arm's actual white-base coordinates.

The frozen worker's physical-white-base prediction is a common counterfactual.
For white_bridge it is NOT the trained bridge; this receipt supersedes it.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from workflows.sbi.e2e_conditional_reference import atomic_json
from workflows.sbi.e2e_conditional_reference_math import problem
from workflows.sbi.e2e_conditional_reference_continue import digest
from workflows.sbi.e2e_spectral_absorption import draw_diagnostic


def transformed_case(case,scale):
    n=scale.shape[0];d=n**3
    def apply(x,s):
        return np.fft.ifftn(np.fft.fftn(x.reshape(-1,n,n,n),axes=(-3,-2,-1))*s,
                           axes=(-3,-2,-1)).real.reshape(-1,d)
    w=apply(np.eye(d),scale);winv=apply(np.eye(d),1/scale)
    values,vectors=np.linalg.eigh(w@case['sigma']@w.T)
    return dict(mu=w@case['mu'],values=values,vectors=vectors),winv@vectors,w


def main(a):
    root=Path(a.output);manifest=json.loads((root/'manifest.json').read_text())
    _,_,_,_,cases,_,radius=problem(8,4)
    power=np.load(root/'train_prior_power.npy');scale=1/np.sqrt(np.maximum(power,power.max()*1e-6))
    transformed=[transformed_case(c,scale) for c in cases]
    rows=[]
    for item in manifest['items']:
        for index in (range(4) if item['fixed'] is None else [0]):
            path=root/'results'/item['name']/'precision'/f'draws_32768_{index}_256.npy'
            draws=np.load(path)
            if item['arm']=='white_bridge':
                case,basis,w=transformed[index]
                result=draw_diagnostic(draws@w.T,case,radius,basis)
            else:result=draw_diagnostic(draws,cases[index],radius)
            rows.append(dict(item=item,case=index,draw_sha256=digest(path),**result))
    atomic_json(root/'BRIDGE_ABSORPTION.json',dict(rows=rows,
        diagnostic_source_sha256=digest(Path(__file__)),
        absorption_source_sha256=digest(Path(__file__).with_name('e2e_spectral_absorption.py')),
        power_sha256=digest(root/'train_prior_power.npy'),
        caveat='Restricted affine approximation in matching base coordinates, not nonlinear causal attribution'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args())
