"""Training-only compact harmonic response fits and fixed mock scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_boundary_mock as experiment
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_footprint_tidal_operator as op


def harmonic_basis(n=16, degree=3):
    """Hessians of all real harmonic quadratics/cubics: 5 + 7 modes.

    Coordinates are in core-width units. Column normalization is fixed by the
    complete geometric core, never learned from evaluation matter.
    """
    polynomials = [
        {(2,0,0):1,(0,2,0):-1}, {(0,0,2):2,(2,0,0):-1,(0,2,0):-1},
        {(1,1,0):1}, {(1,0,1):1}, {(0,1,1):1}]
    if degree==3:
        polynomials += [
            {(3,0,0):1,(1,2,0):-3}, {(0,3,0):1,(2,1,0):-3},
            {(0,0,3):1,(2,0,1):-1.5,(0,2,1):-1.5},
            {(1,0,2):4,(3,0,0):-1,(1,2,0):-1},
            {(0,1,2):4,(2,1,0):-1,(0,3,0):-1},
            {(2,0,1):1,(0,2,1):-1}, {(1,1,1):1}]
    elif degree!=2:
        raise ValueError('Only harmonic degrees 2 and 3 are registered')
    xyz=np.stack(np.meshgrid(*[((np.arange(n)+.5)/n-.5)]*3,indexing='ij'),-1).reshape(-1,3)
    basis=np.zeros((n**3,6,len(polynomials)))
    for k, polynomial in enumerate(polynomials):
        for j,(a,b) in enumerate(op.COMPONENTS):
            for exponents,coefficient in polynomial.items():
                powers=np.array(exponents)
                coefficient*=powers[a]; powers[a]-=1
                coefficient*=powers[b]; powers[b]-=1
                if coefficient==0 or np.any(powers<0): continue
                basis[:,j,k]+=coefficient*np.prod(xyz**powers,axis=-1)
    norm=np.sqrt(np.mean(np.sum(basis*basis*experiment.WEIGHTS[None,:,None],axis=1),axis=0))
    return basis/norm


def coefficients(residual, basis, valid):
    if valid.sum()<basis.shape[-1]:
        raise ValueError('Insufficient fully supported cells for compact oracle')
    w=np.sqrt(experiment.WEIGHTS)
    design=(basis[valid]*w[None,:,None]).reshape(-1,basis.shape[-1])
    target=(residual[valid]*w).ravel()
    result,_,rank,_=np.linalg.lstsq(design,target,rcond=None)
    if rank!=basis.shape[-1]: raise ValueError('Rank-deficient harmonic oracle')
    return result


def features(values, identity, valid):
    t=values['isolated'].reshape(-1,6)[valid]
    density=values['density'].ravel()[valid]
    return np.r_[t.mean(0),density.mean(),density.std(),values['support'].mean(),
        values['distance'].ravel()[valid].mean()/100,np.array(identity['center_mpc_h'])/1000]


def ridge(x,y):
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-12]=1
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    penalty=np.eye(design.shape[1]); penalty[0,0]=0
    beta=np.linalg.solve(design.T@design+penalty,design.T@y)
    return dict(mean=mean,scale=scale,beta=beta)


def predict(model,x):
    return np.column_stack([np.ones(len(x)),(x-model['mean'])/model['scale']])@model['beta']


def fit(x,y,phases):
    if set(phases)!=set(experiment.TRAIN):
        raise PermissionError('Conditional fit is training-only, with both registered phases')
    model=ridge(x,y); residual=np.empty_like(y)
    for phase in experiment.TRAIN:
        test=np.array(phases)==phase
        residual[test]=y[test]-predict(ridge(x[~test],y[~test]),x[test])
    # Include CV residual bias in second moment: do not silently subtract it.
    covariance=residual.T@residual/len(residual)
    eigen,vectors=np.linalg.eigh(covariance)
    if eigen.min() < -1e-10: raise ValueError('Non-PSD residual covariance')
    model.update(covariance=covariance,sqrt=vectors@np.diag(np.sqrt(np.maximum(eigen,0))),
                 cv_mean=residual.mean(0),training_rows=len(x))
    return model


def metrics(t,truth,valid):
    if not valid.any(): return None
    t=t[valid];truth=truth[valid]
    e=op.eigensystem(t);te=op.eigensystem(truth)
    label=(e>.2).sum(-1);actual=(te>.2).sum(-1)
    confusion=np.bincount(4*actual+label,minlength=16).reshape(4,4)
    return dict(cells=int(valid.sum()),tensor_mse=float(np.mean(np.sum((t-truth)**2*experiment.WEIGHTS,axis=-1))),
        eigen_mse=np.mean((e-te)**2,axis=0).tolist(),class_flip=float(np.mean(label!=actual)),
        confusion=confusion.tolist())


def distribution_metrics(tensors,truth,valid):
    e=op.eigensystem(tensors[:,valid]); te=op.eigensystem(truth[valid])
    # M=64: order indices3,60 cover57/65 under exchangeability, not exactly90%.
    ordered=np.sort(e,axis=0);covered=(te>=ordered[3])&(te<=ordered[60])
    labels=(e>.2).sum(-1); actual=(te>.2).sum(-1)
    p=np.stack([(labels==k).mean(0) for k in range(4)],-1)
    actual_onehot=np.eye(4)[actual]
    return dict(central90_coverage=covered.mean(0).tolist(),attainable_reference=57/65,
        mean_interval_width=(ordered[60]-ordered[3]).mean(0).tolist(),
        brier=float(np.mean(np.sum((p-actual_onehot)**2,axis=-1))),
        mean_class_probability=p.mean(0).tolist())


def load(root,phase,factor=2,suffix=''):
    experiment.guard(phase)
    complete=json.loads((root/(phase+suffix)/'COMPLETE.json').read_text())
    if not complete['passed'] or complete['phase']!=phase or complete['factor']!=factor:
        raise ValueError('Unfinished/mismatched phase')
    for cap in ('NGC','SGC'):
        receipt=json.loads((root/(phase+suffix)/(cap+'_COMPLETE.json')).read_text())
        path=Path(receipt['output']['path'])
        if c.sha256(path)!=receipt['output']['sha256']: raise ValueError('Core payload changed')
        with np.load(path) as f:
            arrays={k:f[k] for k in f.files}
        for i,identity in enumerate(receipt['identities']):
            yield identity,{k:v[i] for k,v in arrays.items()}


def summarize(rows):
    names=rows[0]['arms']; result={}
    for name in names:
        vals=[r['arms'][name]['primary'] for r in rows]
        vals=[v for v in vals if v is not None]
        result[name]=dict(cores=len(vals),tensor_rmse=float(np.sqrt(np.mean([v['tensor_mse'] for v in vals]))),
            eigen_rmse=np.sqrt(np.mean([v['eigen_mse'] for v in vals],axis=0)).tolist(),
            class_flip=float(np.mean([v['class_flip'] for v in vals])))
        distributions=[r['arms'][name]['distribution'] for r in rows if 'distribution' in r['arms'][name]]
        if distributions:
            result[name].update(central90_coverage=np.mean([v['central90_coverage'] for v in distributions],0).tolist(),
                brier=float(np.mean([v['brier'] for v in distributions])))
    return result


def run(root,out,factor=2,suffix=''):
    c.require_compute(); out.mkdir(parents=True,exist_ok=False)
    if factor not in (1,2) or (factor==1 and suffix!='_refined_v2') or (factor==2 and suffix):
        raise ValueError('Unregistered source-resolution panel')
    bases={2:harmonic_basis(degree=2),3:harmonic_basis(degree=3)}
    x=[]; ys={2:[],3:[]}; phases=[]; training_ids=[]
    for phase in experiment.TRAIN:
        for identity,v in load(root,phase,factor,suffix):
            valid=v['support'].ravel()==1
            x.append(features(v,identity,valid)); phases.append(phase);training_ids.append(identity)
            residual=(v['truth']-v['isolated']).reshape(-1,6)
            for degree,basis in bases.items(): ys[degree].append(coefficients(residual,basis,valid))
    x=np.stack(x); models={degree:fit(x,np.stack(ys[degree]),phases) for degree in bases}
    np.savez_compressed(out/'MODELS.npz',**{f'd{d}_{k}':v for d,model in models.items() for k,v in model.items()},
        features=x,constant_coefficients=np.stack(ys[2]),linear_coefficients=np.stack(ys[3]))
    c.atomic_json(out/'FIT.json',dict(training_identities=training_ids,evaluation_used_for_fit=False,
        conditional_on='true observed density summaries, not galaxies',ridge_penalty=1.,draws=64,seed=271))
    rng=np.random.default_rng(271); rows=[]
    for phase in experiment.EVAL:
        for identity,v in load(root,phase,factor,suffix):
            valid=v['support'].ravel()==1; secondary=v['support'].ravel()>=.5
            truth=v['truth'].reshape(-1,6); observed=v['isolated'].reshape(-1,6)
            pred={name:v[name].reshape(-1,6) for name in ('isolated','periodic','periodic_dc')}
            row=dict(**identity,arms={},fully_supported_cells=int(valid.sum()))
            for degree,basis in bases.items():
                oracle=coefficients(truth-observed,basis,valid)
                pred[f'oracle_d{degree}']=observed+np.einsum('nck,k->nc',basis,oracle)
                mean=predict(models[degree],features(v,identity,valid)[None])[0]
                pred[f'conditional_d{degree}']=observed+np.einsum('nck,k->nc',basis,mean)
            for name,tensor in pred.items():
                distance=v['distance'].ravel()
                row['arms'][name]=dict(primary=metrics(tensor,truth,valid),
                    secondary=metrics(tensor,truth,secondary),
                    distance_bins=[metrics(tensor,truth,valid&(distance>=lo)&(distance<hi))
                        for lo,hi in zip((0,13.532,27.064,54.128,108.256),(13.532,27.064,54.128,108.256,10000))])
            for degree,basis in bases.items():
                mean=predict(models[degree],features(v,identity,valid)[None])[0]
                draws=mean+rng.normal(size=(64,len(mean)))@models[degree]['sqrt'].T
                samples=observed[None]+np.einsum('nck,sk->snc',basis,draws)
                row['arms'][f'conditional_d{degree}']['distribution']=distribution_metrics(samples,truth,valid)
            rows.append(row)
            experiment.log(stage='scored',pair=identity['pair_id'],side=identity['side'])
    groups={'all':rows}
    for key in ('phase','cap','shell','stratum'):
        for value in sorted({r[key] for r in rows}):groups[f'{key}:{value}']=[r for r in rows if r[key]==value]
    c.atomic_json(out/'SCORES.json',dict(passed=True,cores=rows,groups={k:summarize(v) for k,v in groups.items()},
        sources=[c.file_record(root/(p+suffix)/'COMPLETE.json',content_hash=True) for p in experiment.TRAIN+experiment.EVAL],
        source_factor=factor,source_suffix=suffix,
        source_sha256=c.sha256(__file__),protocol_sha256=c.sha256(c.REPO/'configs/e2e_boundary_mock_20261002.json'),
        refinement_protocol_sha256=c.sha256(c.REPO/'configs/e2e_boundary_refinement_followup_20261002.json') if factor==1 else None,
        posterior_qualified=False,attainable_coverage_reference=57/65))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--factor',type=int,default=2);p.add_argument('--suffix',default='')
    a=p.parse_args();run(a.root,a.out,a.factor,a.suffix)
