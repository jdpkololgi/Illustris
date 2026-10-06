"""Cheap target-specific, conditional-MC mixture feasibility screen; no new draws."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np

BASE = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
OLD = BASE/'coherent_exterior_20261002_v1'
RUN = BASE/'cfm_model_mixture_20261006_v2'
CAL = BASE/'calibration_investigation_20261006_v1'
OUT = RUN/'futility_v1'
PHASES = ('ph012', 'ph013')
LIMITS = dict(mean_shift_sd=.10, covariance_change=.10,
              between_covariance=.05, energy_change_abs=.02)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(16 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def verify(rec):
    if sha(rec['path']) != rec['sha256']:
        raise ValueError('Source binding changed: '+rec['path'])


def covariance(x):
    z = x-x.mean(axis=1, keepdims=True)
    return np.einsum('cmi,cmj->cij', z, z)/(x.shape[1]-1)


def metric(full):
    s = covariance(full); dim = s.shape[-1]
    scale = np.trace(s, axis1=-2, axis2=-1)/dim
    if np.any(scale <= 1e-14):
        raise ValueError('Degenerate baseline covariance')
    diagonal = np.eye(dim)[None]*np.diagonal(s, axis1=-2, axis2=-1)[:,None,:]
    s = .8*s+.2*diagonal+np.eye(dim)[None]*(1e-8*scale[:,None,None])
    return np.linalg.inv(np.linalg.cholesky(s))


def energy(x, y, truth):
    """Equal-M matched-noise component banks; exclude ALL same-address pairs."""
    if x.shape != y.shape or x.ndim != 3 or x.shape[1] < 2:
        raise ValueError('Matching paired banks required')
    n = x.shape[1]; mask = ~np.eye(n, dtype=bool)
    dxx = np.linalg.norm(x[:,:,None,:]-x[:,None,:,:], axis=-1)[:,mask].mean(-1)
    dyy = np.linalg.norm(y[:,:,None,:]-y[:,None,:,:], axis=-1)[:,mask].mean(-1)
    dxy = np.linalg.norm(x[:,:,None,:]-y[:,None,:,:], axis=-1)[:,mask].mean(-1)
    a = np.linalg.norm(x-truth[:,None,:], axis=-1).mean(-1)
    b = np.linalg.norm(y-truth[:,None,:], axis=-1).mean(-1)
    return np.stack((a-.5*dxx, b-.5*dyy,
                     .5*(a+b)-.125*(dxx+dyy)-.25*dxy), axis=-1)


def moments(x, y):
    sx, sy = covariance(x), covariance(y)
    delta = (y-x).mean(1)
    between = .25*np.einsum('ci,cj->cij', delta, delta)
    noise = .25*covariance(y-x)/x.shape[1]
    return dict(mean17=x.mean(1), mean29=y.mean(1), mean_mix=(x.mean(1)+y.mean(1))/2,
                covariance17=sx, covariance29=sy, within_mix=(sx+sy)/2,
                between=between, between_debiased=between-noise,
                covariance_mix=(sx+sy)/2+between)


def effect(x, y, truth, whitening, denominator):
    m = moments(x, y)
    w = lambda a: whitening@a@whitening.transpose(0,2,1)
    delta = np.einsum('cij,cj->ci', whitening, (m['mean29']-m['mean17'])/2)
    change = w(m['covariance_mix']-m['covariance17'])
    between = w(m['between'])
    score = energy(x,y,truth).mean(0)
    return dict(mean_shift_sd=float(np.sqrt(np.mean(delta**2))),
        covariance_change=float(np.max(abs(np.linalg.eigvalsh(change)),axis=-1).mean()),
        between_covariance=float(np.linalg.eigvalsh(between)[:,-1].mean()),
        energy_change_abs=float(abs((score[2]-score[0])/denominator)),
        energy_gain=float((score[0]-score[2])/denominator),
        seed29_energy_gain=float((score[0]-score[1])/denominator))


def decision(groups):
    if len(groups) != 4 or any(g.get('draws',0)<8 or not g.get('valid',False) for g in groups):
        return 'insufficient_evidence'
    if all(g['bands'][k][2]<v for g in groups for k,v in LIMITS.items()):
        return 'futile_on_screened_support'
    if any(g['bands'][k][0]>v for g in groups for k,v in LIMITS.items()):
        return 'material_effect_detected'
    return 'insufficient_evidence'


def prepare():
    OUT.mkdir(exist_ok=False)
    generation = read(RUN/'MANIFEST.json')
    sources = [record(Path(__file__)), record(REPO/'docs/e2e_cfm_mixture_futility_20261006.md'),
               record(RUN/'MANIFEST.json'), record(CAL/'b1b2/prepare/COMPLETE.json'),
               record(CAL/'b1b2/replay/COMPLETE.json')]
    prep, replay = read(sources[-2]['path']), read(sources[-1]['path'])
    assert prep['passed'] and replay['passed'] and prep['panel_complete'] and replay['panel_complete']
    byid = {v['pair_id']:v for v in replay['cases']}
    cases=[]; products=[]
    for phase in PHASES:
        parity = RUN/phase/'SGC/seed17/d000/PARITY.json'
        assert read(parity)['passed']; sources.append(record(parity))
        for row in prep['cases']:
            if row['phase'] != phase or row['cap'] != 'SGC': continue
            pid=row['pair_id']
            cases.append(dict(phase=phase, pair_id=pid, cap='SGC', shell=row['shell'],
                support_stratum=row['support_stratum'], origin=row['origin_mpc_h'],
                selection=row['output'], labels=byid[pid]['output'],
                summaries=record(CAL/'a2'/(pid+'_SUMMARIES.npz'))))
        for first in (0,4):
            for seed in (17,29):
                if seed==17:
                    receipt=OLD/'fullcap_tides'/phase/'SGC'/f'd{first:03d}/COMPLETE.json'
                    assert sha(receipt)==generation['baseline_tides'][phase+f'/SGC/d{first:03d}']['sha256']
                else:
                    done=RUN/phase/'SGC/seed29'/f'd{first:03d}/COMPLETE.json'
                    completed=read(done); assert completed['passed']
                    assert completed['manifest_sha256']==sha(RUN/'MANIFEST.json')
                    receipt=done.parent/'tides/COMPLETE.json'
                    assert sha(receipt)==completed['auxiliary_sha256'];sources.append(record(done))
                r=read(receipt)
                assert r['passed'] and r['phase']==phase and r['cap']=='SGC' and r['first']==first and r['draws']==4
                binding=receipt.parent/'BINDING.json';assert sha(binding)==r['binding']
                bp=read(binding);assert bp['method']=='learned' and bp['padding']==32
                products.append(dict(phase=phase,seed=seed,first=first,receipt=record(receipt),
                                     binding=record(binding),output=r['output']))
    assert len(cases)==16 and len(products)==8
    write(OUT/'CONFIG.json',dict(schema='cfm-target-futility-v1',cases=cases,products=products,
          sources=sources,draws=8,bootstrap=1024,bootstrap_seed=610061,limits=LIMITS,
          cap='SGC',operator='original',threshold=.2,posterior_qualified=False))
    print(json.dumps(dict(config=str(OUT/'CONFIG.json'),sha256=sha(OUT/'CONFIG.json'))),flush=True)


def fractions(labels, weights):
    return np.stack([((labels==k)*weights).sum(-1)/weights.sum(-1) for k in range(4)],-1)


def analyze():
    import h5py
    from workflows.sbi.e2e_coupled_contract import require_compute
    require_compute();start=time.monotonic();cfg=read(OUT/'CONFIG.json')
    if cfg['draws']!=8 or cfg['limits']!=LIMITS:raise ValueError('Screen changed')
    for src in cfg['sources']:verify(src)
    cases=cfg['cases']; count=len(cases)
    gaps=np.empty((2,count,8,4)); fracs=np.empty((2,count,8,6))
    fullg=[];fullf=[];truthg=[];truthf=[];weights=[];oldlabels=[]
    for row in cases:
        for key in ('selection','labels','summaries'):verify(row[key])
        with np.load(row['selection']['path']) as p, np.load(row['labels']['path']) as l, np.load(row['summaries']['path']) as a:
            w=p['cell_counts'];weights.append(w)
            labels=l['labels'][0,0];oldlabels.append(labels[:8])
            fullf.append(fractions(labels,w)[...,:3].reshape(32,6))
            fullg.append(np.diff(a['original'],axis=-1).reshape(32,4))
            truthf.append(fractions(p['truth_cell_labels'][0][None],w)[0,:,:3].reshape(6))
            truthg.append(np.diff(a['physical_truth'],axis=-1).reshape(4))
    parity=0.;payloads=[]
    for product in cfg['products']:
        for key in ('receipt','binding','output'):verify(product[key])
        payloads.append(product['output'])
        arm=int(product['seed']==29);first=product['first'];phase=product['phase']
        with h5py.File(product['output']['path']) as f:
            assert f.attrs['phase']==phase and f.attrs['cap']=='SGC' and int(f.attrs['first_draw'])==first
            assert int(f.attrs['padding_cells'])==32 and float(f.attrs['cell_mpc_h'])==6.766
            for i,row in enumerate(cases):
                if row['phase']!=phase:continue
                offset=(np.asarray(row['origin'])-f.attrs['origin_mpc_h'])/6.766
                lo=np.rint(offset).astype(int)
                if np.max(abs(offset-lo))>1e-8 or np.any(lo<0):raise ValueError('Misaligned core')
                cut=tuple(slice(int(x),int(x+n)) for x,n in zip(lo,(32,16,16)))
                eig=f['eigenvalues'][(slice(None),*cut,slice(None))].astype(float).reshape(4,2,4096,3)
                if not np.isfinite(eig).all() or np.any(np.diff(eig,axis=-1)<0):raise ValueError('Invalid eigensystem')
                labels=(eig>.2).sum(-1)
                gaps[arm,i,first:first+4]=np.diff(eig.mean(2),axis=-1).reshape(4,4)
                fracs[arm,i,first:first+4]=fractions(labels,weights[i])[...,:3].reshape(4,6)
                if not arm:
                    np.testing.assert_array_equal(labels,oldlabels[i][first:first+4])
                    error=float(abs(gaps[0,i,first:first+4]-fullg[i][first:first+4]).max())
                    parity=max(parity,error)
                    if error>2e-7:raise ValueError('Stage A gap parity failed')
        print(json.dumps(dict(stage='extracted',phase=phase,seed=product['seed'],first=first)),flush=True)
    # Preserve a compact source-backed bank before any effect estimates.
    bank=OUT/'PANEL.npz'
    with bank.open('xb') as stream:
        np.savez_compressed(stream,gaps=gaps,fractions=fracs,full_gaps=np.asarray(fullg),
                            full_fractions=np.asarray(fullf),truth_gaps=np.asarray(truthg),truth_fractions=np.asarray(truthf))
    results=[];primary=[];matrices={}
    for phase in PHASES:
        rng=np.random.default_rng(cfg['bootstrap_seed'])
        indices=rng.integers(0,8,size=(cfg['bootstrap'],8))
        for name,bank,full,truth in [('gaps',gaps,np.asarray(fullg),np.asarray(truthg)),
                                    ('fractions',fracs,np.asarray(fullf),np.asarray(truthf))]:
            for stratum in ('all','interior','boundary'):
                mask=np.array([r['phase']==phase and (stratum=='all' or r['support_stratum']==stratum) for r in cases])
                if not mask.any():raise ValueError('Missing expected stratum '+stratum)
                x,y=bank[:,mask];t=truth[mask];w=metric(full[mask])
                scores=energy(x,y,t);den=float(scores[:,0].mean())
                group=dict(phase=phase,target=name,stratum=stratum,pairs=int(mask.sum()),draws=8,valid=den>1e-12)
                if not group['valid']:
                    group['reason']='Nonpositive fair-energy denominator';results.append(group);continue
                point=effect(x,y,t,w,den)
                boot=[effect(x[:,ix],y[:,ix],t,w,den) for ix in indices]
                bands={k:np.percentile([v[k] for v in boot],[5,50,95]).tolist() for k in point}
                mm=moments(x,y);key=phase+'_'+name+'_'+stratum
                matrices.update({key+'_'+k:v for k,v in mm.items()})
                matrices[key+'_truth']=t;matrices[key+'_whitening']=w
                residual=np.einsum('cij,cj->ci',w,mm['mean17']-t)
                shift=np.einsum('cij,cj->ci',w,(mm['mean29']-mm['mean17'])/2)
                projection=np.sum(residual*shift,axis=-1)
                group.update(point=point,bands=bands,energy=scores.mean(0).tolist(),
                    residual_shift_dot=float(projection.mean()),
                    whitened_mean_squared_error=[float(np.mean(v**2)) for v in (residual,residual+shift)],
                    halves=[effect(x[:,sl],y[:,sl],t,w,den) for sl in (slice(0,4),slice(4,8))])
                results.append(group)
                if (name,stratum) in [('gaps','all'),('fractions','interior')]:primary.append(group)
    with (OUT/'MOMENTS.npz').open('xb') as stream:np.savez_compressed(stream,**matrices)
    for src in cfg['sources']:verify(src)
    result=dict(passed=True,decision=decision(primary),groups=results,primary=primary,
        config=record(OUT/'CONFIG.json'),panel=record(OUT/'PANEL.npz'),moments=record(OUT/'MOMENTS.npz'),
        payloads=payloads,baseline_gap_maxabs=parity,baseline_labels_exact=True,
        seconds=time.monotonic()-start,job=os.environ['SLURM_JOB_ID'],
        interpretation='SGC-only resource screen; paired conditional MC bands, not population confidence or calibration',
        further_generation_authorized_by_result=False,posterior_qualified=False)
    write(OUT/'COMPLETE.json',result)
    print(json.dumps(dict(decision=result['decision'],seconds=result['seconds'],primary=[
        {k:g[k] for k in ('phase','target','stratum','point','bands','energy')} for g in primary])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','analyze']);args=p.parse_args()
    (prepare if args.action=='prepare' else analyze)()
