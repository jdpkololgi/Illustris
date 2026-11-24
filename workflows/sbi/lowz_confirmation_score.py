"""Paper 3 scoring only; cannot generate, tune or change predictions."""
import os,sys,gc
from pathlib import Path
import numpy as np
import h5py
from scipy import fft
import lowz_confirmation_inputs as b

def smoothed_spectrum(count_path,workers=32):
    counts=np.load(count_path,mmap_mode='r',allow_pickle=False)
    n=counts.shape[0]
    if counts.shape!=(2048,)*3 or counts.dtype!=np.float32:
        raise ValueError('native2048 float32 counts required')
    mean=float(counts.mean(dtype=np.float64))
    if not mean>0:
        raise ValueError('nonpositive native mean')
    delta=np.empty(counts.shape,dtype=np.float64)
    for i in range(0,n,16):
        x=counts[i:i+16]
        if not np.isfinite(x).all() or np.min(x)<0:
            raise ValueError('invalid count payload')
        delta[i:i+16]=x/mean-1
    spectrum=fft.rfftn(delta,workers=workers,overwrite_x=True)
    del delta,counts; gc.collect()
    k=2*np.pi*fft.fftfreq(n,d=2000./n)
    kz=2*np.pi*fft.rfftfreq(n,d=2000./n)
    for i in range(n):
        spectrum[i]*=np.exp(-.5*7.**2*(k[i]**2+k[:,None]**2+kz[None,:]**2))
    spectrum[0,0,0]=0
    return spectrum,mean

def core(field,row,op):
    parts=[]
    for dx in (0,32):
        ctr=np.asarray(row['center'])+[dx,0,0]
        starts=[v-16+2*np.arange(16) for v in ctr]
        parts.append(op.sample_averaged_local(field,starts,2,row['grid']))
    return np.stack(parts).reshape(32,16,16)

def run():
    import socket
    assert __debug__ and socket.gethostname().startswith('nid') and os.environ['SLURM_JOB_ID']
    assert Path.cwd()==b.DEPLOY and os.environ['GRAPHWEB_CFM_DATA_CONFIG']==b.CONFIG
    _,art=b.release('ph018')
    idx=b.REPO/'docs/evidence/lowz_blind_cfm_prediction_index.json'
    baseline=b.OUT/'matched_reference_v1/BASELINE_FROZEN.json'
    geometry=b.OUT/'GEOMETRY_FROZEN.json'
    assert b.sha(idx)=='e9c78744e3c45b6679dbe42288f95f37d30f1821efc9ace6ddd9ab4f40f52397'
    assert b.sha(baseline)=='beadad0e7baffa11696156810d009501bac3f754a6cea126e9387b6fb2dc7998'
    br=b.read(baseline); assert br['cfm_index']==b.sha(idx) and br['geometry']==b.sha(geometry)
    for rel,h in art['source_files'].items(): assert b.sha(b.DEPLOY/rel)==h
    for key in ['lowz_ensemble_scores.py']:
        v=art['methods'][key]; assert b.sha(v['path'])==v['sha256']
    for v in art['sampler_helpers'].values(): assert b.sha(v['path'])==v['sha256']
    sys.path[:0]=[str(b.DEPLOY),str(Path(art['sampler_helpers']['lowz_canvas_check.py']['path']).parent)]
    from workflows.sbi import e2e_coupled_operators as op,e2e_field_error_budget as nat,e2e_footprint_tidal_operator as tidal
    from lowz_ensemble_scores import metrics,pilot
    rng=np.random.default_rng(61703); files={}
    for ph,cap,i,ih,dh,th in b.read(idx)['rows']:
        d=b.OUT/ph/cap/'predictions'/f'draw_{i:03d}'
        rec=[d/'INPUTS.json',d/'density/COMPLETE.json',d/'tides/COMPLETE.json']
        assert list(map(b.sha,rec))==[ih,dh,th]
        files[ph,cap,i]=tuple(b.verify_record(b.read(p)[k],p.parent) for p,k in zip(rec[1:],['field','output']))
    refs={(r['phase'],r['cap']):b.verify_record(r,baseline.parent) for r in br['outputs']}
    out=b.OUT/('blind_score_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    for ph,expected in [('ph018','098af26bd0211e9bdde5051aa23dbf669dc507004e6eacbb1666473a2bfc943c'),('ph019','4ac31f5c210c89c1a461d71c3857f13e59bc2978364d4327348a8ff0acd4a4be')]:
        b.release(ph); rows=b.read(geometry)['phases'][ph]['pairs']
        p=b.BASE/'coupled_20260918_v1/matter'/ph/'DENSITY_COMPLETE.json'
        assert b.sha(p)==expected
        rr=b.read(p); assert rr['pass'] and rr['phase']==ph and rr['processed_files']==136 and rr['subsample_fraction']==.1 and rr['box_mpc_h']==2000
        spectrum,_=smoothed_spectrum(b.verify_record(rr['outputs'][0],p.parent),32)
        field=fft.irfftn(spectrum.copy(),s=(2048,)*3,workers=32,overwrite_x=True)
        rho=np.stack([1+core(field,row,op) for row in rows]); del field
        tensor=np.empty((*rho.shape,6))
        for j,comp in enumerate(op.COMPONENTS):
            field=nat.inverse_component(spectrum,2000.,comp,workers=32)
            for n,row in enumerate(rows):tensor[n,...,j]=core(field,row,op)
            del field;gc.collect()
        del spectrum
        assert np.max(abs(tensor[...,[0,3,5]].sum(-1)-(rho-1)))<2e-6 and rho.min()>0
        y=np.concatenate([rho[...,None],tidal.eigensystem(tensor)],axis=-1)
        np.save(out/(ph+'_truth.npy'),y,allow_pickle=False)
        results=[]
        for n,row in enumerate(rows):
            cap=row['cap']; pair=row['pair_id']; xs=[]
            for i in range(32):
                a,t=files[ph,cap,i]
                with h5py.File(a) as f,h5py.File(t) as g:
                    sl=pilot.cuts(row,f.attrs['origin_mpc_h'])
                    xs.append(np.concatenate([f['rho'][(0,*sl)][...,None],g['eigenvalues'][(0,*sl)]],axis=-1).reshape(-1,4))
            xs=np.stack(xs)
            with h5py.File(refs[ph,cap]) as f:
                for shell,g in f[pair].items():
                    cells=g['flat_cell'][:]; truth=y[n].reshape(-1,4)[cells]
                    for method,v in [('cfm',xs[:,cells]),('baseline',g['draws'][:])]:
                        assert np.isfinite(v).all() and v.shape==(32,len(cells),4)
                        lo,hi=np.quantile(v,[.05,.95],axis=0)
                        for weighting,w in [('cells',np.ones(len(cells))),('galaxies',g['galaxy_weight'][:])]:
                            if w.sum()==0:continue
                            result=metrics(v,truth,w,rng)
                            result.update(coverage90=np.average((lo<=truth)&(truth<=hi),axis=0,weights=w).tolist(),
                                width90=np.average(hi-lo,axis=0,weights=w).tolist())
                            results.append(dict(phase=ph,cap=cap,pair=pair,shell=int(shell),support=row['support_stratum'],
                                method=method,weighting=weighting,**result))
        pilot.c.atomic_json(out/(ph+'_scores.json'),dict(rows=results,native_receipt=expected,geometry=b.sha(geometry)))
    pilot.c.atomic_json(out/'SCORES_COMPLETE.json',dict(index=b.sha(idx),baseline=b.sha(baseline),runner=b.sha(__file__),
        phases={ph:b.sha(out/(ph+'_scores.json')) for ph in ['ph018','ph019']},passed=True))

if __name__=='__main__':run()
