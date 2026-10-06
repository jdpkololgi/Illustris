"""Independent real-bank score QA against published Stage A/B/C parent results."""
import json
import os
from pathlib import Path
import sys
import time

RUN=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3')
BASE=RUN.parent/'coherent_exterior_20261002_v1'
CAL=RUN.parent/'calibration_investigation_20261006_v1'
sys.path.insert(0,str(BASE/'source_78156ac3417555ef269f'))
sys.path.insert(1,str(RUN/'evaluation_v1/source/workflows/sbi'))
sys.path.insert(2,str(RUN/'source/workflows/sbi'))
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
import e2e_cfm_conditioning_metrics as m


def read(path):return json.loads(Path(path).read_text())


def verify(rec):
    if c.sha256(rec['path'])!=rec['sha256']:raise ValueError('Changed input: '+rec['path'])


def run():
    c.require_compute();tick=time.monotonic();out=RUN/'score_verification_v1';out.mkdir(exist_ok=False)
    sources=[];results=[]
    aa=read(CAL/'a2/COMPLETE.json');bb=read(CAL/'b1b2/score/COMPLETE.json')
    pp=read(CAL/'b1b2/prepare/COMPLETE.json');rr=read(CAL/'b1b2/replay/COMPLETE.json')
    cc=read(CAL/'c_mechanism/full/COMPLETE.json');verify(cc['panel'])
    for q in (aa,bb,pp,rr,cc):
        if not q['passed'] or not q['panel_complete']:raise ValueError('Incomplete baseline')
    for path in ('a2/COMPLETE.json','b1b2/score/COMPLETE.json','b1b2/prepare/COMPLETE.json',
                 'b1b2/replay/COMPLETE.json','c_mechanism/full/COMPLETE.json'):
        sources.append(c.file_record(CAL/path,content_hash=True))
    _,counts=m.resamples()
    with np.load(cc['panel']['path']) as f:
        fracs=f['fractions'];targetfracs=f['truth'];names=list(f['pair_id'])
    if names!=[r['pair_id'] for r in cc['cases']]:raise ValueError('Stage C ordering mismatch')
    metadata=cc['cases'];eigen=[];truth=[];labels=[];labeltruth=[];weights=[]
    for row in metadata:
        if row['phase'] not in ('ph012','ph013'):raise PermissionError('Development only')
        pid=row['pair_id'];path=CAL/'a2'/(pid+'_SUMMARIES.npz')
        arow=next(q for q in aa['cases'] if q['pair_id']==pid);verify(arow['summary']);sources.append(arow['summary'])
        with np.load(path) as f:eigen.append([f['original'],f['corrected']]);truth.append(f['physical_truth'])
        rrow=next(q for q in rr['cases'] if q['pair_id']==pid)['output'];verify(rrow);sources.append(rrow)
        prow=next(q for q in pp['cases'] if q['pair_id']==pid)['output'];verify(prow);sources.append(prow)
        with np.load(rrow['path']) as f:labels.append(f['labels'][:,0])
        with np.load(prow['path']) as f:labeltruth.append(f['truth_cell_labels'][0]);weights.append(f['cell_counts'])
    eigen,truth,labels,labeltruth,weights=map(np.asarray,(eigen,truth,labels,labeltruth,weights))
    for phase in ('ph012','ph013'):
        ix=np.array([i for i,r in enumerate(metadata) if r['phase']==phase])
        inside=np.array([i for i in ix if metadata[i]['support_stratum']=='interior'])
        if len(ix)!=16 or len(inside)!=8:raise ValueError('Baseline support changed')
        ga=next(q for q in aa['groups'] if q['phase']==phase and q['group']=='all')
        gb=next(q for q in bb['groups'] if q['phase']==phase and q['group']=='all')
        gi=next(q for q in bb['groups'] if q['phase']==phase and q['group']=='support_stratum=interior')
        for mi,method in enumerate(('original','corrected')):
            gaps=np.diff(eigen[ix,mi],axis=-1).reshape(16,32,4)
            tg=np.diff(truth[ix],axis=-1).reshape(16,4)
            energy=m.score_from_kernel(m.energy_kernel(gaps,tg))
            np.testing.assert_allclose(energy,ga['metrics'][method+'/physical/joint4gap']['energy'],rtol=0,atol=1e-12)
            lab=labels[ix,mi].transpose(1,0,2,3).reshape(32,-1)
            fair=m.score_from_kernel(m.brier_kernel(lab,labeltruth[ix].reshape(-1),weights[ix].reshape(-1)))
            np.testing.assert_allclose(fair,gb['reliability'][method+'/0.2']['fair_brier'],rtol=0,atol=1e-12)
            # Stage C axes: case,method,threshold,weight,draw,core,class.
            x=fracs[inside,mi,0,0,...,:3].reshape(8,32,6)
            y=targetfracs[inside,0,0,...,:3].reshape(8,6)
            fraction_energy=m.score_from_kernel(m.energy_kernel(x,y))
            np.testing.assert_allclose(fraction_energy,gi['regional'][method+'/0.2/galaxy']['joint_pair']['energy'],rtol=0,atol=1e-12)
            ratio=m.ratio_samples(x,y,np.ones((1,32),int))[0]
            cg=next(q for q in cc['fraction_results'] if q['phase']==phase and q['operator']==method
                and q['threshold']==.2 and q['weighting']=='galaxy' and q['group']=='support_stratum=interior')
            np.testing.assert_allclose(ratio,cg['joint']['trace_adjusted_mse_over_variance'],rtol=0,atol=1e-10)
            if not np.isfinite(m.ratio_samples(x,y,counts)).all():raise ValueError('Nonfinite real-bank MC')
            results.append(dict(phase=phase,method=method,checks=4,passed=True))
    result=dict(passed=True,checks=16,results=results,seconds=time.monotonic()-tick,
        source=c.file_record(__file__,content_hash=True),metrics=c.file_record(m.__file__,content_hash=True),
        inputs=sources,job=os.environ['SLURM_JOB_ID'],scientific_gate_result=False)
    c.atomic_json(out/'COMPLETE.json',result)
    print(json.dumps(dict(passed=True,checks=16,seconds=result['seconds'])),flush=True)


if __name__=='__main__':run()
