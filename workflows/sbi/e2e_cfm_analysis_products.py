"""Source-bound corrected-operator analysis access to existing qualified replay.

Adopts a development analysis default; never alters frozen generation products,
physical truth, or production defaults. Engineering replay is not calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

REPO=Path(__file__).resolve().parents[2]
PROFILE=REPO/'configs/e2e_cfm_analysis_v1.json'
BASE=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
CAL=BASE/'calibration_investigation_20261006_v1'
OUT=BASE/'cfm_tier1_analysis_20261006_v1'
METHODS=('original','corrected')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(16<<20),b''):h.update(part)
    return h.hexdigest()


def record(path):return dict(path=str(path),sha256=sha(path))


def verify(rec):
    if sha(rec['path'])!=rec['sha256']:raise ValueError('Changed analysis source: '+rec['path'])


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def guard(phase,method,threshold):
    if phase not in ('ph012','ph013'):raise PermissionError('Only registered development phases')
    if method not in METHODS or threshold not in (.2,0.):raise ValueError('Unregistered analysis convention')


def fraction_draws(labels,weights):
    if labels.shape[1:]!=weights.shape or np.any(weights<0) or np.any(weights.sum(-1)<=0):
        raise ValueError('Invalid/empty regional weights')
    return np.stack([((labels==k)*weights).sum(-1)/weights.sum(-1) for k in range(4)],axis=-1)


def select_arrays(summary,labels,counts,method='corrected',threshold=.2):
    if method not in METHODS or threshold not in (.2,0.):raise ValueError('Unregistered operator/threshold')
    mi=METHODS.index(method);ti=(.2,0.).index(threshold)
    eigen=np.asarray(summary[method]);chosen=labels[mi,ti]
    if eigen.shape!=(32,2,3) or labels.shape!=(2,2,32,2,4096):raise ValueError('Wrong replay shape')
    if not np.isfinite(eigen).all() or np.any(np.diff(eigen,axis=-1)<0) or np.any(labels>3):
        raise ValueError('Invalid eigenvalues or classes')
    np.testing.assert_allclose(summary['corrected'].sum(-1),summary['original'].sum(-1),atol=1e-10,rtol=0)
    volume=fraction_draws(chosen,np.ones_like(counts))
    np.testing.assert_array_equal(volume,summary[f'{method}_fractions_{threshold}'])
    galaxy=fraction_draws(chosen,counts)
    probability=np.stack([(chosen==k).mean(0) for k in range(4)],axis=-1)
    np.testing.assert_allclose(probability.sum(-1),1,atol=1e-14)
    return dict(eigenvalue_means=eigen.copy(),tidal_gap_means=np.diff(eigen,axis=-1),
        galaxy_fraction_draws=galaxy,volume_fraction_draws=volume,
        cell_class_probabilities=probability,physical_truth_eigenvalue_means=np.asarray(summary['physical_truth']).copy(),
        cell_galaxy_counts=counts.copy())


def load_pair(manifest,pair_id,operator=None,threshold=.2):
    manifest=read(manifest) if isinstance(manifest,(str,Path)) else manifest
    if manifest['schema']!='cfm-tier1-analysis-v1' or manifest['posterior_qualified']:
        raise ValueError('Wrong manifest or unsupported qualification')
    verify(manifest['profile']);profile=read(manifest['profile']['path'])
    method=profile['default_operator'] if operator is None else operator
    row=next((r for r in manifest['cases'] if r['pair_id']==pair_id),None)
    if row is None:raise KeyError('Pair is not in the registered development panel')
    guard(row['phase'],method,threshold)
    for key in ('summary','selection','labels'):verify(row[key])
    with np.load(row['summary']['path']) as s,np.load(row['labels']['path']) as l,np.load(row['selection']['path']) as p:
        arrays=select_arrays(s,l['labels'],p['cell_counts'],method,threshold)
        arrays['physical_truth_cell_labels']=p['truth_cell_labels'][(.2,0.).index(threshold)].copy()
    return dict(metadata=dict(pair_id=pair_id,phase=row['phase'],cap=row['cap'],shell=row['shell'],
                support_stratum=row['support_stratum'],operator=method,threshold=threshold,
                scope=profile['scope'],posterior_qualified=False),**arrays)


def build():
    start=time.monotonic();profile=read(PROFILE)
    assert profile['phases']==['ph012','ph013'] and profile['draws']==32 and profile['pairs']==32
    assert profile['default_operator']=='corrected' and not profile['posterior_qualified']
    paths=dict(a2=CAL/'a2/COMPLETE.json',a_review=CAL/'REVIEW.json',
               prepare=CAL/'b1b2/prepare/COMPLETE.json',replay=CAL/'b1b2/replay/COMPLETE.json',
               scores=CAL/'b1b2/score/COMPLETE.json')
    sources={k:record(p) for k,p in paths.items()};records={k:read(p) for k,p in paths.items()}
    for k,r in records.items():
        if not r['passed']:raise ValueError('Unverified source '+k)
        if k!='a_review' and not r['panel_complete']:raise ValueError('Incomplete source '+k)
    assert records['a_review']['a2']['sha256']==sources['a2']['sha256']
    assert records['replay']['binding']['prepared']['sha256']==sources['prepare']['sha256']
    assert records['scores']['binding']['replay']['sha256']==sources['replay']['sha256']
    prep={r['pair_id']:r for r in records['prepare']['cases'] if r['phase'] in profile['phases']}
    replay={r['pair_id']:r for r in records['replay']['cases']}
    cases=[]
    for r in records['a2']['cases']:
        pid=r['pair_id'];guard(r['phase'],'corrected',.2)
        cases.append({**{k:r[k] for k in ('pair_id','phase','cap','shell','support_stratum')},
            'summary':r['summary'],'selection':prep[pid]['output'],'labels':replay[pid]['output']})
    assert len(cases)==32 and len({r['pair_id'] for r in cases})==32
    phase_reports=[]
    for phase in profile['phases']:
        sg=next(v for v in records['a_review']['supplement']['groups'] if v['phase']==phase and v['group']=='all')
        bg=next(v for v in records['scores']['groups'] if v['phase']==phase and v['group']=='all')
        ag=next(v for v in records['a2']['groups'] if v['phase']==phase and v['group']=='all')
        metric_names=('width','joint4gap','joint6')
        phase_reports.append(dict(phase=phase,density=sg['metrics']['cached_fullcap/density'],
            regional_density=sg['metrics']['cached_fullcap/core_mass'],
            coarse_joint_trace=sg['metrics']['coarse_joint_trace'],
            tidal={method:{name:ag['metrics'][f'{method}/physical/{name}'] for name in metric_names} for method in METHODS},
            cell_tidal={method:{f'lambda{i}':ag['metrics'][f'{method}/physical/cell_lambda{i}'] for i in (1,2,3)} for method in METHODS},
            galaxy_reliability={method:bg['reliability'][f'{method}/0.2'] for method in ('baseline',*METHODS)},
            galaxy_fractions={method:bg['regional'][f'{method}/0.2/galaxy'] for method in METHODS}))
    manifest=dict(schema='cfm-tier1-analysis-v1',profile=record(PROFILE),source=record(Path(__file__)),
        sources=sources,cases=cases,posterior_qualified=False,production_default_changed=False,
        summary_recalibration_fitted=False,confirmation_opened=False)
    # Bounded read/verification of existing compact products, no FFT or sampling.
    checks=0
    for row in cases:
        for method in METHODS:
            for threshold in (.2,0.):
                q=load_pair(manifest,row['pair_id'],method,threshold)
                assert q['metadata']['operator']==method
                np.testing.assert_allclose(q['galaxy_fraction_draws'].sum(-1),1,atol=1e-14)
                checks+=1
        assert load_pair(manifest,row['pair_id'])['metadata']['operator']=='corrected';checks+=1
    training=[]
    for stage in ('coarse','fine'):
        for seed in (17,29):
            path=BASE/'cfm_pilot_20260924_v1'/f'{stage}_{seed}'/'BINDING.json';bound=read(path)
            assert bound['alpha']==.3 and bound['ema']==.999 and bound['lr']==[1e-4,1e-5]
            training.append(dict(stage=stage,seed=seed,source=record(path),alpha=bound['alpha'],
                ema=bound['ema'],lr=bound['lr'],profile=bound['source_profile']))
    # Inventory only: do not touch cancelled generation payloads.
    mixture=BASE/'cfm_model_mixture_20261006_v2';inventory=[]
    for phase in profile['phases']:
        for cap in ('NGC','SGC'):
            for first in range(0,32,4):
                root=mixture/phase/cap/'seed29'/f'd{first:03d}'
                status='complete' if (root/'COMPLETE.json').exists() else ('partial_preserved_do_not_use' if root.exists() else 'not_started')
                inventory.append(dict(phase=phase,cap=cap,first=first,status=status,
                    receipt=record(root/'COMPLETE.json') if status=='complete' else None))
    OUT.mkdir(exist_ok=False)
    write(OUT/'MANIFEST.json',manifest)
    write(OUT/'EVIDENCE.json',dict(phases=phase_reports,training=training,
        labels=dict(density='development_marginal_screen_pass_not_population_qualification',
            regional_density='phase_dependent_full_curve_screen_fail',
            tidal_summaries='bias_reduced_but_joint_coverage_fails',
            galaxy_probabilities='informative_not_reliably_calibrated',
            class_fractions='joint_coverage_fails_operator_effect_mixed',
            boundary_uncertainty='aggregate_moments_encouraging_not_qualified',
            downstream_property_contrasts='not_tested',summary_recalibration='not_fitted')))
    write(OUT/'MIXTURE_INVENTORY.json',dict(job='59422691',state='cancelled_by_user_request',tasks=inventory,outputs_deleted=False))
    write(OUT/'COMPLETE.json',dict(passed=True,checks=checks,manifest=record(OUT/'MANIFEST.json'),
        evidence=record(OUT/'EVIDENCE.json'),seconds=time.monotonic()-start,
        meaning='Source/selection/array parity verified; no new scientific validation',posterior_qualified=False))
    print(json.dumps(dict(output=str(OUT),checks=checks,seconds=time.monotonic()-start,
        mixture_completed_batches=sum(r['status']=='complete' for r in inventory),
        mixture_partial_batches=sum(r['status']=='partial_preserved_do_not_use' for r in inventory))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build']);parser.parse_args();build()
