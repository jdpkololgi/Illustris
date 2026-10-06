"""Independent arithmetic checks on the actual completed calibration products."""
import argparse
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_coupled_contract as c


def load(item):
    if c.sha256(item['path'])!=item['sha256']:raise ValueError('Payload changed')
    with np.load(item['path']) as f:return {k:f[k] for k in f.files}


def verify(root):
    c.require_compute()
    summary=json.loads((root/'summary32/SUMMARY.json').read_text())
    prepared=json.loads((root/'inputs/PREPARED.json').read_text())
    scores=json.loads((root/'scores32/SCORES.json').read_text())
    if len(summary['cases'])!=32 or scores['draws']!=32:raise ValueError('Wrong evaluation panel')
    identities={(v['phase'],v['cap'],v['shell'],v['support_stratum']) for v in summary['cases']}
    expected={(phase,cap,shell,kind) for phase in ('ph016','ph017') for cap in ('NGC','SGC')
              for shell in range(4) for kind in ('interior','boundary')}
    if identities!=expected:raise ValueError('Missing or duplicate evaluation stratum')
    coverage_checks=0;max_error=0.;max_mass=0.;max_overlap=0.;max_crps=0.;seconds=0.
    for case in prepared['cases']:
        if case['phase'] not in ('ph016','ph017'):raise PermissionError('Unregistered phase')
        target=load(case['truth'])['rho_joint']-1
        # Independently spell out crops instead of using the scoring helper.
        truth_density=np.stack([target[16:32,16:32,16:32],target[32:48,16:32,16:32]])
        truth_sides=np.array([target[40:48,16:32,16:32].mean(),target[48:56,16:32,16:32].mean()])
        row=next(v for v in summary['cases'] if v['pair_id']==case['pair_id'])
        marker=json.loads((root/'samples32'/case['pair_id']/'COMPLETE.json').read_text())
        seconds+=marker['seconds']
        if marker['binding']!=c.digest(scores['binding']):raise ValueError('Case binding mismatch')
        for v in marker['numerical']:
            max_mass=max(max_mass,v['max_mass_error'])
            if v['variant']=='shared_state':max_overlap=max(max_overlap,v['overlap_maxabs'])
        for item in marker['outputs']:
            data=load(item);name=Path(item['path']).stem
            pairs=[('density',data['density'],truth_density),
                   ('core_mass',data['regional'][:,:2],truth_density.mean((1,2,3))),
                   ('seam_average',data['seam_mass'].mean(1),truth_sides.mean()),
                   ('seam_contrast',data['seam_mass'][:,1]-data['seam_mass'][:,0],truth_sides[1]-truth_sides[0])]
            np.testing.assert_allclose(data['density'].mean((2,3,4)),data['regional'][:,:2],rtol=0,atol=1e-14)
            for label,x,y in pairs:
                ordered=np.sort(x,axis=0)
                # M32 central nearest90%: zero-based order1 and30, null29/33.
                coverage=float(np.mean((y>=ordered[1])&(y<=ordered[30])))
                error=abs(coverage-row['methods'][name][label]['coverage']['0.9'])
                max_error=max(max_error,error);coverage_checks+=1
            # Independent direct O(M^2) CRPS for both core means, all actual cases.
            x=data['regional'][:,:2];y=truth_density.mean((1,2,3))
            pair=np.abs(x[:,None]-x[None,:]).sum((0,1))/(2*32*31)
            direct=float((abs(x-y).mean(0)-pair).mean())
            max_crps=max(max_crps,abs(direct-row['methods'][name]['core_mass']['crps']))
    if max_error>1e-14 or max_crps>1e-12 or max_overlap!=0 or max_mass>2e-6:
        raise ValueError('Independent verification failed')
    groups=summary['groups']
    for method in ('original','independent_tiled','shared_state'):
        for metric,value in groups['all']['statistics'][method].items():
            if isinstance(value,dict) and 'coverage' in value:
                for level,v in value['coverage'].items():
                    mean=np.mean([groups[f'phase={phase}']['statistics'][method][metric]['coverage'][level]
                                  for phase in ('ph016','ph017')])
                    if abs(v-mean)>1e-14:raise ValueError('Pooled coverage does not reconcile')
    for key,value in summary['tarp']['curves'].items():
        if '/all/' in key:
            mean=np.mean([summary['tarp']['curves'][key.replace('/all/',f'/{phase}/')]
                          for phase in ('ph016','ph017')],axis=0)
            np.testing.assert_allclose(value,mean,rtol=0,atol=1e-14)
    record=dict(passed=True,cases=32,coverage_checks=coverage_checks,
                max_coverage_difference=max_error,max_direct_core_crps_difference=max_crps,
                max_relative_block_mass_error=max_mass,max_shared_overlap_difference=max_overlap,
                generation_case_seconds_total=seconds,summary_sha256=c.sha256(root/'summary32/SUMMARY.json'),
                script_sha256=c.sha256(__file__))
    c.atomic_json(root/'VERIFIED.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    verify(p.parse_args().root)
