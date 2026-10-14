"""Independent source, panel and scientific-gate receipt reconciliation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(root,pilot=False):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('Compute allocation required')
    root = Path(root); receipt = read(root/'COMPLETE.json'); checks = 0

    def check(value):
        nonlocal checks
        if not value:
            raise AssertionError(f'Check {checks+1} failed')
        checks += 1

    def stationary(result):
        if not result['stable']:
            check(result['spectral_radius']>=1 and not result['accuracy_passed'])
            return
        check(result['spectral_radius']<1 and result['lyapunov_relative_residual']<=1e-8)
        coverage = max(abs(x-.9) for x in result['probe_coverage90'])
        check(abs(coverage-result['maximum_coverage_error'])<1e-14)
        expected = (result['worst_relative_covariance_error']<=.01 and result['covariance_relative']<=.01
                    and result['whitened_mean_error']<=.01 and coverage<=.005)
        check(result['accuracy_passed']==expected)

    check(receipt['passed'] and not receipt['abacus_inputs'] and not receipt['posterior_qualified'])
    for source in receipt['sources']:
        check(sha(source['path'])==source['sha256'])
    if not pilot:
        check(receipt['diagnostic_complete'] and not receipt['training'])
        check(len(receipt['targets'])==20 and len(receipt['rows'])==264)
        keys = {(r['case'],r['shift'],r['order'],r['context'],r['arm']) for r in receipt['rows']}
        check(len(keys)==264)
        for target in receipt['targets']:
            part = read(root/(target['case']+'.json'))
            check(part['passed'] and part['target']==target and part['sources']==receipt['sources'])
            check(part['runtime']==receipt['runtime'])
            check(part['rows']==[r for r in receipt['rows'] if r['case']==target['case']])
            geometries = [(0,'forward')]
            if target['case']=='n8_long_range_o1':
                geometries += [(0,'reverse'),(1,'forward')]
            expected = {(target['case'],s,o,c,a) for s,o in geometries
                        for c in receipt['screening'] for a in ('local','local_global')}
            check({k for k in keys if k[0]==target['case']}==expected)
        for row in receipt['rows']:
            stationary(row['stationary'])
            inv = row['invariance']
            check(row['exact_invariance_passed']==(inv['mean_relative']<=1e-8 and inv['covariance_relative']<=1e-8 and inv['spectral_radius']<1))
            check(row['expected_conditional_kl_mean']>=0 and row['expected_conditional_kl_max']>=row['expected_conditional_kl_mean']-1e-12)
            if row['stationary']['stable']:
                check(row['worst_probe_tau']==max(row['stationary_tau']))
                check(abs(row['median_sweep_seconds']*row['worst_probe_tau']-row['seconds_per_effective_draw_worst_probe'])<1e-12)
            else:
                check(row['stationary_tau'] is None and row['worst_probe_tau'] is None and row['curve']==[])
            if row['context']=='full' or 'independent' in row['case']:
                check(row['exact_invariance_passed'] and row['stationary']['accuracy_passed'])
        for name,screen in receipt['screening'].items():
            rows = [r for r in receipt['rows'] if r['context']==name]
            check(screen['total']==len(rows))
            check(screen['exact_passes']==sum(r['exact_invariance_passed'] for r in rows))
            check(screen['stationary_accuracy_passes']==sum(r['stationary']['accuracy_passed'] for r in rows))
            check(screen['all_stationary_accurate']==all(r['stationary']['accuracy_passed'] for r in rows))
        print('SCREENING',json.dumps(receipt['screening']))
        print('UNSTABLE',json.dumps([dict(case=r['case'],context=r['context'],arm=r['arm'],shift=r['shift'],order=r['order'],radius=r['stationary']['spectral_radius']) for r in receipt['rows'] if not r['stationary']['stable']]))
        for row in receipt['rows']:
            if row['case'] in ('n8_spectral_o1','n8_long_range_o1','n6_open_o1') and row['shift']==0 and row['order']=='forward':
                print('SELECTED',json.dumps(dict(case=row['case'],context=row['context'],arm=row['arm'],
                    stationary=row['stationary'],conditional_kl=row['expected_conditional_kl_mean'])))
    else:
        check(receipt['engineering_complete'] and receipt['updates']==20000 and len(receipt['losses'])==20000)
        check(len(receipt['checkpoints'])==4 and len(receipt['evaluations'])==4)
        for item in receipt['checkpoints']+receipt['evaluations']:
            check(sha(item['path'])==item['sha256'])
        controls = read(root/'CONTROLS.json')
        check(controls['passed'] and len(controls['rows'])==32)
        for row in controls['rows']:
            stationary(row['stationary'])
            if row['nfe']==512:
                check(row['stationary']['accuracy_passed'])
                check(max(r['worst_relative_covariance_error'] for r in row['components'])<=.001)
        final = None
        for update in (0,5000,10000,20000):
            evaluation = read(root/f'evaluation_{update}.json')
            check(evaluation['update']==update and evaluation['sources']==receipt['sources'])
            keys = {(r['observation_case'],r['nfe'],r['arm']) for r in evaluation['rows']}
            check(keys=={(c,n,a) for c in range(8) for n in (128,256,512) for a in ('learned_local','learned_local_oracle_global')})
            for row in evaluation['rows']:
                stationary(row['stationary'])
            for component in evaluation['conditional_components']:
                check(len(component['blocks'])==8)
                for row in component['blocks']:
                    check(row['component_accuracy_passed']==(row['mean_whitened_rms']<=.01 and row['worst_relative_covariance_error']<=.01))
                check(component['all_component_accuracy_passed']==all(r['component_accuracy_passed'] for r in component['blocks']))
            finest = [r for r in evaluation['rows'] if r['nfe']==512 and r['arm']=='learned_local']
            expected = all(r['stationary']['accuracy_passed'] for r in finest)
            check(evaluation['learned_local_stationary_passed']==expected)
            refined = evaluation['endpoint_refinement'][-1]
            check(evaluation['refinement_passed']==(refined['covariance_relative_change']<=.001 and refined['mean_response_relative_change']<=.001))
            check(evaluation['actual_model_check']['passed'] and evaluation['actual_model_check']['max_absolute_error']<1e-8)
            check(evaluation['pilot_accuracy_passed']==(expected and evaluation['refinement_passed'] and evaluation['conditional_components'][-1]['all_component_accuracy_passed']))
            final = evaluation
            print('CHECKPOINT',json.dumps(dict(update=update,passed=evaluation['pilot_accuracy_passed'],
                components=evaluation['conditional_components'][-1],
                max_stationary_covariance=max((r['stationary'].get('worst_relative_covariance_error',0) for r in finest)),
                max_stationary_mean=max((r['stationary'].get('whitened_mean_error',0) for r in finest)))))
        check(receipt['pilot_accuracy_passed']==final['pilot_accuracy_passed'])
        print('REFINEMENT',json.dumps(final['endpoint_refinement']))
        for arm in ('learned_local','learned_local_oracle_global'):
            rows = [r for r in final['rows'] if r['nfe']==512 and r['arm']==arm]
            print('FINAL_ARM',json.dumps(dict(arm=arm,accurate=sum(r['stationary']['accuracy_passed'] for r in rows),
                total=len(rows),stable=sum(r['stationary']['stable'] for r in rows),
                coverage_error=max(r['stationary'].get('maximum_coverage_error',0) for r in rows),
                covariance_relative=max(r['stationary'].get('covariance_relative',0) for r in rows),
                worst_covariance=max(r['stationary'].get('worst_relative_covariance_error',0) for r in rows))))
    result = dict(passed=True,checks=checks,complete_sha256=sha(root/'COMPLETE.json'),
        validator_sha256=sha(__file__),job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
        scope='Independent receipt, source and gate reconciliation; no second numerical implementation')
    with (root/'VALIDATION.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print('VALIDATED',json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--pilot',action='store_true')
    args = parser.parse_args()
    validate(args.root,args.pilot)
