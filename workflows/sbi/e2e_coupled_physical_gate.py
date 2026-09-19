"""Predeclared training-only physical-reference qualification of I/J operators."""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_target_products as targets

CONFIG=c.REPO/'configs/e2e_coupled_physics_v1.json'


def eigenvalues(tensor):
    matrix=np.empty(tensor.shape[:-1]+(3,3),dtype=np.float64)
    for index,(a,b) in enumerate(op.COMPONENTS):
        matrix[...,a,b]=tensor[...,index]; matrix[...,b,a]=tensor[...,index]
    return np.linalg.eigvalsh(matrix)


def core_rmse(prediction,truth):
    return np.sqrt(np.mean((prediction-truth)**2,axis=(1,2,3)))


def evaluate_pair(rho,wide_extended,fullbox,offset,workers=4):
    cfg=op.layout(); delta=rho-1
    truth=eigenvalues(fullbox)
    baseline=[]; independent=[]
    start=np.array(cfg['wide_crop_base_start'])+np.array(offset)//8
    wide=wide_extended[tuple(slice(v,v+48) for v in start)]-1
    joint_crop=np.array(cfg['joint_coarse_crop_in_wide'])-np.array(offset)[:,None]//8
    joint_tensor=op.consistent_tensor(delta,wide,joint_crop,workers=workers)
    joint_core=targets.owned_core_arrays(joint_tensor)
    for side,crop in enumerate(cfg['independent_parent_crops_in_joint']):
        parent=delta[tuple(slice(*sl) for sl in crop)]
        baseline.append(op.tensor_from_delta(parent,6.766,workers)[16:32,16:32,16:32])
        wide_crop=joint_crop.copy()
        wide_crop[0]=[joint_crop[0,0]+4*side,joint_crop[0,0]+4*side+12]
        tensor=op.consistent_tensor(parent,wide,wide_crop,workers=workers)
        independent.append(tensor[16:32,16:32,16:32])
    baseline=np.stack(baseline); independent=np.stack(independent)
    owned=targets.owned_core_arrays(delta)
    trace_errors={name:float(np.max(np.abs(tensor[...,[0,3,5]].sum(-1)-owned)))
                  for name,tensor in (('parent_only',baseline),('independent',independent),('joint',joint_core))}
    if max(trace_errors.values())>2e-6: raise ValueError('candidate physical operator trace failure')
    base_error=core_rmse(eigenvalues(baseline),truth)
    if np.any(base_error<=1e-12): raise ValueError('parent-only reference has degenerate error ratio')
    report={}
    for name,tensor in (('independent',independent),('joint',joint_core)):
        eigen=eigenvalues(tensor); error=core_rmse(eigen,truth)
        gap=core_rmse(np.diff(eigen,axis=-1),np.diff(truth,axis=-1))
        report[name]=dict(eigen_rmse=error.tolist(),gap_rmse=gap.tolist(),
                         improvement=(1-error/base_error).tolist())
    return dict(offset_raw=list(offset),parent_only_eigen_rmse=base_error.tolist(),
                operators=report,trace_max_abs=trace_errors,
                independent_joint_tensor_rmse=core_rmse(independent,joint_core).tolist())


def decide(cases,cfg):
    primary=[v for v in cases if v['offset_raw']==cfg['primary_context_offset_raw']]
    expected={(phase,cap,shell,kind) for phase in cfg['phases'] for cap in ('NGC','SGC')
              for shell in range(4) for kind in ('interior','boundary')}
    identities=[(v['phase'],v['cap'],v['shell'],v['support_stratum']) for v in primary]
    if len(primary)!=32 or set(identities)!=expected or len(set(identities))!=32:
        raise ValueError('physical qualification panel incomplete/duplicated')
    results={}
    for name in ('independent','joint'):
        pooled=np.concatenate([v['operators'][name]['improvement'] for v in primary])
        medians=np.median(pooled,axis=0)
        phases={p:np.median(np.concatenate([v['operators'][name]['improvement'] for v in primary
                                          if v['phase']==p]),axis=0).tolist() for p in cfg['phases']}
        passed=(np.all(medians>=cfg['required_pooled_median_improvement']) and
                all(np.all(np.array(v)>=cfg['required_each_phase_median_improvement']) for v in phases.values()))
        results[name]=dict(pooled_median_improvement=medians.tolist(),phase_medians=phases,
                           minimum_improvement=pooled.min(0).tolist(),**{'pass':bool(passed)})
    return dict(operators=results,**{'pass':all(v['pass'] for v in results.values())})


def run(workers=4):
    c.require_compute()
    cfg=json.loads(CONFIG.read_text())
    if cfg['phases']!=['ph007','ph008'] or any(p not in c.TRAIN for p in cfg['phases']):
        raise PermissionError('physical reference gate is restricted to registered training phases')
    directory=coord.ROOT/'physical_gate'; started=time.monotonic()
    binding=dict(config_sha256=c.sha256(CONFIG),layout_sha256=c.sha256(op.LAYOUT),
                 operator_sha256=c.sha256(op.__file__),builder_sha256=c.sha256(__file__))
    with c.single_writer(directory):
        final=directory/'PHYSICAL_GATE_COMPLETE.json'
        if final.exists():
            # A failed scientific gate is still a completed measurement, so
            # validate authorities without mistaking pass=False for corruption.
            result=json.loads(final.read_text())
            if (result['config_sha256']!=c.sha256(c.CONFIG)
                    or result['coordinate_sha256']!=c.sha256(coord.CONFIG)):
                raise ValueError('physical gate data/coordinate authority drift')
            if result['binding']!=binding: raise ValueError('physical gate source changed under existing result')
            for item in result['sources']:
                if c.sha256(item['path'])!=item['sha256']: raise ValueError('physical gate input receipt drift')
            return result
        cases=[]; sources=[]
        for phase in cfg['phases']:
            gp=coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
            tp=coord.ROOT/'targets'/phase/'TARGETS_COMPLETE.json'
            geo=coord.verify_receipt(gp,payload=False); target=coord.verify_receipt(tp,payload=False)
            sources.extend(c.file_record(p,content_hash=True) for p in (gp,tp))
            receipts={Path(v['path']).stem:v for v in target['pair_receipts']}
            selected=[r for r in geo['pairs'] if r['pair_id'].endswith('_00')]
            if len(selected)!=16: raise ValueError('physical geometry strata missing')
            for row in selected:
                item=receipts[row['pair_id']]
                if c.sha256(item['path'])!=item['sha256']: raise ValueError('physical target receipt drift')
                record=coord.verify_receipt(item['path'])
                with h5py.File(c.guarded(record['outputs'][0]['path'],phase),'r') as f:
                    rho=f['rho_joint'][:]; wide=f['coarse_rho_extended'][:]; tensor=f['fullbox_tensor_cores'][:]
                for offset in op.layout()['context_offsets_raw']:
                    diagnostic=evaluate_pair(rho,wide,tensor,offset,workers)
                    cases.append(dict(phase=phase,pair_id=row['pair_id'],cap=row['cap'],shell=row['shell'],
                                      support_stratum=row['support_stratum'],**diagnostic))
                sources.append(c.file_record(item['path'],content_hash=True))
                print(json.dumps(dict(phase=phase,physical_pair=row['pair_id'],offsets=7)),flush=True)
        decision=decide(cases,cfg)
        result=dict(**coord.provenance(),binding=binding,sources=sources,cases=cases,
            decision=decision,elapsed_seconds=time.monotonic()-started,
            no_posterior_or_predictive_scoring=True,outputs=[],**{'pass':decision['pass']})
        c.atomic_json(final,result)
        return result


def smoke(workers=4):
    """Check one fixed training pair and all offsets; never claim the full gate."""
    c.require_compute(); phase='ph007'
    gp=coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
    geo=coord.verify_receipt(gp,payload=False); row=geo['pairs'][0]
    tp=coord.ROOT/'targets'/phase/'TARGETS_COMPLETE.json'
    done=coord.verify_receipt(tp,payload=False)
    item=next(v for v in done['pair_receipts'] if Path(v['path']).stem==row['pair_id'])
    if c.sha256(item['path'])!=item['sha256']: raise ValueError('smoke target receipt drift')
    record=coord.verify_receipt(item['path'])
    with h5py.File(c.guarded(record['outputs'][0]['path'],phase),'r') as f:
        rho=f['rho_joint'][:]; wide=f['coarse_rho_extended'][:]; tensor=f['fullbox_tensor_cores'][:]
    cases=[evaluate_pair(rho,wide,tensor,offset,workers) for offset in op.layout()['context_offsets_raw']]
    result=dict(**coord.provenance(),phase=phase,pair_id=row['pair_id'],cases=cases,
        source_sha256=c.sha256(__file__),physics_config_sha256=c.sha256(CONFIG),
        target_receipt_sha256=c.sha256(item['path']),full_registered_gate_pass=None,
        interpretation='Numerical/operator smoke only; full 64-core physical gate not evaluated',**{'pass':True})
    directory=coord.ROOT/'physical_gate'
    with c.single_writer(directory):
        c.atomic_json(directory/f'OPERATOR_SMOKE_{time.time_ns()}.json',result)
    return result


def wait_for_targets(seconds):
    """Bounded preparation dependency wait inside an existing compute step."""
    c.require_compute()
    deadline=time.monotonic()+seconds
    required=[coord.ROOT/'targets'/phase/'TARGETS_COMPLETE.json' for phase in ('ph007','ph008')]
    while True:
        missing=[str(path) for path in required if not path.exists()]
        if not missing: return True
        if time.monotonic()>=deadline: return False
        print(json.dumps(dict(physical_gate_waiting_for=missing)),flush=True)
        time.sleep(min(60,max(0,deadline-time.monotonic())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--wait-seconds',type=float,default=0)
    a=p.parse_args()
    if a.wait_seconds<0 or (a.smoke and a.wait_seconds): p.error('invalid dependency wait')
    if not a.smoke and a.wait_seconds and not wait_for_targets(a.wait_seconds):
        print(json.dumps(dict(paused=True,physical_gate_not_evaluated=True)),flush=True)
        raise SystemExit(75)
    result=smoke(a.workers) if a.smoke else run(a.workers)
    print(json.dumps(result.get('decision',{'operator_smoke_pass':result['pass'],'full_gate_pass':None})),flush=True)
    raise SystemExit(0 if result['pass'] else 2)
