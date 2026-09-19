"""Strict thirteen-phase global normalization; no per-field target rescaling."""
import argparse
import json
from pathlib import Path

import numpy as np

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_condition_products as products
from workflows.sbi import e2e_coupled_conditions as conditions

IDENTITY={'support_random','angular_response','exposure_apodized_random',
          'geometry_valid_fraction','los_x','los_y','los_z'}


class Moments:
    def __init__(self,channels):
        self.n=0; self.total=np.zeros(channels); self.square=np.zeros(channels)

    def add(self,values):
        x=np.asarray(values,dtype=np.float64).reshape(len(self.total),-1)
        if not np.isfinite(x).all(): raise ValueError('nonfinite normalization input')
        self.n+=x.shape[1]; self.total+=x.sum(1); self.square+=np.square(x).sum(1)

    def report(self):
        if not self.n: raise ValueError('empty training statistics')
        return dict(voxels_per_channel=self.n,mean=(self.total/self.n).tolist(),
                    second_moment=(self.square/self.n).tolist())


def equal_phase_statistics(phase_reports):
    if set(phase_reports)!=set(c.TRAIN):
        raise PermissionError('normalization requires exactly the thirteen training phases')
    result={}
    for name in ('joint','wide','coarse_logrho','fine_residual'):
        means=np.asarray([phase_reports[p][name]['mean'] for p in c.TRAIN])
        squares=np.asarray([phase_reports[p][name]['second_moment'] for p in c.TRAIN])
        if not np.isfinite(means).all() or not np.isfinite(squares).all():
            raise ValueError('nonfinite phase statistics')
        mean=means.mean(0); variance=np.maximum(0.,squares.mean(0)-mean**2)
        std=np.sqrt(variance)
        if name=='fine_residual':
            if np.max(np.abs(means))>2e-12:
                raise ValueError('fine chart is not block-mean-zero')
            mean=np.zeros_like(mean)
            std=np.sqrt(squares.mean(0))
        # A constant observed feature has no fitted scale information. It is
        # left on unit scale, not amplified by an epsilon denominator.
        std=np.where(std>1e-12,std,1.)
        result[name]=dict(mean=mean.tolist(),std=std.tolist())
    for name,channels in (('joint',products.LOCAL_CHANNELS),('wide',products.WIDE_CHANNELS)):
        result[name]['channels']=list(channels)
        for index,channel in enumerate(channels):
            if channel in IDENTITY:
                result[name]['mean'][index]=0.; result[name]['std'][index]=1.
    return result


def source_binding():
    return dict(builder_sha256=c.sha256(__file__),condition_builder_sha256=c.sha256(products.__file__),
                 reader_sha256=c.sha256(conditions.__file__),operators_sha256=c.sha256(op.__file__),
                 phase_auditor_sha256=c.sha256(c.REPO/'workflows/sbi/e2e_coupled_product_audit.py'),
                 layout_sha256=c.sha256(op.LAYOUT))


def phase_statistics(phase,directory,binding):
    # This guard must run before even consulting a held-out receipt or pointer.
    if phase not in c.TRAIN: raise PermissionError('normalization statistics are train-only')
    from workflows.sbi import e2e_coupled_audit_worker as audits
    from workflows.sbi.e2e_coupled_product_audit import read_target
    if not audits.qualified(phase): raise FileNotFoundError('phase has no completed independent audit')
    observations=coord.ROOT/'conditions'/phase/'CONDITIONS_COMPLETE.json'
    targets=coord.ROOT/'targets'/phase/'TARGETS_COMPLETE.json'
    pointer=coord.ROOT/'product_audit'/phase/'LATEST_AUDIT.json'
    audit_pointer=coord.verify_receipt(pointer,payload=False)
    audited=Path(audit_pointer['audit']['path'])
    cond=coord.verify_receipt(observations,payload=False)
    target=coord.verify_receipt(targets,payload=False)
    if cond['phase']!=phase or target['phase']!=phase or cond['role']!='train' or target['role']!='train':
        raise PermissionError('normalization source phase/role mismatch')
    if len(cond['pair_receipts'])!=128 or len(target['pair_receipts'])!=128:
        raise ValueError('incomplete training pair panel')
    phase_marker=directory/f'{phase}_MOMENTS.json'
    phase_binding=dict(**binding,condition_receipt_sha256=c.sha256(observations),
                       target_receipt_sha256=c.sha256(targets),audit_sha256=c.sha256(audited))
    sources=[c.file_record(p,content_hash=True) for p in (observations,targets,audited)]
    if phase_marker.exists():
        receipt=coord.verify_receipt(phase_marker,payload=False)
        if receipt['binding']!=phase_binding or receipt['phase']!=phase:
            raise ValueError('phase moments source drift')
        return receipt['moments'],sources+[c.file_record(phase_marker,content_hash=True)]
    target_map={Path(v['path']).stem:v for v in target['pair_receipts']}
    condition_map={Path(v['path']).stem:v for v in cond['pair_receipts']}
    if set(target_map)!=set(condition_map) or len(target_map)!=128:
        raise ValueError('observation/target pair IDs differ')
    moments=dict(joint=Moments(12),wide=Moments(12),coarse_logrho=Moments(1),fine_residual=Moments(1))
    pair_sources=[]
    for pair_id in sorted(target_map):
        for item in (target_map[pair_id],condition_map[pair_id]):
            if c.sha256(item['path'])!=item['sha256']: raise ValueError('pair receipt drift')
            pair_sources.append(item)
        arrays=conditions.load_pair(phase,pair_id)
        target_record=coord.verify_receipt(target_map[pair_id]['path'])
        if target_record['phase']!=phase or target_record['pair_id']!=pair_id:
            raise ValueError('target pair identity mismatch')
        values=read_target(c.guarded(target_record['outputs'][0]['path'],phase),phase,pair_id)
        rho=values['rho_joint']; coarse=values['coarse_rho_extended']
        _,residual=op.encode(rho)
        moments['joint'].add(arrays['joint']); moments['fine_residual'].add(residual[None])
        for offset in op.layout()['context_offsets_raw']:
            cropped=conditions.crop_context(arrays,phase,offset)
            moments['wide'].add(cropped['wide'])
            start=np.array(op.layout()['wide_crop_base_start'])+np.array(offset)//8
            wide=coarse[tuple(slice(s,s+48) for s in start)]
            if np.any(wide<=0): raise ValueError('invalid coarse density')
            moments['coarse_logrho'].add(np.log(wide)[None])
    report={key:value.report() for key,value in moments.items()}
    receipt=dict(**coord.provenance(),phase=phase,binding=phase_binding,
                 moments=report,pair_receipts=pair_sources,outputs=[],**{'pass':True})
    c.atomic_json(phase_marker,receipt)
    print(json.dumps(dict(phase=phase,normalization_pairs=128)),flush=True)
    return report,sources+[c.file_record(phase_marker,content_hash=True)]


def fit(collect_ready=False):
    """Collect immutable per-phase moments; publish only the complete panel.

    Collection may visit completed phases out of order. The final equal-phase
    calculation and its iteration order are identical to the all-ready fit.
    """
    c.require_compute(); coord.bind(); coord.require_host_checks()
    directory=coord.ROOT/'normalization'; binding=source_binding()
    with c.single_writer(directory):
        marker=directory/'NORMALIZATION_COMPLETE.json'
        if marker.exists():
            result=coord.verify_receipt(marker,payload=False)
            if result['binding']!=binding or result['fit_phases']!=list(c.TRAIN):
                raise ValueError('normalizer source/panel drift')
            for item in result['sources']:
                if c.sha256(item['path'])!=item['sha256']: raise ValueError('normalizer source drift')
            return result
        reports={}; sources=[]
        for phase in c.TRAIN:
            if collect_ready:
                from workflows.sbi import e2e_coupled_audit_worker as audits
                if not audits.qualified(phase): continue
            report,phase_sources=phase_statistics(phase,directory,binding)
            reports[phase]=report; sources.extend(phase_sources)
        if set(reports)!=set(c.TRAIN):
            return dict(complete=False,collected_phases=list(reports),
                        pending_phases=[p for p in c.TRAIN if p not in reports])
        result=dict(**coord.provenance(),binding=binding,fit_phases=list(c.TRAIN),sources=sources,
            normalization=equal_phase_statistics(reports),phase_weight=1/13,
            policy='Global train-only moments; equal phase weight; no per-field scaling; all voxels including padding',
            context_offsets_raw=op.layout()['context_offsets_raw'],outputs=[],**{'pass':True})
        c.atomic_json(marker,result)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect-ready',action='store_true')
    args=parser.parse_args()
    result=fit(collect_ready=args.collect_ready)
    print(json.dumps(dict(complete=True,fit_phases=result['fit_phases'])
                     if 'fit_phases' in result else result),flush=True)
