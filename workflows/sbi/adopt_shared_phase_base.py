"""Qualify immutable CFM native/observation inputs under the new phase authority.

Freshly hash output payloads. Historical raw-source validation remains inherited
and explicitly recorded; no claim of a fresh particle or CFS catalogue rescan.
Geometry, targets, conditions, audits and normalizers are not adopted here.
"""
import argparse
import copy
import json
from pathlib import Path
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord


def physical_equivalence(old,new):
    for section in ('target','geometry'):
        if old[section]!=new[section]:raise ValueError('Physical configuration drift: '+section)
    a={k:v for k,v in old['observation'].items() if k!='selection_policy'}
    b={k:v for k,v in new['observation'].items() if k!='selection_policy'}
    if a!=b:raise ValueError('Observation configuration drift')


def adopt(phase):
    c.require_compute();c.phase_guard(phase);coord.bind();coord.require_host_checks()
    cfg=c.config()
    if not cfg.get('phase_policy'):raise ValueError('Versioned campaign required')
    old_config=c.REPO/'configs/e2e_coupled_data_v1.json'
    old=json.loads(old_config.read_text());physical_equivalence(old,cfg)
    if phase not in old['phase_roles']:raise ValueError('No historical phase')
    source=Path(cfg['reuse_source_root']);old_hash=c.sha256(old_config)
    root=c.ROOT/'input_adoption'/phase
    with c.single_writer(root):
        marker=root/'BASE_INPUTS_COMPLETE.json'
        if marker.exists():
            record=c.verify_receipt(marker,payload=False)
            for item in record['receipts']:
                if c.sha256(item['path'])!=item['sha256']:raise ValueError('Adopted receipt drift')
            return record
        relative=[f'matter/{phase}/DENSITY_COMPLETE.json',f'observations/{phase}/PARENT_COMPLETE.json',
                  f'observations/{phase}/OBSERVED_COMPLETE.json',f'observations/{phase}/angular/ANGULAR_COMPLETE.json',
                  f'cartesian_v2/observations/{phase}/NGC_COMPLETE.json',f'cartesian_v2/observations/{phase}/SGC_COMPLETE.json']
        records=[];hashes={};verified={}
        for rel in relative:
            src=source/rel;dst=c.ROOT/rel;value=json.loads(src.read_text());source_hash=c.sha256(src)
            if value['config_sha256']!=old_hash or not value['pass'] or value['phase']!=phase:
                raise ValueError('Historical receipt authority/identity failed')
            if 'coordinate_sha256' in value and value['coordinate_sha256']!=c.sha256(coord.CONFIG):
                raise ValueError('Historical coordinate drift')
            if value.get('target_columns_present'):raise ValueError('Target leakage in observation source')
            for output in value.get('outputs',[]):
                path=c.guarded(output['path'],phase)
                if path not in verified:
                    if path.stat().st_size!=output['bytes'] or c.sha256(path)!=output['sha256']:
                        raise ValueError('Historical payload changed: '+str(path))
                    verified[path]=output['sha256']
                elif verified[path]!=output['sha256']:raise ValueError('Inconsistent payload hash')
            if rel.startswith('matter/'):
                if (value['ngrid']!=2048 or value['box_mpc_h']!=2000 or value['subsample_fraction']!=.1
                    or value['processed_files']!=136 or value['relative_count_error']>1e-5):
                    raise ValueError('Native density provenance failed')
            if '/observations/' in rel and 'grid' in value:coord.validate_grid(value['grid'])
            reused=copy.deepcopy(value)
            reused.update(c.provenance())
            if 'role' in reused:reused['role']=c.ROLES[phase]
            if 'parent_receipt_sha256' in reused:
                expected=c.sha256(source/f'observations/{phase}/PARENT_COMPLETE.json')
                if reused['parent_receipt_sha256']!=expected:raise ValueError('Parent lineage drift')
                reused['parent_receipt_sha256']=hashes[expected]
            reused['reuse']=dict(source_receipt=str(src),source_receipt_sha256=source_hash,
                original_role=value.get('role'),original_config_sha256=old_hash,
                scope='Fresh full output payload hashes and physical config parity; inherited historical raw-source validation',
                adopter_sha256=c.sha256(__file__))
            if dst.exists():
                existing=c.verify_receipt(dst)
                if existing.get('reuse',{}).get('source_receipt_sha256')!=source_hash:raise ValueError('Conflicting adopted receipt')
            else:c.atomic_json(dst,reused)
            hashes[source_hash]=c.sha256(dst);records.append(c.file_record(dst,content_hash=True))
        result=dict(**c.provenance(),phase=phase,role=c.ROLES[phase],receipts=records,
                    outputs=[],geometry_and_targets_adopted=False,**{'pass':True})
        c.atomic_json(marker,result)
    print(json.dumps(dict(phase=phase,base_inputs_adopted=True,payloads_verified=len(verified))),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phases',nargs='+',required=True)
    args=p.parse_args()
    for phase in args.phases:adopt(phase)
