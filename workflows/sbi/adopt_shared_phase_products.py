"""Rebind unchanged-role CFM products with explicit historical provenance.

Only geometry, target and condition panels whose roles and quotas are unchanged
are eligible. Condition and target payloads are copied privately to preserve each reader's
strict directory isolation.
This does not adopt fitted normalization or historical audit green flags.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi.adopt_shared_phase_base import physical_equivalence


def adopt(phase):
    c.require_compute();c.phase_guard(phase);coord.bind();coord.require_host_checks()
    cfg=c.config();old_config=c.REPO/'configs/e2e_coupled_data_v1.json'
    old=json.loads(old_config.read_text());physical_equivalence(old,cfg)
    if not c.SHARED_POLICY or old['phase_roles'].get(phase)!=c.ROLES[phase]:
        raise ValueError('Only unchanged-role historical panels may be adopted')
    source=Path(cfg['reuse_source_root']);old_hash=c.sha256(old_config)
    base=c.ROOT/'input_adoption'/phase/'BASE_INPUTS_COMPLETE.json'
    base_record=c.verify_receipt(base,payload=False)
    hash_map={};record_map={};payloads={}
    for item in base_record['receipts']:
        if c.sha256(item['path'])!=item['sha256']:raise ValueError('Base receipt drift')
        new=c.verify_receipt(item['path'],payload=False)
        prior=Path(new['reuse']['source_receipt']);digest=c.sha256(prior)
        if digest!=new['reuse']['source_receipt_sha256']:raise ValueError('Historical base receipt drift')
        hash_map[digest]=item['sha256'];record_map[str(prior)]=item
    dest_root=coord.ROOT/'panel_adoption'/phase
    with c.single_writer(dest_root):
        final=dest_root/'PANEL_ADOPTED.json'
        if final.exists():return c.verify_receipt(final,payload=False)
        def read(path):
            value=json.loads(path.read_text())
            if value.get('config_sha256')!=old_hash or not value.get('pass') or value.get('phase')!=phase:
                raise ValueError('Historical panel authority mismatch: '+str(path))
            if value.get('coordinate_sha256')!=c.sha256(coord.CONFIG):raise ValueError('Coordinate authority mismatch')
            if 'role' in value and value['role']!=c.ROLES[phase]:raise ValueError('Role changed')
            return value
        def rewrite(value):
            if isinstance(value,list):return [rewrite(v) for v in value]
            if isinstance(value,dict):
                if value.get('path') in record_map:
                    ref=record_map[value['path']]
                    if value.get('sha256') not in (None,c.sha256(value['path'])):raise ValueError('Receipt reference hash mismatch')
                    return {**value,**{k:ref[k] for k in ('path','sha256')},**{k:ref[k] for k in ('bytes','mtime_ns') if k in value}}
                return {k:rewrite(v) for k,v in value.items()}
            if isinstance(value,str):return hash_map.get(value,value)
            return value
        def publish(src,override=None):
            prior=read(src);digest=c.sha256(src);dest=c.ROOT/src.relative_to(source)
            value=rewrite(copy.deepcopy(prior));value.update(coord.provenance())
            if override:value.update(override)
            outputs=[]
            for item in prior.get('outputs',[]):
                path=c.guarded(item['path'],phase)
                if path not in payloads:
                    if path.stat().st_size!=item['bytes'] or c.sha256(path)!=item['sha256']:raise ValueError('Panel payload changed')
                    payloads[path]=item['sha256']
                elif payloads[path]!=item['sha256']:raise ValueError('Conflicting payload hashes')
                if '/conditions/' in str(src) or '/targets/' in str(src):
                    copied=dest.parent/path.name;copied.parent.mkdir(parents=True,exist_ok=True)
                    if not copied.exists():shutil.copy2(path,copied)
                    if copied.stat().st_size!=item['bytes'] or c.sha256(copied)!=item['sha256']:raise ValueError('Private condition copy mismatch')
                    outputs.append(c.file_record(copied,content_hash=True))
                else:outputs.append(item)
            if 'outputs' in value:value['outputs']=outputs
            value['reuse']=dict(source_receipt=str(src),source_receipt_sha256=digest,
                scope='Unchanged role/geometry and full payload hashes; current independent audit still required',adopter_sha256=c.sha256(__file__))
            if dest.exists():
                existing=coord.verify_receipt(dest,payload=False)
                if existing.get('reuse',{}).get('source_receipt_sha256')!=digest:raise ValueError('Conflicting panel receipt')
            else:c.atomic_json(dest,value)
            rec=c.file_record(dest,content_hash=True);record_map[str(src)]=rec;hash_map[digest]=rec['sha256']
            return rec
        geo=source/f'cartesian_v2/geometry/{phase}/GEOMETRY_COMPLETE.json'
        old_geo=read(geo);geometry.verify_rows(old_geo['pairs'],phase)
        if old_geo['counts_or_targets_read'] is not False:raise ValueError('Geometry contains target selection')
        sources={}
        for cap in ('NGC','SGC'):
            ref=old_geo['sources'][cap];path=Path(ref['receipt'])
            if c.sha256(path)!=ref['sha256'] or str(path) not in record_map:raise ValueError('Geometry observation lineage mismatch')
            sources[cap]=dict(receipt=record_map[str(path)]['path'],sha256=record_map[str(path)]['sha256'])
        publish(geo,dict(sources=sources))
        for category,marker in [('targets','TARGETS_COMPLETE.json'),('conditions','CONDITIONS_COMPLETE.json')]:
            folder=source/f'cartesian_v2/{category}/{phase}';complete=read(folder/marker)
            if category=='targets':
                from workflows.sbi import e2e_coupled_operators as op
                from workflows.sbi import e2e_field_error_budget as native_ops
                for key,path in [('operators_sha256',op.__file__),('native_operator_sha256',native_ops.__file__),('layout_sha256',op.LAYOUT)]:
                    if complete['binding'][key]!=c.sha256(path):raise ValueError('Historical target operator changed: '+key)
                for item in complete['stage_receipts']:
                    if c.sha256(item['path'])!=item['sha256']:raise ValueError('Target stage drift')
                    publish(Path(item['path']))
            else:
                for cap in ('NGC','SGC'):publish(folder/f'{cap}_FACTOR4.json')
            expected={r['pair_id'] for r in old_geo['pairs']}
            if {Path(item['path']).stem for item in complete['pair_receipts']}!=expected or len(complete['pair_receipts'])!=len(expected):
                raise ValueError('Pair panel missing or duplicated')
            for item in complete['pair_receipts']:
                path=Path(item['path'])
                if path.parent!=folder or c.sha256(path)!=item['sha256']:raise ValueError('Pair lineage mismatch')
                publish(path)
            publish(folder/marker)
        result=dict(**coord.provenance(),phase=phase,role=c.ROLES[phase],base_receipt=c.file_record(base,content_hash=True),
            receipts=list(record_map.values()),payloads_hashed=len(payloads),outputs=[],audit_required=True,**{'pass':True})
        c.atomic_json(final,result)
    print(json.dumps(dict(phase=phase,panel_adopted=True,payloads=len(payloads))),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phases',nargs='+',required=True)
    a=p.parse_args()
    for phase in a.phases:adopt(phase)
