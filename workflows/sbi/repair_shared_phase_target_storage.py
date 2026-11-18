"""Preserve pre-audit metadata snapshots while isolating adopted target payloads.

Applies only to this unqualified new campaign, never historical or audited data.
No numerical data changes. Copies hash-match source bytes before atomic receipts.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord


def repair(phase):
    c.require_compute();c.phase_guard(phase)
    if not c.SHARED_POLICY:raise ValueError('Shared campaign required')
    directory=coord.ROOT/'targets'/phase;panel=coord.ROOT/'panel_adoption'/phase/'PANEL_ADOPTED.json'
    if (coord.ROOT/'product_audit'/phase/'LATEST_AUDIT.json').exists():raise ValueError('Cannot change qualified products')
    proof=coord.verify_receipt(panel,payload=False)
    if proof.get('audit_required') is not True:raise ValueError('Not an unqualified adopted panel')
    history=directory/'storage_isolation_repair_v1'
    with c.single_writer(directory):
        done=history/'COMPLETE.json'
        if done.exists():return coord.verify_receipt(done,payload=False)
        history.mkdir(exist_ok=True)
        changed={};copies={}
        def save(path,value):
            before=c.file_record(path,content_hash=True)
            backup=history/(path.name+'.'+before['sha256']+'.json')
            if not backup.exists():shutil.copy2(path,backup)
            if c.sha256(backup)!=before['sha256']:raise ValueError('Metadata backup mismatch')
            c.atomic_json(path,value,replace=True)
            after=c.file_record(path,content_hash=True);changed[str(path)]=after
            return dict(before=before,backup=str(backup),after=after)
        complete=coord.verify_receipt(directory/'TARGETS_COMPLETE.json',payload=False)
        updates=[]
        for item in complete['pair_receipts']:
            path=Path(item['path'])
            if path.parent!=directory or c.sha256(path)!=item['sha256']:raise ValueError('Target pair receipt drift')
            value=coord.verify_receipt(path,payload=False)
            if 'reuse' not in value:raise ValueError('Not an adopted target pair')
            outputs=[]
            for output in value['outputs']:
                src=c.guarded(output['path'],phase);dst=directory/src.name
                if src.parent==directory:
                    if c.sha256(src)!=output['sha256']:raise ValueError('Already isolated target drift')
                    outputs.append(output);continue
                allowed=Path(c.config()['reuse_source_root'])/'cartesian_v2/targets'/phase
                if src.parent!=allowed:raise ValueError('Unexpected historical target source')
                if src not in copies:
                    if c.sha256(src)!=output['sha256']:raise ValueError('Historical target drift')
                    if not dst.exists():shutil.copy2(src,dst)
                    if dst.stat().st_size!=output['bytes'] or c.sha256(dst)!=output['sha256']:raise ValueError('Target copy mismatch')
                    copies[src]=c.file_record(dst,content_hash=True)
                outputs.append(copies[src])
            value['outputs']=outputs
            value['storage_isolation']=dict(source_outputs=value.get('storage_isolation',{}).get('source_outputs',coord.verify_receipt(path,payload=False)['outputs']),
                repair_source_sha256=c.sha256(__file__),numerical_bytes_unchanged=True)
            updates.append(save(path,value))
        complete['pair_receipts']=[dict(path=item['path'],sha256=changed[item['path']]['sha256']) for item in complete['pair_receipts']]
        complete['storage_isolation']=dict(repair_source_sha256=c.sha256(__file__),numerical_bytes_unchanged=True)
        updates.append(save(directory/'TARGETS_COMPLETE.json',complete))
        proof['receipts']=[changed.get(item['path'],item) for item in proof['receipts']]
        proof['target_storage_repair']=str(done)
        updates.append(save(panel,proof))
        result=dict(**coord.provenance(),phase=phase,updates=updates,copied_payloads=len(copies),
            outputs=[],numerical_bytes_unchanged=True,**{'pass':True})
        c.atomic_json(done,result)
    print(json.dumps(dict(phase=phase,targets_isolated=True,copies=len(copies))),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phases',nargs='+',required=True);a=p.parse_args()
    for phase in a.phases:repair(phase)
