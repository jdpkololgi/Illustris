"""Canonicalize imported native-count bookkeeping without changing array bytes.

Only unqualified new-campaign imports are eligible. Replay both reductions,
require the original block reduction exactly, keep conservation gates unchanged,
back up every changed receipt, and refresh downstream hashes before re-audit.
"""
import argparse
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import shutil
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_wholefield_pilot_data as whole


def repair(phase):
    c.require_compute();c.phase_guard(phase)
    if not c.SHARED_POLICY or phase not in ('ph001','ph004','ph005','ph006'):raise ValueError('Only imported native grids')
    native_path=c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json'
    history=c.ROOT/'native_count_reduction_repair'/phase
    if (history/'COMPLETE.json').exists():return c.verify_receipt(history/'COMPLETE.json',payload=False)
    if (coord.ROOT/'product_audit'/phase/'LATEST_AUDIT.json').exists():raise ValueError('Never modify audited products')
    target_dir=coord.ROOT/'targets'/phase
    if not (target_dir/'TARGETS_COMPLETE.json').exists():raise ValueError('No repair during target construction')
    if (whole.ROOT/'data'/phase/'COMPLETE.json').exists():raise ValueError('Never modify finalized whole-field data')
    with ExitStack() as stack:
        for directory in (native_path.parent,target_dir,whole.ROOT/'data'/phase):stack.enter_context(c.single_writer(directory))
        native=c.verify_receipt(native_path,payload=False)
        if 'source_manifest' not in native or 'source_contract' not in native:raise ValueError('Not a P10 import')
        payload=native['outputs'][0];array=np.load(payload['path'],mmap_mode='r',allow_pickle=False)
        if array.shape!=(2048,)*3 or array.dtype!=np.float32:raise ValueError('Wrong native grid')
        full=float(array.sum(dtype=np.float64))
        if abs(full-native['particle_count'])/native['particle_count']>2e-6:raise ValueError('Native conservation fails')
        if abs(full-native['deposited_count'])<=64*np.spacing(full):
            print(json.dumps(dict(phase=phase,native_reduction_already_consistent=True)),flush=True);return
        block=0.
        for start in range(0,2048,8):
            slab=array[start:start+8]
            if not np.isfinite(slab).all() or np.min(slab)<0:raise ValueError('Invalid counts')
            block+=float(slab.sum(dtype=np.float64))
        del array
        if block!=native['deposited_count']:raise ValueError('Original block reduction does not replay exactly')
        if c.sha256(payload['path'])!=payload['sha256']:raise ValueError('Native payload changed')
        history.mkdir(parents=True,exist_ok=True);changes={};hashes={};backups={}
        def save(path,value):
            old=c.file_record(path,content_hash=True)
            if str(path) not in backups:
                backup=history/(path.name+'.'+old['sha256']+'.json');shutil.copy2(path,backup)
                if c.sha256(backup)!=old['sha256']:raise ValueError('Receipt backup failed')
                backups[str(path)]=dict(original=old,backup=str(backup))
            c.atomic_json(path,value,replace=True);new=c.file_record(path,content_hash=True)
            for key,digest in list(hashes.items()):
                if digest==old['sha256']:hashes[key]=new['sha256']
            hashes[old['sha256']]=new['sha256'];changes[str(path)]=new
        native['deposited_count']=full
        native['relative_count_error']=abs(full-native['particle_count'])/native['particle_count']
        native['count_reduction_correction']=dict(original_block_sum=block,full_array_float64_sum=full,
            numerical_payload_unchanged=True,repair_source_sha256=c.sha256(__file__))
        save(native_path,native)
        paths=sorted(target_dir.glob('*.json'))+sorted((whole.ROOT/'data'/phase).glob('*/TARGETS.json'))
        def rewrite(value):
            if isinstance(value,str):return hashes.get(value,value)
            if isinstance(value,list):return [rewrite(v) for v in value]
            if isinstance(value,dict):
                result={k:rewrite(v) for k,v in value.items()}
                if value.get('path') in changes:
                    item=changes[value['path']]
                    for key in ('sha256','bytes','mtime_ns'):
                        if key in result:result[key]=item[key]
                return result
            return value
        for iteration in range(8):
            updated=False
            for path in paths:
                old=json.loads(path.read_text());new=rewrite(old)
                if new!=old:save(path,new);updated=True
            if not updated:break
        else:raise ValueError('Metadata dependency graph did not converge')
        for path in paths:
            if rewrite(json.loads(path.read_text()))!=json.loads(path.read_text()):raise ValueError('Stale downstream metadata')
        result=dict(**c.provenance(),phase=phase,original_block_sum=block,full_array_sum=full,
            relative_count_error=native['relative_count_error'],unchanged_payload=payload,
            backups=backups,current_receipts=changes,source_sha256=c.sha256(__file__),
            unchanged_conservation_tolerance=2e-6,audit_required=True,outputs=[],**{'pass':True})
        c.atomic_json(history/'COMPLETE.json',result)
    print(json.dumps(dict(phase=phase,native_reduction_corrected=True,receipts=len(changes),difference=full-block)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phases',nargs='+',required=True);a=p.parse_args()
    for phase in a.phases:repair(phase)
