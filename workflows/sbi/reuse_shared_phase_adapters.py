"""Qualify unchanged raw adapters; never reuse fitted transforms.

Stage private metadata and immutable payload links away from active builders.
Publish only still-absent adapter directories, in reverse serial-driver order;
the active and immediately next serial phases are always left untouched.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
from workflows.sbi import e2e_coupled_contract as c
from workflows.abacus_tweb.p10_prepare_training_contract import phase_files
from shared.phase_policy import check_preparation_phase

ORDER=['ph000','ph002','ph003','ph004','ph005','ph007','ph008','ph009','ph010','ph011','ph020','ph021','ph022','ph023','ph024','ph006','ph012','ph013']
ALLOWED=ORDER[5:]


def active_index(log):
    commands=[json.loads(line)['command'] for line in log.read_text().splitlines() if line.startswith('{"command":')]
    if not commands:raise ValueError('Cannot identify active serial adapter phase')
    phases=set(re.findall(r'/(ph\d{3})/', ' '.join(commands[-1])))
    if len(phases)!=1:raise ValueError('Ambiguous active adapter phase')
    return ORDER.index(phases.pop())


def run():
    c.require_compute()
    registry=json.loads((c.REPO/'configs/p10_phase_registry_v3.json').read_text())
    inputs=json.loads((c.REPO/'configs/p12a_corrected_expanded_inputs_20261026.json').read_text())
    old=Path(inputs['full_contract']);source_root=old.parent
    ready=old/'TRAINING_LOADER_READY.json'
    if c.sha256(ready)!=inputs['full_contract_sha256']:raise ValueError('Historical full contract drift')
    proof=json.loads(ready.read_text());inventory=Path(proof['adapter_inventory'])
    if not proof['pass'] or c.sha256(inventory)!=proof['adapter_inventory_sha256']:raise ValueError('Historical adapter inventory drift')
    inventory=json.loads(inventory.read_text())['phases']
    root=Path(registry['path_templates']['phase_output'].format(phase='ph000')).parent
    contract=root/'training_contract_v1';log=root/f'adapters_existing_{os.environ["SLURM_JOB_ID"]}.log'
    staging=contract/f'adapter_reuse_{os.environ["SLURM_JOB_ID"]}'
    for phase in reversed(ALLOWED):
        check_preparation_phase(registry,phase);dest=contract/'adapters'/phase
        if dest.exists() or dest.is_symlink() or ORDER.index(phase)<=active_index(log)+1:
            print(json.dumps(dict(phase=phase,adapter_reuse_skipped='active, next, or already present')),flush=True);continue
        verified={}
        def verify(path,expected=None):
            p=Path(path);key=str(p.resolve())
            if key not in verified:verified[key]=c.sha256(p)
            if expected is not None and verified[key]!=expected:raise ValueError('Adapter source drift: '+str(p))
            return verified[key]
        before=phase_files(source_root,phase);after=phase_files(root,phase)
        for key in before:
            if key=='p4_root':continue
            if verify(before[key])!=verify(after[key]):raise ValueError('Adapter input changed: '+phase+' '+key)
        base=staging/phase
        for kind in ('graph','field'):
            item=inventory[phase];path=Path(item[kind+'_manifest']);verify(path,item[kind+'_manifest_sha256'])
            value=json.loads(path.read_text())
            if not value['pass']:raise ValueError('Historical adapter failed')
            if kind=='graph':
                for artifact in value['artifacts'].values():verify(artifact['path'],artifact['sha256'])
                for key,expected in value['inputs'].items():
                    if key.endswith('_sha256'):verify(value['inputs'][key[:-7]],expected)
            else:
                for cap in value['caps'].values():verify(cap['field_path'],cap['field_sha256'])
                for key in ('p3_manifest','p4_active_assignment','p4_cores'):verify(value[key],value[key+'_sha256'])
            folder=base/kind;folder.mkdir(parents=True,exist_ok=True)
            for src in path.parent.iterdir():
                target=folder/src.name
                if target.exists() or target.is_symlink():continue
                if src.suffix=='.json':shutil.copy2(src,target)
                elif src.is_file():target.symlink_to(src)
                else:raise ValueError('Unexpected nested adapter directory')
        record=dict(phase=phase,phase_policy=registry['phase_policy'],source_contract=str(ready),
                    source_contract_sha256=inputs['full_contract_sha256'],verified_sha256=verified,
                    fitted_transforms_reused=False,source_sha256=c.sha256(__file__),**{'pass':True})
        marker=base/'RAW_ADAPTER_REUSE_COMPLETE.json'
        if not marker.exists():c.atomic_json(marker,record)
        if dest.exists() or dest.is_symlink() or ORDER.index(phase)<=active_index(log)+1:
            print(json.dumps(dict(phase=phase,adapter_staged_only=True)),flush=True);continue
        try:dest.symlink_to(base,target_is_directory=True)
        except FileExistsError:
            print(json.dumps(dict(phase=phase,adapter_staged_only=True)),flush=True);continue
        print(json.dumps(dict(phase=phase,raw_adapters_reused=True,verified_files=len(verified))),flush=True)

if __name__=='__main__':run()
