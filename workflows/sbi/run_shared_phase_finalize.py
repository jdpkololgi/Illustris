"""Bounded dependent finalization for the approved shared-phase CPU preparation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from workflows.sbi import e2e_coupled_contract as c


def wait_for(paths,deadline):
    while not all(p.is_file() for p in paths):
        if time.monotonic()>deadline-600:raise TimeoutError('Prerequisite receipts incomplete within preparation budget')
        time.sleep(30)


def run(action,seconds):
    c.require_compute()
    if not c.SHARED_POLICY or not 0<seconds<=4800:raise ValueError('Bounded shared campaign required')
    deadline=time.monotonic()+seconds
    phases=list(c.TRAIN)+list(c.DEVELOPMENT)
    if len(phases)!=23 or set(phases)&{'ph018','ph019'}:raise ValueError('Preparation phase policy mismatch')
    if action=='wholefield':
        from workflows.sbi import prepare_shared_wholefield as w
        from workflows.sbi import e2e_coupled_coordinates as coord
        wait_for([coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json'],deadline)
        for phase in phases:
            rows,_=w.rows_for(phase)
            wait_for([w.data.ROOT/'data'/phase/r['pair_id']/f'{kind}.json' for r in rows for kind in ('OBSERVATIONS','TARGETS')],deadline)
            w.finalize(phase)
    else:
        registry=c.REPO/'configs/p10_phase_registry_v3.json'
        value=json.loads(registry.read_text())
        root=Path(value['path_templates']['phase_output'].format(phase='ph000')).parent
        contract=root/'training_contract_v1'
        wait_for([root/ph/'PHASE_COMPLETE.json' for ph in phases]+[contract/'adapters'/ph/k/'adapter_manifest.json' for ph in phases for k in ('graph','field')],deadline)
        subprocess.run([sys.executable,'-u','-m','workflows.abacus_tweb.p10_prepare_training_contract','--root',str(root),'--registry',str(registry),'--contract-root',str(contract),'--stage','all'],check=True)
        if time.monotonic()>deadline-900:raise TimeoutError('Full contract ready; cross-fit needs another bounded window')
        subprocess.run([sys.executable,'-u','-m','workflows.sbi.p12_prepare_crossfit_contracts','--source-root',str(root),'--full-contract',str(contract),'--output-root',str(root/'crossfit_v1'),'--workers','2'],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['wholefield','p12']);p.add_argument('--seconds',type=float,default=4200);a=p.parse_args();run(a.action,a.seconds)
