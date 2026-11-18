"""Sequentially read a verified field on its annotation node; no data changes."""
import argparse
import json
import os
from pathlib import Path
import socket
import time
from shared.phase_policy import REPO, check_preparation_phase
from workflows.sbi.e2e_coupled_contract import require_compute, atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True);args=p.parse_args()
    require_compute()
    cfg=json.loads((REPO/'configs/phase_migration_20261027.json').read_text())
    check_preparation_phase(cfg,args.phase)
    root=Path(cfg['new_p12a_root'])/args.phase
    field=root/'targets/tweb/backend_optimized_ngrid_2048_rsmooth_7'
    receipt=json.loads((field/'TWEB_COMPLETE.json').read_text())
    if receipt['phase']!=args.phase or receipt['target_contract']['tidal_smoothing_mpc_h']!=7:
        raise ValueError('Wrong field identity')
    started=time.monotonic();total=0;buffer=bytearray(8<<20)
    for record in receipt['outputs']['records']:
        path=Path(record['path'])
        if path.resolve().parent!=field.resolve() or path.stat().st_size!=record['bytes']:
            raise ValueError('Field source differs')
        with path.open('rb',buffering=0) as stream:
            while count:=stream.readinto(buffer): total+=count
    result=dict(phase=args.phase,host=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
                bytes=total,elapsed_seconds=time.monotonic()-started,
                scope='Read-only sequential page-cache warming; not a new validation receipt')
    atomic_json(root/f"field_cache_warm_{os.environ['SLURM_JOB_ID']}_{socket.gethostname()}.json",result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
