"""Single-use CPU continuation: remaining phase products or physical gate.

No retry. Requested CPU ceilings are counted conservatively before submission.
The product continuation uses the already frozen builder, not live code.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from workflows.sbi.e2e_vdm_context_launch import PYTHON,check_root,digest,publish


def launch(root,mode):
    root=check_root(root)
    if mode not in ('remaining','physics_v2'):
        raise ValueError('unregistered CPU stage')
    request=root/(mode.upper()+'_REQUEST.json')
    result=root/(mode.upper()+'_RETURN.json')
    if request.exists() or result.exists():
        raise FileExistsError('single-use CPU launch already attempted')
    source_record=json.loads((root/('BUILD_SOURCE.json' if mode=='remaining' else 'PHYSICS_V2_SOURCE.json')).read_text())
    source=Path(source_record['source'])
    for name,expected in source_record['source_sha256'].items():
        if digest(source/name)!=expected:
            raise ValueError('frozen CPU source drift')
    if mode=='physics_v2':
        for phase in ('ph000','ph002'):
            if not (root/'data'/phase/'COMPLETE.json').is_file():
                raise ValueError('both A32 phase products required')
    hours=2 if mode=='remaining' else .5
    used=sum(json.loads(p.read_text()).get('cpu_node_hours_limit',0)
             for p in root.glob('*_REQUEST.json'))
    if used+hours>8:
        raise RuntimeError('CPU requested-time ceiling exhausted')
    initial=json.loads((root/'SCREEN_REQUEST.json').read_text())
    if time.time()+hours*3600>initial['deadline_epoch']:
        raise RuntimeError('experiment elapsed ceiling insufficient for request')
    listing=subprocess.check_output(['squeue','--me','-h','-o','%i|%q'],text=True)
    if sum('interactive' in row for row in listing.splitlines())>=2:
        raise RuntimeError('two interactive allocations already pending/running')
    cmd=['salloc','--nodes=1','--ntasks=1','--cpus-per-task=64','--constraint=cpu',
         '--qos=interactive','--account=desi','--licenses=scratch','--immediate=600',
         '--time=02:00:00' if mode=='remaining' else '--time=00:30:00',
         '--job-name=vdm-context-'+mode,'srun','--nodes=1','--ntasks=1','--cpus-per-task=64',
         '--cpu-bind=cores',PYTHON,'-u','-m']
    cmd+=(['workflows.sbi.e2e_vdm_context_products','--root',str(root),'--phases','ph003','ph004','ph005']
          if mode=='remaining' else ['workflows.sbi.e2e_vdm_context_physics','--root',str(root)])
    start=time.time()
    publish(request,dict(command=cmd,epoch=start,cpu_node_hours_limit=hours,
                         prior_requested_cpu_node_hours=used,deadline_epoch=initial['deadline_epoch'],
                         launcher_sha256=digest(__file__)))
    env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','LD_PRELOAD'):
        env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1')
    with (root/'logs'/f'{mode}.log').open('x') as log:
        status=subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    publish(result,dict(exit_code=status,elapsed_seconds=time.time()-start,no_automatic_retry=True))
    raise SystemExit(status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--mode',choices=['remaining','physics_v2'],required=True)
    a=p.parse_args()
    launch(a.root,a.mode)


if __name__=='__main__':
    main()
