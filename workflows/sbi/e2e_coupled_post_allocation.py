"""Exactly one approved CPU product-development allocation after B transfer."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_coupled_prepare_ops import accounting


def require_planned_terminal(after_job):
    result=subprocess.run(['sacct','-X','-n','-P','-j',str(after_job),
                           '-o','JobIDRaw,State,ExitCode'],check=True,text=True,capture_output=True)
    row=next((line.split('|') for line in result.stdout.splitlines()
              if line.split('|')[0]==str(after_job)),None)
    if row is None or (row[1],row[2]) not in (('COMPLETED','0:0'),('FAILED','75:0')):
        raise RuntimeError('predecessor is not a verified success/planned checkpoint pause: '+str(row))


def run(after_job,snapshot,step_name='e2e_coupled_post_step.sh',
        job_name='coupled-products-01',require_clean=False,wait_seconds=7200):
    if not 0 < wait_seconds <= 14400:
        raise ValueError('bounded allocation wait must be positive and at most four hours')
    if not re.fullmatch(r'e2e_coupled_[a-z0-9_]+\.sh',step_name):
        raise ValueError('bounded coupled preparation shell name required')
    if not re.fullmatch(r'coupled-[a-z0-9-]+',job_name): raise ValueError('invalid preparation job name')
    script=snapshot/'workflows/sbi'/step_name
    if not script.exists() or not (snapshot/'SOURCE.json').exists():
        raise ValueError('frozen product source missing')
    deadline=time.monotonic()+wait_seconds
    while True:
        rows=[line.split('|') for line in subprocess.run(
            ['squeue','-h','-u','dkololgi','-o','%i|%T'],check=True,text=True,
            capture_output=True).stdout.splitlines() if line.strip()]
        if all(row[0]!=after_job for row in rows) and len(rows)<2: break
        if time.monotonic()>deadline: raise RuntimeError('bounded allocation wait expired; no job requested')
        print(json.dumps(dict(waiting_for=after_job,jobs=rows)),flush=True); time.sleep(60)
    if require_clean: require_planned_terminal(after_job)
    usage=accounting()
    charged=sum(row['hours_cap'] if row['state'] in ('RUNNING','PENDING','COMPLETING') else
                row['elapsed_seconds']/3600 for row in usage['allocations'] if row['kind']=='cpu')
    if charged+4>c.config()['approval']['cpu_node_hours']:
        raise RuntimeError('new product allocation would exceed CPU budget')
    # One request, no retries, no cancellation, no automatic successor. The
    # concurrently live preparation controller is already inside its sole job.
    command=['salloc','--nodes=1','--ntasks=1','--cpus-per-task=128','--mem=0',
        '--constraint=cpu','--qos=interactive','--time=04:00:00','--account=desi',
        '--licenses=scratch','--immediate=600','--job-name='+job_name,
        '/bin/bash',str(script),str(snapshot)]
    print(json.dumps(dict(command=command,usage=usage)),flush=True)
    return subprocess.run(command,check=False).returncode


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after-job',required=True); p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--step-name',default='e2e_coupled_post_step.sh')
    p.add_argument('--job-name',default='coupled-products-01')
    p.add_argument('--require-clean-predecessor',action='store_true')
    p.add_argument('--wait-seconds',type=float,default=7200)
    a=p.parse_args()
    with c.single_writer(c.ROOT/'ops'/('controller_'+a.job_name)):
        raise SystemExit(run(a.after_job,a.snapshot,a.step_name,a.job_name,a.require_clean_predecessor,a.wait_seconds))
