"""Request exactly one approved four-hour CPU continuation after the smoke job.

This is a deterministic controller, not an autonomous agent. No automatic retry,
no third allocation, no cancellation, and no model-training entrypoint.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_coupled_prepare_ops import accounting


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after-job',required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    a=p.parse_args()
    deadline=time.monotonic()+7200
    while True:
        result=subprocess.run(['squeue','-h','-u','dkololgi','-o','%i|%T'],
                              check=True,text=True,capture_output=True)
        rows=[line.split('|') for line in result.stdout.splitlines() if line.strip()]
        if all(row[0]!=a.after_job for row in rows) and len(rows)<2:
            break
        if time.monotonic()>deadline:
            raise RuntimeError('bounded wait for previous allocation expired; no request issued')
        print(json.dumps(dict(waiting_for=a.after_job,jobs=rows)),flush=True)
        time.sleep(60)
    usage=accounting()
    # Include full reserved CPU hours for any still-live registered allocation.
    charged=sum(row['hours_cap'] if row['state'] in ('RUNNING','PENDING','COMPLETING')
                else row['elapsed_seconds']/3600 for row in usage['allocations'] if row['kind']=='cpu')
    if charged+4>c.config()['approval']['cpu_node_hours']:
        raise RuntimeError('four-hour continuation would exceed approved CPU cap')
    script=a.snapshot/'workflows/sbi/e2e_coupled_cpu_step.sh'
    if not script.exists():
        raise ValueError('frozen CPU step script missing')
    command=['salloc','--nodes=1','--ntasks=1','--cpus-per-task=128','--mem=0',
             '--constraint=cpu','--qos=interactive','--time=04:00:00','--account=desi',
             '--licenses=scratch','--immediate=600','--job-name=coupled-prep-02',
             '/bin/bash',str(script),str(a.snapshot)]
    print(json.dumps(dict(command=command,usage=usage)),flush=True)
    return subprocess.run(command,check=False).returncode


def main():
    # Keep this lock file permanently; unlinking a live lock enables duplicate
    # writers on a shared filesystem. At most one continuation controller.
    with c.single_writer(c.ROOT/'ops/continuation_controller'):
        return run()


if __name__=='__main__':
    raise SystemExit(main())
