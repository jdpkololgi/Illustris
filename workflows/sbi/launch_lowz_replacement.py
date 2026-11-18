"""Finite login-side orchestration inside the explicitly approved salloc.

All project compute runs through srun. Wait on the actual child processes; never
infer completion from scheduler-listing text. No resubmission or retry loop.
"""
import json
import os
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[2]
PYTHON='/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python'
ROOT=Path('/pscratch/sd/d/dkololgi/abacus')


def wait_for_all(processes):
    return {name:process.wait() for name,process in processes.items()}


def main():
    job=os.environ['SLURM_JOB_ID']
    directory=ROOT/('lowz_replacement_'+job)
    directory.mkdir(exist_ok=False)
    environment=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','LD_PRELOAD'):
        environment.pop(key,None)
    environment.update(PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1')
    commands={
        'p12':[PYTHON,'-u','-m','workflows.sbi.prepare_p12a_lowz_training','--finish'],
        'cfm':[PYTHON,'-u','-m','workflows.sbi.run_lowz_completion_window',
               '--seconds','13700','--mode','campaign','--cached-geometry',
               '--replacement-job','60522523'],
    }
    streams={};processes={};launch_error=None
    try:
        for name,command in commands.items():
            env=environment.copy();env['OMP_NUM_THREADS']='8' if name=='p12' else '32'
            if name=='cfm':env['GRAPHWEB_CFM_DATA_CONFIG']='configs/e2e_coupled_data_lowz_20261029.json'
            streams[name]=(directory/(name+'.log')).open('x')
            full=['srun','--jobid='+job,'--nodes=1','--ntasks=1','--cpus-per-task=64',
                  '--exact','--cpu-bind=cores','--export=ALL','timeout','13800',*command]
            processes[name]=subprocess.Popen(full,cwd=REPO,env=env,stdout=streams[name],stderr=subprocess.STDOUT)
            print(json.dumps(dict(job=job,name=name,pid=processes[name].pid,command=full,log=str(directory/(name+'.log')))),flush=True)
    except Exception as error:
        launch_error=repr(error)
    finally:
        # Even launch errors must not release a successfully launched sibling.
        result=wait_for_all(processes)
        for stream in streams.values():stream.close()
        receipt=dict(job=job,process_exit_codes=result,launch_error=launch_error,
                     all_processes_waited=True,no_retries=True,training_started=False)
        (directory/'PROCESS_COMPLETION.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
    return int(launch_error is not None or len(result)!=2 or any(result.values()))


if __name__=='__main__':raise SystemExit(main())
