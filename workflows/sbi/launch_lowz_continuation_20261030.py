"""Fixed approved CFM-only window; tmux owns salloc, Slurm owns compute."""
import json
import os
import subprocess
from workflows.sbi.launch_lowz_replacement import PYTHON, REPO, ROOT


def main(approval_flag='--continuation-20261030'):
    if approval_flag not in ('--continuation-20261030','--continuation-20261102'):
        raise PermissionError('Unknown continuation approval')
    job=os.environ['SLURM_JOB_ID']
    directory=ROOT/('lowz_continuation_'+job)
    directory.mkdir(exist_ok=False)
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','LD_PRELOAD'):
        env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='32',
               GRAPHWEB_CFM_DATA_CONFIG='configs/e2e_coupled_data_lowz_20261029.json')
    command=['srun','--jobid='+job,'-N','1','-n','1','-c','64','--exact',
             '--cpu-bind=cores','--export=ALL','timeout','14100',PYTHON,'-u','-m',
             'workflows.sbi.run_lowz_completion_window','--seconds','13800',
             '--mode','campaign','--cached-geometry',approval_flag]
    print(json.dumps(dict(job=job,command=command,log=str(directory/'cfm.log'))),flush=True)
    with (directory/'cfm.log').open('x') as log:
        result=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    receipt=dict(job=job,exitcode=result.returncode,actual_process_waited=True,
                 no_retries=True,training_started=False)
    (directory/'PROCESS_COMPLETION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
