"""User-approved fixed resume, under existing allocation only; no retry loop."""
import json
import os
from pathlib import Path
import subprocess
from workflows.sbi.launch_lowz_replacement import wait_for_all,PYTHON,REPO,ROOT

def main():
    job='60522907'
    remaining=subprocess.check_output(['squeue','-h','-j',job,'-o','%L'],text=True).strip()
    parts=remaining.split(':')
    if len(parts)!=3:raise RuntimeError('Expected H:M:S remaining allocation time: '+remaining)
    seconds=sum(int(v)*m for v,m in zip(parts,[3600,60,1]))
    if not 600<seconds<14160:raise RuntimeError('Insufficient or invalid approved remaining time')
    directory=ROOT/'lowz_resume_60522907_20261030';directory.mkdir(exist_ok=False)
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','LD_PRELOAD'):env.pop(k,None)
    env.update(PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1')
    commands={
        'cfm':[PYTHON,'-u','-m','workflows.sbi.run_lowz_completion_window','--seconds',str(seconds-240),
               '--mode','campaign','--cached-geometry','--replacement-job','60522523'],
        'p12_figures':[PYTHON,'-u','-m','workflows.sbi.p12a_training_examples_20261030'],
    }
    children={};streams={};error=None
    try:
        for name,command in commands.items():
            child_env=env.copy();child_env['OMP_NUM_THREADS']='32' if name=='cfm' else '4'
            if name=='cfm':child_env['GRAPHWEB_CFM_DATA_CONFIG']='configs/e2e_coupled_data_lowz_20261029.json'
            streams[name]=(directory/(name+'.log')).open('x')
            args=['srun','--jobid='+job,'-N','1','-n','1','-c','64','--exact','--cpu-bind=cores',
                  'timeout',str(seconds-150 if name=='cfm' else min(1800,seconds-150)),*command]
            children[name]=subprocess.Popen(args,cwd=REPO,env=child_env,stdout=streams[name],stderr=subprocess.STDOUT)
            print(name,children[name].pid,flush=True)
    except Exception as exc:error=repr(exc)
    finally:
        codes=wait_for_all(children)
        for stream in streams.values():stream.close()
        (directory/'PROCESS_COMPLETION.json').write_text(json.dumps(dict(job=job,exit_codes=codes,error=error,
            actual_processes_waited=True,remaining_at_start=seconds,no_retries=True),indent=2)+'\n')
        # Release only this approved, now idle allocation after both own children.
        subprocess.run(['scancel',job],check=False)

if __name__=='__main__':main()
