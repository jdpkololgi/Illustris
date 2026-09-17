"""Finite authorized Slurm chain. No adaptive science, agent loop, or batch fallback.

The login process only verifies metadata and requests allocations. All model,
FFT, replay and reporting work runs in explicit srun compute-node steps.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from workflows.sbi.e2e_vdm_context_launch import PYTHON,check_root,digest,publish

STAGES=('smoke','train','freeze','refinement','sampler','all','report')
GPU_STAGES={'smoke','train','refinement','all'}
STOP=False
CHILD=None


def read(path):
    return json.loads(Path(path).read_text())


def environment():
    env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','LD_PRELOAD'):
        env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8',SLURM_EXPORT_ENV='ALL')
    return env


def manifest(root):
    m=read(root/'MANIFEST.json')
    source=Path(m['source']).resolve()
    if source!=Path(__file__).resolve().parents[2]:
        raise ValueError('controller must execute immutable full-run source')
    for name,expected in m['source_sha256'].items():
        if digest(source/name)!=expected:
            raise ValueError('frozen source drift: '+name)
    for name,expected in m['data_receipts'].items():
        if digest(root/name)!=expected:
            raise ValueError('frozen data receipt drift: '+name)
    return m


def gates(root,stage):
    m=manifest(root)
    if stage!='smoke':
        for name in ('SMOKE.json','RESTART_TEST.json'):
            gate=read(root/name)
            if not gate['passed'] or gate['manifest_sha256']!=digest(root/'MANIFEST.json'):
                raise PermissionError('matching GPU gate missing: '+name)
        if read(root/'SMOKE.json')['forecast_gpu_hours']>m['spec']['budget']['gpu_hours']:
            raise PermissionError('actual compute forecast exceeds budget')
    if stage in ('refinement','sampler','all','report'):
        frozen=read(root/'MODELS_FROZEN.json')
        if not frozen['all_models_frozen'] or frozen['manifest_sha256']!=digest(root/'MANIFEST.json'):
            raise PermissionError('full model matrix freeze missing')
    if stage in ('all','report') and not read(root/'analysis/SAMPLER_GATE.json')['passed']:
        raise PermissionError('new-checkpoint sampler gate failed')
    return m


def parse_accounting(text,expected):
    records={}
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL'}
    for line in text.splitlines():
        values=line.split('|')
        if len(values)<5 or values[0] not in expected:
            continue
        job,state,code,seconds,tres=values[:5]
        if state.split()[0] not in terminal:
            raise RuntimeError('prior allocation is not terminal: '+job)
        gpu=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
        count=int(gpu.group(1)) if gpu else 0
        if count!=expected[job]:
            raise ValueError('allocated GPU count differs from request')
        records[job]=dict(state=state,exit_code=code,seconds=int(seconds),gpus=count)
    if set(records)!=set(expected):
        raise RuntimeError('terminal scheduler accounting is incomplete; do not request again')
    return dict(jobs=records,gpu_hours=sum(r['seconds']*r['gpus']/3600 for r in records.values()),
                cpu_node_hours=sum(r['seconds']/3600 for r in records.values() if r['gpus']==0))


def accounting(root):
    expected={}
    for name in ('geometry','products','remaining','physics','physics_v2'):
        log=root/'logs'/(name+'.log')
        if not log.exists():
            # The initial geometry launcher used screen.log.
            log=root/'logs'/'screen.log' if name=='geometry' else log
        jobs=re.findall(r'Granted job allocation (\d+)',log.read_text())
        if len(jobs)!=1:
            raise ValueError('expected one historical CPU allocation: '+name)
        expected[jobs[0]]=0
    for path in (root/'resources').glob('*_START.json'):
        record=read(path)
        expected[record['job']]=record['gpus']
    output=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(expected),
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'],text=True)
    return parse_accounting(output,expected)


def request_size(stage,left,usage,budget,deadline,now):
    if stage not in STAGES:
        raise ValueError('unregistered stage')
    gpus=(1 if stage=='smoke' else (1 if left==1 else 2 if left==2 else 4)) if stage in GPU_STAGES else 0
    desired={'smoke':60,'train':240,'freeze':15,'refinement':60,'sampler':30,'all':240,'report':120}[stage]
    remaining=(budget['gpu_hours']-usage['gpu_hours'])/gpus if gpus else budget['cpu_node_hours']-usage['cpu_node_hours']
    minutes=min(desired,math.floor(remaining*60),math.floor((deadline-now-600)/60))
    if minutes<15:
        raise RuntimeError('remaining approved resource/elapsed budget cannot safely fit a stage')
    return gpus,minutes


def work_items(root,stage):
    if stage=='train':
        factors=[('D',s,'coarse') for s in (0,1)]+[(a,s,'fine') for a in 'ABCD' for s in (0,1)]
        return [dict(key=f'{a}_{f}_seed{s}',arm=a,seed=s,factor=f) for a,s,f in factors]
    if stage in ('refinement','all'):
        from workflows.sbi.e2e_vdm_context_sample import selected_tasks
        ledger=read(root/'DRAW_LEDGER.json')
        return [dict(key=t['task_id'],task=t,limit=limit) for a in 'ABCD' for s in (0,1)
                for t,limit in selected_tasks(ledger,a,s,stage)]
    return []


def item_done(root,stage,item):
    if stage=='train':
        path=root/'models'/item['key']/'COMPLETE.json'
        if not path.exists():
            return False
        complete=read(path)
        pointer=read(path.parent/'LATEST.json')
        if (complete['updates']!=20480 or complete['examples_seen']!=40960 or complete['checkpoint']!=pointer
                or complete['binding']['manifest_sha256']!=digest(root/'MANIFEST.json')):
            raise ValueError('model completion drift')
        return True
    from workflows.sbi.e2e_vdm_context_queue import task_done
    return task_done(root,item['task'],item['limit'])


def remaining(root,stage):
    return [item for item in work_items(root,stage) if not item_done(root,stage,item)]


def child_command(root,stage,item):
    cmd=[PYTHON,'-u','-m']
    if stage=='train':
        return cmd+['workflows.sbi.e2e_vdm_context_train','--root',str(root),
                    '--arm',item['arm'],'--seed',str(item['seed']),'--factor',item['factor']]
    task=item['task']
    return cmd+['workflows.sbi.e2e_vdm_context_sample','--root',str(root),'--arm',task['arm'],
                '--replica',str(task['replica']),'--mode',stage,'--task-id',task['task_id']]


def stop(signum,frame):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGUSR1)


def worker(root,index,slot):
    global STOP,CHILD
    from workflows.sbi.e2e_vdm_context_queue import task_lock,disk_bytes
    request=read(root/'resources'/f'{index:02d}_REQUEST.json')
    start=read(root/'resources'/f'{index:02d}_START.json')
    stage=request['stage']
    gates(root,stage)
    if os.environ.get('SLURM_JOB_ID')!=start['job']:
        raise ValueError('worker outside its registered compute allocation')
    signal.signal(signal.SIGUSR1,stop)
    signal.signal(signal.SIGTERM,stop)
    deadline=start['started_unix']+request['minutes']*60-600
    abort=root/'resources'/f'{index:02d}_ABORT.json'
    for item in work_items(root,stage):
        if STOP or time.time()>=deadline or abort.exists():
            return 75
        with task_lock(root/'queue'/stage/item['key']) as acquired:
            if not acquired or item_done(root,stage,item):
                continue
            # Four fits can each add ~8GiB of durable generations. Reserve32GiB
            # before starting a work item, never write up to the hard300GiB edge.
            if disk_bytes(root)>(300-32)*2**30:
                raise RuntimeError('storage safety headroom reached')
            log=root/'logs'/f'{index:02d}_{slot}_{item["key"]}.log'
            with log.open('x') as stream:
                CHILD=subprocess.Popen(child_command(root,stage,item),env=environment(),
                    cwd=root/'source',stdout=stream,stderr=subprocess.STDOUT)
                sent=None
                while CHILD.poll() is None:
                    if (time.time()>=deadline or abort.exists() or STOP) and sent is None:
                        STOP=True
                        CHILD.send_signal(signal.SIGUSR1)
                        sent=time.monotonic()
                    if sent is not None and time.monotonic()-sent>570:
                        CHILD.kill()
                        CHILD.wait()
                        raise RuntimeError('child did not stop within the clean-pause margin')
                    time.sleep(1)
                code=CHILD.returncode
                CHILD=None
            if code not in (0,75):
                return code
            if code==75:
                if stage=='train':
                    branch=root/'models'/item['key']
                    pointer=read(branch/'LATEST.json')
                    pause=read(branch/f'PAUSE_{pointer["step"]:06d}_{start["job"]}.json')
                    if not pause['clean'] or pause['checkpoint']!=pointer:
                        raise ValueError('training pause not committed')
                publish(root/'resources'/f'{index:02d}_{slot}_PAUSE.json',
                    dict(item=item,clean=True,child_exit=75,job=start['job']))
                return 75
            if not item_done(root,stage,item):
                raise ValueError('worker exited successfully without completing its task')
    return 0


def cpu_action(root,stage):
    from workflows.sbi.e2e_vdm_context_control import freeze_models
    if stage=='freeze':
        freeze_models(root)
    elif stage=='sampler':
        from workflows.sbi.e2e_vdm_context_sample import finalize
        for arm in 'ABCD':
            for seed in (0,1):
                finalize(root,arm,seed,'refinement')
        from workflows.sbi.e2e_vdm_context_analysis import sampler_gate
        sampler_gate(root)
    elif stage=='report':
        from workflows.sbi.e2e_vdm_context_sample import finalize
        for arm in 'ABCD':
            for seed in (0,1):
                finalize(root,arm,seed,'all')
        from workflows.sbi.e2e_vdm_context_report import report
        report(root,workers=8)
    else:
        raise ValueError('unknown CPU stage')


def allocation(root,index):
    request=read(root/'resources'/f'{index:02d}_REQUEST.json')
    stage=request['stage']
    gates(root,stage)
    job=os.environ.get('SLURM_JOB_ID')
    if not job or not job.isdecimal():
        raise ValueError('allocation required; all compute uses srun below')
    publish(root/'resources'/f'{index:02d}_START.json',dict(job=job,gpus=request['gpus'],
        started_unix=time.time(),stage=stage,host=socket.gethostname()))
    processes=[]
    handles=[]
    count=request['gpus'] if stage in ('train','refinement','all') else 1
    for slot in range(count):
        cmd=['srun','--nodes=1','--ntasks=1','--cpus-per-task='+('32' if request['gpus'] else '64'),
             '--exclusive','--exact','--cpu-bind=cores','--chdir='+str(root/'source')]
        if request['gpus']:
            cmd+=['--gpus=1']
        cmd += [PYTHON,'-u','-m']
        if stage=='smoke':
            cmd+=['workflows.sbi.e2e_vdm_context_smoke','--root',str(root)]
        else:
            cmd+=['workflows.sbi.e2e_vdm_context_interactive','--root',str(root),'--mode',
                  'worker' if request['gpus'] else 'cpu','--index',str(index),'--slot',str(slot),'--stage',stage]
        handle=(root/'logs'/f'{index:02d}_{stage}_worker{slot}.log').open('x')
        handles.append(handle)
        processes.append(subprocess.Popen(cmd,env=environment(),stdout=handle,stderr=subprocess.STDOUT))
    abort=root/'resources'/f'{index:02d}_ABORT.json'
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None,0,75) for p in processes) and not abort.exists():
            publish(abort,dict(unexpected_worker_failure=True))
        time.sleep(1)
    for stream in handles:
        stream.close()
    codes=[p.returncode for p in processes]
    status=1 if any(c not in (0,75) for c in codes) else (75 if 75 in codes else 0)
    if status==0 and stage in ('train','refinement','all') and remaining(root,stage):
        status=75
    publish(root/'resources'/f'{index:02d}_END.json',dict(job=job,worker_codes=codes,status=status,
        ended_unix=time.time(),stage=stage,manifest_sha256=digest(root/'MANIFEST.json')))
    return status


def allocation_command(root,index,gpus,minutes):
    return ['salloc','--nodes=1','--ntasks='+str(gpus or 1),
        '--cpus-per-task='+('32' if gpus else '64'),'--constraint='+('gpu&hbm80g' if gpus else 'cpu'),
        '--qos='+('shared_interactive' if gpus in (1,2) else 'interactive'),
        '--account='+('desi_g' if gpus else 'desi'),'--licenses=scratch','--immediate=600',
        f'--time={minutes//60:02d}:{minutes%60:02d}:00','--job-name=vdm-context-'+str(index)]+(
        ['--gpus='+str(gpus)] if gpus else [])+[PYTHON,'-u','-m','workflows.sbi.e2e_vdm_context_interactive',
        '--mode','allocation','--root',str(root),'--index',str(index)]


def request(root,stage):
    m=gates(root,stage)
    folder=root/'resources'
    requests=sorted(folder.glob('*_REQUEST.json'))
    if stage in GPU_STAGES and sum(read(p)['gpus']>0 for p in requests)>=8:
        raise RuntimeError('approved eight GPU requests exhausted')
    usage=accounting(root)
    left=len(remaining(root,stage)) if stage in ('train','refinement','all') else 1
    gpus,minutes=request_size(stage,left,usage,m['spec']['budget'],m['deadline_epoch'],time.time())
    listing=subprocess.check_output(['squeue','--me','-h','-o','%i|%q|%b'],text=True)
    if sum('interactive' in row for row in listing.splitlines())>=2:
        raise RuntimeError('two interactive allocations already active; no new request')
    if gpus and any('gpu' in row.split('|')[-1].lower() for row in listing.splitlines()):
        raise RuntimeError('an existing GPU allocation must be reviewed/reused before requesting another')
    index=len(requests)
    cmd=allocation_command(root,index,gpus,minutes)
    publish(folder/f'{index:02d}_REQUEST.json',dict(stage=stage,gpus=gpus,minutes=minutes,
        command=cmd,prior_accounting=usage,deadline_epoch=m['deadline_epoch'],
        manifest_sha256=digest(root/'MANIFEST.json'),controller_sha256=digest(__file__)))
    with (root/'logs'/f'resource_{index:02d}.log').open('x') as log:
        code=subprocess.run(cmd,cwd=root/'source',env=environment(),stdout=log,stderr=subprocess.STDOUT).returncode
    publish(folder/f'{index:02d}_RETURN.json',dict(exit_code=code,epoch=time.time()))
    end=folder/f'{index:02d}_END.json'
    if code not in (0,75) or not end.exists() or read(end)['status']!=code:
        raise RuntimeError('unexpected allocation/scientific failure; no automatic retry')
    # Slurm accounting can lag release briefly; never infer a retry from timeout.
    for attempt in range(6):
        try:
            usage=accounting(root)
            break
        except RuntimeError:
            if attempt==5:
                raise
            time.sleep(10)
    publish(folder/f'{index:02d}_ACCOUNTING.json',usage)
    return code


def controller(root,smoke_only=False):
    manifest(root)
    folder=root/'resources'
    folder.mkdir(exist_ok=True)
    marker=folder/('SMOKE_CONTROLLER.json' if smoke_only else 'MATRIX_CONTROLLER.json')
    publish(marker,dict(pid=os.getpid(),host=socket.gethostname(),epoch=time.time(),
        manifest_sha256=digest(root/'MANIFEST.json'),controller_sha256=digest(__file__)))
    stages=('smoke',) if smoke_only else STAGES
    for stage in stages:
        if stage=='smoke' and (root/'SMOKE.json').exists():
            gates(root,'train')
            continue
        if stage in ('train','refinement','all'):
            while remaining(root,stage):
                request(root,stage)
        else:
            request(root,stage)
        print('STAGE_COMPLETE',stage,flush=True)
    if not smoke_only:
        from workflows.sbi.e2e_vdm_context_queue import disk_bytes
        report=root/'analysis/RESULTS.json'
        if not report.is_file() or not (root/'analysis/REPORT.md').is_file():
            raise ValueError('final scientific report missing')
        publish(root/'EXPERIMENT_COMPLETE.json',dict(complete=True,manifest_sha256=digest(root/'MANIFEST.json'),
            results_sha256=digest(report),resource_accounting=accounting(root),scratch_bytes=disk_bytes(root),
            production_ready=False,automatic_architecture_change=False))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--mode',required=True,choices=['smoke-controller','controller','allocation','worker','cpu'])
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--slot',type=int,default=0)
    p.add_argument('--stage',choices=STAGES)
    a=p.parse_args()
    root=check_root(a.root)
    if a.mode in ('smoke-controller','controller'):
        return controller(root,a.mode=='smoke-controller')
    if a.mode=='allocation':
        return allocation(root,a.index)
    if a.mode=='worker':
        return worker(root,a.index,a.slot)
    cpu_action(root,a.stage)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
