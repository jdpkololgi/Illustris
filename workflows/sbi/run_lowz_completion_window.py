"""One explicitly approved CPU preparation window, with no failed-stage retry.

The user authorizes at most two sequential four-hour allocations in addition to
60481835. This entrypoint never submits jobs. It commits each stage separately,
stops before an estimated stage would exceed its budget, and refuses any failed
stage marker on a later invocation. Budget pauses are not execution failures.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import prepare_cfm_lowz_campaign as campaign


class Window:
    def __init__(self,seconds,replacement_job=None,continuation_20261030=False,continuation_20261102=False):
        c.require_compute();campaign.authority()
        if not 300<=seconds<=14100:raise ValueError('Must fit approved four-hour CPU window')
        self.started=time.monotonic();self.deadline=self.started+seconds
        windows=c.ROOT/'completion_windows'
        windows.mkdir(parents=True,exist_ok=True)
        job_directory=windows/os.environ['SLURM_JOB_ID']
        limit=2
        self.replacement_job=replacement_job
        if replacement_job is not None:
            approval=json.loads((c.REPO/'configs/lowz_replacement_60522523.json').read_text())
            if (replacement_job!=approval['interrupted_job'] or seconds>approval['maximum_runner_seconds']
                    or not (windows/replacement_job).is_dir()
                    or not approval['one_controlled_resume_approved']):
                raise PermissionError('Replacement does not match explicit approval')
            limit=approval['maximum_total_completion_allocations']
        if continuation_20261030:
            if replacement_job is not None:
                raise PermissionError('Continuation and replacement approvals are distinct')
            approval=json.loads((c.REPO/'configs/lowz_continuation_20261030.json').read_text())
            if seconds>approval['maximum_runner_seconds']:
                raise PermissionError('Continuation exceeds approved budget')
            limit=approval['maximum_total_completion_allocations']
        if continuation_20261102:
            if replacement_job is not None or continuation_20261030:
                raise PermissionError('Use exactly one continuation approval')
            if seconds>13800:
                raise PermissionError('Continuation exceeds approved budget')
            limit=5  # One additional four-hour CPU window explicitly approved 2026-11-02.
        self.allocation_limit=limit
        if not job_directory.exists() and len(list(windows.iterdir()))>=limit:
            raise PermissionError('Two approved additional allocations already consumed')
        job_directory.mkdir(exist_ok=True)
        self.directory=job_directory/('execution_'+str(time.time_ns()))
        self.directory.mkdir(exist_ok=False)
        self.stages=[];self.stop=None
        self.registry=c.ROOT/'completion_stages';self.registry.mkdir(exist_ok=True)

    def stage(self,key,module,arguments=(),estimate=600):
        marker=self.registry/(key+'_COMPLETE.json')
        failure=self.registry/(key+'_FAILED.json')
        if failure.exists():
            self.stop=dict(reason='existing failed stage; no automatic retry',stage=key,record=str(failure))
            return False
        command=[sys.executable,'-u','-m',module,*map(str,arguments)]
        source=c.REPO/(module.replace('.','/')+'.py')
        binding=dict(command=command,source_sha256=c.sha256(source),config_sha256=c.sha256(c.CONFIG))
        if marker.exists():
            prior=json.loads(marker.read_text())
            if prior['binding']!=binding:raise ValueError('Stage implementation changed: '+key)
            return True
        remaining=self.deadline-time.monotonic()
        if remaining<estimate+180:
            self.stop=dict(reason='budget pause before new stage',next_stage=key,
                           remaining_seconds=remaining,estimated_seconds=estimate)
            return False
        log=self.directory/(key+'.log');started=time.monotonic()
        print('STAGE_START',key,flush=True)
        with log.open('x') as stream:
            try:
                result=subprocess.run(command,cwd=c.REPO,stdout=stream,stderr=subprocess.STDOUT,
                                      timeout=remaining-90,check=False)
                exitcode=result.returncode;error=None
            except subprocess.TimeoutExpired:
                exitcode=124;error='stage exceeded remaining approved window'
        record=dict(key=key,binding=binding,exitcode=exitcode,error=error,log=str(log),
                    elapsed_seconds=time.monotonic()-started,job=os.environ['SLURM_JOB_ID'])
        self.stages.append(record)
        c.atomic_json(marker if exitcode==0 else self.registry/(key+'_FAILED.json'),record)
        print('STAGE_END',key,exitcode,round(record['elapsed_seconds'],1),flush=True)
        if exitcode:
            self.stop=dict(reason='stage failed; no automatic retry',stage=key,log=str(log),exitcode=exitcode)
            return False
        return True

    def close(self):
        c.atomic_json(self.directory/'WINDOW_COMPLETE.json',dict(job=os.environ['SLURM_JOB_ID'],
            stages=self.stages,stop=self.stop,elapsed_seconds=time.monotonic()-self.started,
            maximum_new_allocations=self.allocation_limit,maximum_additional_node_hours=8,
            replacement_for_interrupted_job=self.replacement_job,
            training_started=False,model_inputs_ready=False))
        print('LOWZ_COMPLETION_WINDOW_FINISHED',self.directory,flush=True)


def run(seconds,mode='canary',cached_geometry=False,replacement_job=None,continuation_20261030=False,continuation_20261102=False):
    window=Window(seconds,replacement_job,continuation_20261030,continuation_20261102)
    try:
        if not window.stage('cfm_coordinate_checks','workflows.sbi.prepare_cfm_lowz_campaign',['checks'],180):return
        # Diagnose the uncertain P12 identity linkage before a 23-phase rollout.
        p12_hosts=window.stage('p12_host_canaries','workflows.sbi.audit_p12a_lowz_native_hosts',
                            ['--phases','ph001','ph006'],1800)
        if not p12_hosts:
            if window.stop['reason']=='budget pause before new stage':return
            # A terminal P12 diagnostic failure is not a reason to discard the
            # independent CFM work. The persistent failure marker bars retries.
            print('P12_BLOCKED',window.stop,flush=True)
            window.stop=None
        elif mode=='canary':
            if not window.stage('p12_response_canary','workflows.sbi.build_p12a_matched_response',
                                ['--phase','ph001','--caps','0','--workers','16'],900):
                if window.stop['reason']=='budget pause before new stage':return
                print('P12_RESPONSE_BLOCKED',window.stop,flush=True)
                window.stop=None
        phases=['ph001','ph006']+[p for p in (*c.TRAIN,*c.DEVELOPMENT) if p not in ('ph001','ph006')]
        if mode=='canary':phases=['ph001']
        elif not (window.registry/'ph001_whole_targets_COMPLETE.json').exists():
            raise PermissionError('Complete the ph001 canary before broad rollout')
        for phase in phases:
            steps=[
                ('matched_response','workflows.sbi.build_cfm_matched_response',['--phase',phase,'--workers','16'],900),
                ('base','workflows.sbi.prepare_cfm_lowz_campaign',['base','--phases',phase],300),
                ('response_adoption','workflows.sbi.prepare_cfm_lowz_campaign',['responses','--phases',phase],300),
                ('geometry','workflows.sbi.prepare_cfm_lowz_geometry_cached' if cached_geometry else 'workflows.sbi.prepare_cfm_lowz_campaign',
                 ['--phase',phase] if cached_geometry else ['geometry','--phases',phase],600),
                ('targets','workflows.sbi.e2e_coupled_target_products',['--phase',phase,'--workers','32','--stop-after-seconds','12000'],1800),
                ('conditions','workflows.sbi.e2e_coupled_condition_products',['--phase',phase],900),
                ('audit','workflows.sbi.qualify_cfm_lowz_phase',['--phase',phase],900),
                ('whole_observations','workflows.sbi.prepare_shared_wholefield',['observations','--phases',phase],900),
                ('whole_targets','workflows.sbi.prepare_shared_wholefield',['targets','--phases',phase,'--workers','32'],1200),
            ]
            for name,module,arguments,estimate in steps:
                if not window.stage(phase+'_'+name,module,arguments,estimate):return
        if mode=='canary':
            window.stop=dict(reason='canary complete; inspect scientific receipts before campaign continuation')
            return
        if not window.stage('cfm_normalization','workflows.sbi.e2e_coupled_normalization',[],1800):return
        for phase in phases:
            if not window.stage(phase+'_whole_finalize','workflows.sbi.prepare_shared_wholefield',
                                ['finalize','--phases',phase],300):return
        if not window.stage('cfm_loader_checks','workflows.sbi.check_shared_phase_loaders',['--seconds','600'],900):return
    finally:
        window.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seconds',type=int,default=14100)
    parser.add_argument('--mode',choices=('canary','campaign'),default='canary')
    parser.add_argument('--cached-geometry',action='store_true')
    parser.add_argument('--replacement-job')
    parser.add_argument('--continuation-20261030',action='store_true')
    parser.add_argument('--continuation-20261102',action='store_true')
    args=parser.parse_args();run(args.seconds,args.mode,args.cached_geometry,args.replacement_job,args.continuation_20261030,args.continuation_20261102)
