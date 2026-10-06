"""Source-bound, fail-closed task orchestration for the authorized Tier 2 sweep."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
RUN=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3')
EVAL=RUN/'evaluation_v1'
CONFIG='configs/e2e_cfm_conditioning_evaluation_v1.json'
FILES=[CONFIG]+['workflows/sbi/'+name for name in (
    'e2e_cfm_conditioning_oracle.py','e2e_cfm_conditioning_extract.py',
    'e2e_cfm_conditioning_metrics.py','e2e_cfm_conditioning_score.py',
    'e2e_cfm_conditioning_evaluation.py','run_e2e_cfm_conditioning_extension.sh',
    'submit_e2e_cfm_conditioning_gpu.slurm','submit_e2e_cfm_conditioning_cpu.slurm')]
SETTINGS=[('parent',0.),('control',0.)]+[('augmented',s) for s in (0.,.025,.05,.1,.2)]


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(16<<20),b''):value.update(part)
    return value.hexdigest()


def record(path):return dict(path=str(path),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def verify(rec):
    if sha(rec['path'])!=rec['sha256']:raise ValueError('Changed binding: '+rec['path'])


def fullcap_task(index):
    if not 0<=index<224:raise ValueError('Full-cap task index outside frozen grid')
    setting,offset=divmod(index,32);panel,chunk=divmod(offset,8)
    return ('ph012','ph013')[panel//2],('NGC','SGC')[panel%2],chunk*4,*SETTINGS[setting]


def freeze():
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    for name in FILES:
        expected=subprocess.check_output(['git','show',revision+':'+name],cwd=REPO)
        if (REPO/name).read_bytes()!=expected:raise ValueError('Uncommitted evaluation source: '+name)
    smoke_paths=[RUN/'oracle_smoke/ph012_NGC_s0_interior_00/COMPLETE.json',
        RUN/'extract_recomplete_smoke/ph012/NGC/parent/s0/d000/COMPLETE.json',
        RUN/'fields/ph012/SGC/control/s0/d000/COMPLETE.json']
    for path in smoke_paths:
        if not read(path)['passed']:raise ValueError('Required representative smoke has not passed')
    oracle=read(smoke_paths[0]);extraction=read(smoke_paths[1]);fine=read(smoke_paths[2])
    if oracle['source']['sha256']!=sha(REPO/'workflows/sbi/e2e_cfm_conditioning_oracle.py'):
        raise ValueError('Oracle changed after smoke')
    verify(extraction['binding'])
    bound=read(extraction['binding']['path'])
    if bound['source']['sha256']!=sha(REPO/'workflows/sbi/e2e_cfm_conditioning_extract.py'):
        raise ValueError('Extractor changed after smoke')
    if not extraction['exact_exterior_recompletion'] or not extraction['stage_b_labels_exact'] or extraction['stage_ab_parity_maxabs']!=0:
        raise ValueError('Exact exterior and Stage A/B parity required')
    verify(fine['field'])
    if fine['arm']!='control' or fine['strength']!=0 or fine['nfe']!=128 or fine['draws']!=4:
        raise ValueError('Wrong fine continuation benchmark')
    EVAL.mkdir(exist_ok=False);snapshot=EVAL/'source';sources={}
    for name in FILES:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/name,target);sources[name]=sha(target);target.chmod(0o444)
    (EVAL/'logs').mkdir();(EVAL/'tasks').mkdir()
    prior_source=RUN/'source'
    frozen=read(RUN/'SMOKE.json')
    dependencies={name:sha(prior_source/'workflows/sbi'/name) for name in frozen['sources']}
    if dependencies!=frozen['sources']:raise ValueError('Frozen training sources changed')
    checkpoints={arm:record(RUN/arm/'CHECKPOINT_003328.pt') for arm in ('control','augmented')}
    value=dict(schema='cfm-conditioning-evaluation-launch-v1',revision=revision,sources=sources,
        profile=record(prior_source/'configs/e2e_cfm_conditioning_augmentation_v1.json'),
        configuration=read(REPO/CONFIG),frozen_training_sources=dependencies,checkpoints=checkpoints,
        smokes=[record(p) for p in smoke_paths],authorized='User 2026-10-06: full frozen sweep; all necessary compute',
        task_grid=[fullcap_task(i) for i in range(224)],automatic_retry=False,posterior_qualified=False)
    write(EVAL/'MANIFEST.json',value)
    print(json.dumps(record(EVAL/'MANIFEST.json')),flush=True)


def launch_check():
    manifest=read(EVAL/'MANIFEST.json')
    if REPO!=EVAL/'source':raise PermissionError('Execute the committed evaluation snapshot')
    if manifest['configuration']!=read(REPO/CONFIG):raise ValueError('Evaluation config drift')
    for name,digest in manifest['sources'].items():
        if sha(REPO/name)!=digest:raise ValueError('Evaluation source drift: '+name)
    verify(manifest['profile'])
    for rec in manifest['checkpoints'].values():verify(rec)
    for rec in manifest['smokes']:verify(rec)
    for name,digest in manifest['frozen_training_sources'].items():
        if sha(RUN/'source/workflows/sbi'/name)!=digest:raise ValueError('Training dependency drift')
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Compute allocation required')
    return manifest


def call(script,args,cwd=None):
    cmd=[sys.executable,str(REPO/'workflows/sbi'/script),*map(str,args)]
    subprocess.run(cmd,cwd=cwd or REPO,check=True)


def complete_or_absent(directory,kind):
    path=directory/'COMPLETE.json'
    if not directory.exists():return False
    if not path.exists():raise RuntimeError('Partial '+kind+' preserved; no automatic retry: '+str(directory))
    q=read(path)
    if not q['passed']:raise ValueError('Failed '+kind+' cannot be reused')
    for key in ('field','output'):
        if key in q:verify(q[key])
    return True


def worker(kind,index):
    launch_check();started=time.monotonic()
    receipt=EVAL/'tasks'/f'{kind}_{index:03d}.json'
    if receipt.exists():raise RuntimeError('Task receipt exists; no automatic task replay')
    if kind=='fullcap':
        phase,cap,first,arm,strength=fullcap_task(index)
        args=['--phase',phase,'--cap',cap,'--first',first,'--arm',arm,'--strength',strength]
        if arm!='parent':
            directory=RUN/'fields'/phase/cap/arm/f's{strength:g}'/f'd{first:03d}'
            if not complete_or_absent(directory,'fine generation'):
                subprocess.run([sys.executable,'-m','workflows.sbi.e2e_cfm_conditioning_sample',*map(str,args)],
                    cwd=RUN/'source',check=True)
        out=RUN/'cores'/phase/cap/arm/f's{strength:g}'/f'd{first:03d}'
        if not complete_or_absent(out,'core extraction'):call('e2e_cfm_conditioning_extract.py',args)
        output=record(out/'COMPLETE.json')
    elif kind=='oracle':
        if index!=0:raise ValueError('Single frozen oracle-panel task')
        for case in range(32):call('e2e_cfm_conditioning_oracle.py',['--case',case])
        output=dict(panel='32 case receipts under '+str(RUN/'oracle'))
    elif kind=='score':
        if not 0<=index<7:raise ValueError('Unregistered score setting')
        arm,strength=SETTINGS[index]
        call('e2e_cfm_conditioning_score.py',['arm','--arm',arm,'--strength',strength])
        output=record(RUN/'scores'/f'{arm}_s{strength:g}'/'COMPLETE.json')
    elif kind=='summary':
        if index!=0:raise ValueError('Single summary task')
        call('e2e_cfm_conditioning_score.py',['summarize'])
        output=record(RUN/'decision/COMPLETE.json')
    else:raise ValueError('Unregistered task kind')
    write(receipt,dict(passed=True,kind=kind,index=index,output=output,seconds=time.monotonic()-started,
        manifest=record(EVAL/'MANIFEST.json'),job=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','worker'])
    ap.add_argument('--kind',choices=['fullcap','oracle','score','summary']);ap.add_argument('--index',type=int,default=0)
    args=ap.parse_args()
    if args.action=='freeze':freeze()
    else:worker(args.kind,args.index)
