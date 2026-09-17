"""Explicitly authorized, bounded Slurm handoff after GPU/resume gates.

Standalone launcher binds its own source hash; does not alter the frozen runner.
No automatic retries, requeues, scheduler mutations or training extensions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def commands(root,dependency=None):
    common=['sbatch','--parsable','--nodes=1','--ntasks=1','--licenses=scratch','--no-requeue',
        '--chdir='+str(root/'source')]
    script=str(root/'source/workflows/sbi/submit_e2e_vdm_assessment.slurm')
    if dependency is None:
        return common+['--account=desi_g','--constraint=gpu','--gpus=1','--cpus-per-task=32','--qos=shared',
            '--array=0-3%2','--time=02:30:00','--signal=USR1@180','--job-name=vdm-assessment',
            '--output='+str(root/'logs/eval_%A_%a.out'),'--error='+str(root/'logs/eval_%A_%a.err'),script,str(root),'run']
    if not str(dependency).isdigit():raise ValueError('numeric dependency required')
    return common+['--account=desi','--constraint=cpu','--cpus-per-task=8','--qos=debug','--time=00:10:00',
        '--job-name=vdm-assessment-report','--dependency=afterok:'+str(dependency),
        '--output='+str(root/'logs/report_%j.out'),'--error='+str(root/'logs/report_%j.err'),script,str(root),'report']


def publish(path,obj):
    with path.open('x') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def submit(root):
    root=root.resolve()
    if root.parent!=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1') or not root.name.startswith('vdm_assessment_'):
        raise ValueError('registered assessment root required')
    manifest=json.loads((root/'MANIFEST.json').read_text());digest=sha(root/'MANIFEST.json')
    for name,h in manifest['source_sha256'].items():
        if sha(root/'source'/name)!=h:raise ValueError('frozen source drift')
    for file in ['SMOKE.json','RESTART_TEST.json']:
        proof=json.loads((root/file).read_text())
        if not proof['passed'] or proof['manifest_sha256']!=digest:raise ValueError('missing matching technical gate')
    cmd=commands(root)
    publish(root/'SUBMISSION_INTENT.json',dict(command=cmd,manifest_sha256=digest,launcher_sha256=sha(__file__),
        maximum_concurrent_gpus=2,maximum_gpu_hours=10,no_training=True))
    result=subprocess.run(cmd,capture_output=True,text=True)
    publish(root/'EVALUATION_SUBMISSION.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode:raise RuntimeError('evaluation submit failed; no retry')
    job=result.stdout.strip().split(';')[0]
    report_cmd=commands(root,job)
    publish(root/'REPORT_SUBMISSION_INTENT.json',dict(command=report_cmd,evaluation_job=job))
    result=subprocess.run(report_cmd,capture_output=True,text=True)
    publish(root/'REPORT_SUBMISSION.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode:raise RuntimeError('report submit failed; evaluation remains submitted; no retry')
    print('SUBMITTED evaluation',job,'report',result.stdout.strip(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--authorized',action='store_true',help='invoke only after explicit user batch/resource approval')
    args=parser.parse_args()
    if not args.authorized:parser.error('explicit authorization required')
    submit(args.root)
