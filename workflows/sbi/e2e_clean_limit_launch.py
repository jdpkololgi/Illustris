"""Stage committed source and submit a bounded, disconnect-independent chain.

Submission requires explicit user authorization and SMOKE.json from the GPU
checkpoint/resume test. No automatic retries; interrupted submissions fail closed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi import e2e_durable as durable
from workflows.sbi.e2e_clean_limit import read_spec

SCRATCH = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1')


def stage(root):
    root = root.resolve()
    if root.parent != SCRATCH or not root.name.startswith('clean_limit_'):
        raise ValueError('use a new clean_limit_ run immediately under the registered Scratch root')
    if subprocess.check_output(['git','status','--porcelain'],cwd=p.REPO,text=True).strip():
        raise ValueError('source must be committed and clean before snapshot')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.REPO,text=True).strip()
    spec = read_spec()
    old=Path(spec['parent_root'])
    if p.sha256(old/'PREPARED.json') != spec['prepared_sha256']:
        raise ValueError('prepared receipt drift')
    root.mkdir(exist_ok=False)
    source=root/'source'; source.mkdir(); (root/'logs').mkdir()
    archive=subprocess.Popen(['git','archive',revision],cwd=p.REPO,stdout=subprocess.PIPE)
    unpack=subprocess.run(['tar','-xf','-','-C',str(source)],stdin=archive.stdout)
    archive.stdout.close()
    if archive.wait() or unpack.returncode:
        raise RuntimeError('source snapshot failed; preserve partial root')
    # Bind every tracked source/config/doc byte, not just the top-level runner.
    names=subprocess.check_output(['git','ls-files','-z'],cwd=p.REPO).decode().split('\0')
    hashes={name:p.sha256(source/name) for name in names if name and (source/name).is_file()}
    parents={str(r):p.sha256(old/f'replica_{r}/n15_current/update_003072.pt') for r in spec['replicas']}
    durable.publish_json(root/'MANIFEST.json',dict(schema='e2e-clean-limit-source-v1',
        source=str(source),git_revision=revision,source_sha256=hashes,spec=spec,parent_sha256=parents,
        frozen_sha256=p.sha256(old/'FROZEN.json'),heldout_payloads_read=False,training_ready=False))
    print('STAGED',root,revision,flush=True)


def command(root, stage_name, dependency=None):
    gpu=stage_name!='analysis'
    args=['sbatch','--parsable','--nodes=1','--ntasks=1','--qos=shared','--licenses=scratch',
          '--no-requeue','--signal=USR1@180','--open-mode=append',
          '--job-name=e2e-limit-'+stage_name,'--chdir='+str(root/'source'),
          '--output='+str(root/f'logs/{stage_name}_%A_%a.out'),
          '--error='+str(root/f'logs/{stage_name}_%A_%a.err')]
    if gpu:
        args+=['--account=desi_g','--constraint=gpu','--gpus=1','--cpus-per-task=32']
        args+=['--array=0-1%2','--time=03:00:00'] if stage_name=='optimization' else ['--array=0-3%2','--time=02:00:00']
    else:
        args+=['--account=desi','--constraint=cpu','--cpus-per-task=8','--mem=16G','--time=00:30:00']
    if dependency:
        args+=['--dependency=afterok:'+str(dependency)]
    args += [str(root/'source/workflows/sbi/submit_e2e_clean_limit.slurm'),str(root),stage_name]
    return args


def submit(root):
    from workflows.sbi.e2e_clean_limit import verify_snapshot
    manifest=verify_snapshot(root)
    smoke=json.loads((root/'SMOKE.json').read_text())
    if not smoke['passed'] or smoke['manifest_sha256']!=p.sha256(root/'MANIFEST.json'):
        raise ValueError('representative smoke gate missing or stale')
    restart=json.loads((root/'RESTART_TEST.json').read_text())
    if not restart['passed'] or not restart['exact'] or restart['manifest_sha256']!=p.sha256(root/'MANIFEST.json'):
        raise ValueError('actual runner signal/restart gate missing or stale')
    # Exclusive intent creation occurs BEFORE contacting Slurm. If submission or
    # this process is interrupted, inspect Slurm and recover manually, never retry
    # automatically and risk duplicate jobs whose IDs were not recorded.
    intent=root/'SUBMISSION_INTENT.json'
    durable.publish_json(intent,dict(manifest_sha256=p.sha256(root/'MANIFEST.json'),
                                    stages=['optimization','coverage','analysis']))
    dependency=None; jobs=[]
    for stage_name in ('optimization','coverage','analysis'):
        args=command(root,stage_name,dependency)
        result=subprocess.run(args,text=True,capture_output=True)
        if result.returncode:
            durable.publish_json(root/'SUBMISSION_FAILED.json',dict(stage=stage_name,command=args,
                stdout=result.stdout,stderr=result.stderr,jobs=jobs))
            raise RuntimeError('submission stopped; inspect partial chain, no automatic retry')
        jobid=result.stdout.strip().split(';')[0]
        if not jobid.isdecimal():
            raise RuntimeError('ambiguous submission response; inspect Slurm before retry')
        receipt=dict(stage=stage_name,job_id=jobid,dependency=dependency,command=args,
                     source_revision=manifest['git_revision'])
        durable.publish_json(root/f'SUBMITTED_{stage_name}.json',receipt)
        jobs.append(receipt);dependency=jobid
        print('SUBMITTED',stage_name,jobid,flush=True)
    durable.publish_json(root/'SUBMITTED.json',dict(jobs=jobs,automatic_retry=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['stage','submit','print-commands'])
    ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='stage':stage(args.root)
    elif args.command=='submit':submit(args.root)
    else:
        for name in ('optimization','coverage','analysis'):
            print(json.dumps(command(args.root,name)))
