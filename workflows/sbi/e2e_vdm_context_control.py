"""Immutable source/data registration and exact model-matrix freeze receipts."""
import argparse
from pathlib import Path
import subprocess

import torch

from workflows.sbi import e2e_durable as durable,e2e_wide_pipeline as existing
from workflows.sbi.e2e_field_build_products import require_compute
from workflows.sbi.e2e_clean_limit_launch import snapshot_paths
from workflows.sbi.e2e_vdm_context_data import CONFIG,REPO,ROLES,read_json,spec,output_root
from workflows.sbi.e2e_vdm_context_tasks import draw_tasks
from workflows.sbi.e2e_vdm_context_train import verify_launch


def factors():
    return [(arm,seed,'fine') for arm in 'ABCD' for seed in (0,1)]+[('D',seed,'coarse') for seed in (0,1)]


def snapshot(root,label):
    if label not in ('physics','run'):
        raise ValueError('unregistered source snapshot')
    root=output_root(root)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('commit tested source before staging')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    names=snapshot_paths(subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0'))
    names+=['docs/e2e_vdm_context_diversity_v1.md','docs/e2e_vdm_context_audit_proposal_20260917.md']
    names=sorted(set(names))
    source=root/('source_physics' if label=='physics' else 'source')
    source.mkdir(exist_ok=False)
    archive=subprocess.Popen(['git','archive',revision,'--',*names],cwd=REPO,stdout=subprocess.PIPE)
    result=subprocess.run(['tar','-xf','-','-C',str(source)],stdin=archive.stdout)
    archive.stdout.close()
    if archive.wait() or result.returncode:
        raise RuntimeError('partial source archive retained')
    return dict(source=str(source),source_sha256={n:existing.sha256(source/n) for n in names if (source/n).is_file()},
                git_revision=revision,config_sha256=existing.sha256(CONFIG))


def stage_physics(root):
    root=output_root(root)
    if (root/'PHYSICS_SOURCE.json').exists():
        raise FileExistsError('physical evaluator already frozen')
    result=snapshot(root,'physics')
    durable.publish_json(root/'PHYSICS_SOURCE.json',result)
    print('PHYSICS_SOURCE',result['source'],flush=True)


def stage_run(root):
    root=output_root(root)
    if (root/'MANIFEST.json').exists():
        raise FileExistsError('full run already registered')
    paths=['data/GEOMETRY.json','data/REPRESENTATION_GATE.json','data/NORMALIZATION.json',
           'PHYSICS_SOURCE.json','BUILD_SOURCE.json']+[f'data/{phase}/COMPLETE.json' for phase in ROLES]
    hashes={p:existing.sha256(root/p) for p in paths}
    gate=read_json(root/'data/REPRESENTATION_GATE.json')
    norm=read_json(root/'data/NORMALIZATION.json')
    if not gate['training_launch_allowed'] or len(norm['fit_ids'])!=32 or norm['fit_phases']!=['ph000','ph002']:
        raise PermissionError('physical/normalization gate not passed')
    builders=set()
    for phase in ROLES:
        receipt=read_json(root/f'data/{phase}/COMPLETE.json')
        if (receipt['phase']!=phase or receipt['role']!=ROLES[phase]
                or receipt['binding']['geometry_sha256']!=hashes['data/GEOMETRY.json']
                or receipt['binding']['config_sha256']!=existing.sha256(CONFIG)):
            raise ValueError('data role/source mismatch')
        builders.add(receipt['binding']['builder_sha256'])
    if len(builders)!=1:
        raise ValueError('different physical builders across phases')
    ledger=draw_tasks(read_json(root/'data/GEOMETRY.json')['rows'])
    durable.publish_json(root/'DRAW_LEDGER.json',ledger)
    hashes['DRAW_LEDGER.json']=existing.sha256(root/'DRAW_LEDGER.json')
    result=dict(snapshot(root,'run'),schema='e2e-vdm-context-run-v1',data_receipts=hashes,
        physical_builder_sha256=builders.pop(),spec=spec(),phase_roles=ROLES,
        deadline_epoch=read_json(root/'SCREEN_REQUEST.json')['deadline_epoch'],
        production_ready=False,all_models_frozen=False)
    durable.publish_json(root/'MANIFEST.json',result)
    print('RUN_SOURCE',result['source'],flush=True)


def freeze_models(root):
    require_compute()
    root=output_root(root)
    verify_launch(root)
    if (root/'MODELS_FROZEN.json').exists():
        verify_models_frozen(root)
        return
    checkpoints,branches={},{}
    for arm,seed,factor in factors():
        branch=root/'models'/f'{arm}_{factor}_seed{seed}'
        complete=read_json(branch/'COMPLETE.json')
        if (complete['updates']!=20480 or complete['examples_seen']!=40960
                or complete['distinct_patches']!=(32 if arm=='A' else 384)
                or complete['independent_training_phases']!=(2 if arm=='A' else 3)
                or complete['binding']['manifest_sha256']!=existing.sha256(root/'MANIFEST.json')):
            raise ValueError('training exposure/matrix incomplete')
        branches[str(branch.relative_to(root))]=existing.sha256(branch/'COMPLETE.json')
        for update in spec()['checkpoint_updates']:
            pointer=read_json(branch/f'CHECKPOINT_{update:06d}.json')
            path=(branch/pointer['path']).resolve()
            if branch.resolve() not in path.parents or read_json(path.parent/'COMMITTED.json')!=pointer:
                raise ValueError('checkpoint pointer not committed')
            state=existing.load_checkpoint(path,complete['binding'],stage=factor,method='vdm')
            if state['step']!=update or len(state['history'])!=update:
                raise ValueError('checkpoint update/history mismatch')
            del state
            checkpoints[str(path.relative_to(root))]=pointer['sha256']
    durable.publish_json(root/'MODELS_FROZEN.json',dict(all_models_frozen=True,checkpoints=checkpoints,
        branch_receipts=branches,geometry_sha256=existing.sha256(root/'data/GEOMETRY.json'),
        manifest_sha256=existing.sha256(root/'MANIFEST.json'),choice_policy='all20480 final, no validation-selected checkpoint'))
    print('MODELS_FROZEN',len(branches),len(checkpoints),flush=True)


def verify_models_frozen(root):
    receipt=read_json(root/'MODELS_FROZEN.json')
    if (not receipt['all_models_frozen'] or len(receipt['checkpoints'])!=30 or len(receipt['branch_receipts'])!=10
            or receipt['manifest_sha256']!=existing.sha256(root/'MANIFEST.json')
            or receipt['geometry_sha256']!=existing.sha256(root/'data/GEOMETRY.json')):
        raise ValueError('invalid full matrix freeze')
    expected={f'models/{arm}_{factor}_seed{seed}' for arm,seed,factor in factors()}
    if set(receipt['branch_receipts'])!=expected:
        raise ValueError('frozen matrix branches differ from registration')
    for branch,digest in receipt['branch_receipts'].items():
        if existing.sha256(root/branch/'COMPLETE.json')!=digest:
            raise ValueError('completed model changed after freeze')
        for update in spec()['checkpoint_updates']:
            pointer=read_json(root/branch/f'CHECKPOINT_{update:06d}.json')
            path=str(Path(branch)/pointer['path'])
            if receipt['checkpoints'].get(path)!=pointer['sha256']:
                raise ValueError('selected model pointer changed after freeze')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['stage-physics','stage-run','freeze-models'])
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    {'stage-physics':stage_physics,'stage-run':stage_run,'freeze-models':freeze_models}[a.mode](a.root)


if __name__=='__main__':
    main()
