"""Explicit 192->384 continuation; preserve the original canary contract."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch

from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION

PARENT = p.SCRATCH_ROOT/'wide_pipeline_v1/research_20260911_58196924'
CONTRACT_FILE = p.REPO/'configs/e2e_wide_continuation_20260914.json'


def contract(base):
    c = json.loads(CONTRACT_FILE.read_text())
    if (c['start_update'], c['stop_update'], c['loss_checkpoints']) != (192, 384, [288, 336, 384]):
        raise ValueError('only the separately authorized 384-update continuation is supported')
    if c['heldout_access_authorized'] or c['automatic_extension']:
        raise ValueError('no heldout opening or automatic extension')
    receipt = PARENT/'RESEARCH_CANARY_COMPLETE.json'
    old = json.loads(receipt.read_text())
    if old['training_complete'] is not True or len(old['stages']) != 4:
        raise ValueError('parent training incomplete')
    parents = {}
    for row in old['stages']:
        path = PARENT/f'{row["method"]}_{row["stage"]}/step_000192.pt'
        if row['updates'] != 192 or p.sha256(path) != row['checkpoint_sha256']:
            raise ValueError('parent checkpoint changed')
        parents[f'{row["method"]}/{row["stage"]}'] = row['checkpoint_sha256']
    return {'specification': c, 'base_binding_sha256': p.digest(base),
            'parent_root': str(PARENT), 'parent_receipt_sha256': p.sha256(receipt),
            'parent_checkpoint_sha256': parents,
            'source_sha256': {str(Path(__file__).relative_to(p.REPO)): p.sha256(__file__),
                              str(CONTRACT_FILE.relative_to(p.REPO)): p.sha256(CONTRACT_FILE)}}


def checked_binding(root, base):
    """Evaluation rejects changed continuation sources, ancestry or an incomplete fit."""
    root = p.output_path(base['config'], root)
    done = json.loads((root/'CONTINUATION_COMPLETE.json').read_text())
    expected = contract(base)
    if (done['training_complete'] is not True or done['contract'] != expected or
            len(done['stages']) != 4 or any(x['updates'] != 384 for x in done['stages'])):
        raise ValueError('continuation receipt mismatch')
    return {**base, 'continuation': expected}


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a == b


def advance(c, ds, binding, state, stage, method, out, stop, device):
    """Same train_update/order/RNG primitives; a separate immutable continuation binding."""
    if not 192 <= state['step'] < stop <= 384:
        raise ValueError('continuation must advance within [192,384]')
    cfg = c['training']
    out.mkdir(parents=True, exist_ok=False)
    model = p.build_model(c, stage, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator(device=device)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    p.restore_rng(state['rng'], generator)
    if not equal_state(model.state_dict(), state['model']) or not equal_state(optimizer.state_dict(), state['optimizer']):
        raise ValueError('loaded model/optimizer differs')
    if not equal_state(p.rng_state(generator), state['rng']):
        raise ValueError('loaded RNG differs')
    history = copy.deepcopy(state['history'])
    if len(history) != state['step']:
        raise ValueError('history/update mismatch')
    p.write_json(out/'RUN.json', {'binding': binding, 'start_step': state['step'],
                 'stop_step': stop, 'stage': stage, 'method': method,
                 'slurm_job_id': os.environ['SLURM_JOB_ID']})
    start = time.monotonic()
    for step in range(state['step'], stop):
        item = ds[p.example_index(step, len(ds), cfg['seed'])]
        record = p.train_update(model, optimizer, p.tensor(item[stage+'_target'], device),
            p.tensor(item[stage+'_condition'], device), method, generator, cfg['clip_gradient_norm'],
            p.tensor(item['coarse_condition'], device) if stage == 'fine' else None)
        record.update(step=step+1, anchor_id=item['anchor_id'])
        history.append(record)
        print(json.dumps({'method':method,'stage':stage,**record},allow_nan=False),flush=True)
        if (step+1) % cfg['checkpoint_every'] == 0 or step+1 == stop:
            p.save_checkpoint(out/f'step_{step+1:06d}.pt',model=model,optimizer=optimizer,
                generator=generator,binding=binding,stage=stage,method=method,step=step+1,history=history)
    p.write_json(out/'CONTINUED_STAGE_COMPLETE.json', {'updates':stop,'elapsed_seconds':time.monotonic()-start,
                 'history':history,'training_ready':False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    device = p.runtime()
    c,_,_,_ = preflight()
    ds = p.dataset_for(c,NORMALIZATION)
    base = p.provenance(c,ds)
    extension = contract(base)
    binding = {**base,'continuation':extension}
    root = p.output_path(c,args.output)
    root.mkdir(parents=True,exist_ok=False)
    p.write_json(root/'CONTINUATION_STARTED.json', {'contract':extension,
        'job_id':os.environ['SLURM_JOB_ID'],'node':socket.gethostname(),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.REPO,text=True).strip()})
    start = time.monotonic()
    # Fail closed: exact two-update versus one+one parity for all four fits,
    # using the actual 192-update checkpoints, before any production continuation.
    smoke = []
    for method in ('cfm','diffusion'):
        for stage in ('coarse','fine'):
            state = p.load_checkpoint(PARENT/f'{method}_{stage}/step_000192.pt',base,stage,method)
            for name,stop in [('direct',194),('split',193)]:
                advance(c,ds,binding,state,stage,method,root/f'smoke_{method}_{stage}_{name}',stop,device)
            mid = p.load_checkpoint(root/f'smoke_{method}_{stage}_split/step_000193.pt',binding,stage,method)
            advance(c,ds,binding,mid,stage,method,root/f'smoke_{method}_{stage}_resumed',194,device)
            states = [p.load_checkpoint(root/f'smoke_{method}_{stage}_{n}/step_000194.pt',binding,stage,method)
                      for n in ('direct','resumed')]
            if not equal_state(states[0],states[1]):
                raise ValueError(f'continuation parity failed: {method}/{stage}')
            smoke.append({'method':method,'stage':stage,'exact_state_parity':True})
            print(f'CONTINUATION PARITY PASS {method}/{stage}',flush=True)
    p.write_json(root/'RESUME_SMOKE_COMPLETE.json',{'checks':smoke,'technical_pass':True})
    results = []
    for method in ('cfm','diffusion'):
        for stage in ('coarse','fine'):
            preflight()
            if contract(base) != extension:
                raise ValueError('continuation provenance changed')
            state = p.load_checkpoint(PARENT/f'{method}_{stage}/step_000192.pt',base,stage,method)
            dest = root/f'{method}_{stage}'
            advance(c,ds,binding,state,stage,method,dest,384,device)
            final = p.load_checkpoint(dest/'step_000384.pt',binding,stage,method)
            if final['step'] != 384 or not equal_state(final['history'][:192],state['history']):
                raise ValueError('continuation history corrupted')
            results.append({'method':method,'stage':stage,'updates':384,
                'checkpoint':str(dest/'step_000384.pt'),'checkpoint_sha256':p.sha256(dest/'step_000384.pt')})
    preflight()
    if contract(base) != extension:
        raise ValueError('continuation provenance changed')
    p.write_json(root/'CONTINUATION_COMPLETE.json',{'contract':extension,'stages':results,
        'elapsed_seconds':time.monotonic()-start,'training_complete':True,'training_ready':False,
        'heldout_payloads_read':False,'calibration_pass':None})
    print('CONTINUATION COMPLETE: four fits at 384; no automatic extension',flush=True)


if __name__ == '__main__':
    main()
