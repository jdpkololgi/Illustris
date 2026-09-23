"""Post-run finite-state, optimizer-step and paired-random-stream audit."""
import argparse
import json
from pathlib import Path
import torch
from workflows.sbi.e2e_conditional_reference import atomic_json
from workflows.sbi.e2e_conditional_reference_continue import digest


def audit(root,kind):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());records=[];streams={}
    for relative,h in m['sources'].items():
        if digest(root/'source'/relative)!=h:raise ValueError('frozen source drift')
    update=65536 if kind=='adaptive' else 16384
    for item in m['items']:
        p=root/'results'/item['name']/f'checkpoint_{update}.pt'
        s=torch.load(p,map_location='cpu',weights_only=False)
        if s['item']!=item or s['sources']!=m['sources'] or s['update']!=update:
            raise ValueError('checkpoint identity mismatch')
        if any(not torch.isfinite(v).all() for v in s['model'].values()):
            raise ValueError('nonfinite learned state')
        steps={int(v['step']) for v in s['optimizer']['state'].values()}
        continuation=kind=='target' and not item['branch'].startswith('affine')
        expected=81920 if continuation else update
        if steps!={expected}:raise ValueError(f'optimizer steps {steps} != {expected}')
        rate=3e-6 if kind=='target' and item['branch']=='decay' else 3e-4
        if any(abs(g['lr']-rate)>1e-14 for g in s['optimizer']['param_groups']):
            raise ValueError('unexpected final learning rate')
        if continuation:
            key=item['parent']
            if key in streams and not torch.equal(streams[key],s['generator']):
                raise ValueError('paired branches consumed different random streams')
            streams[key]=s['generator']
        records.append(dict(name=item['name'],checkpoint_sha256=digest(p),optimizer_steps=expected,
                            final_lr=rate,finite=True))
    return dict(kind=kind,records=records,paired_parent_streams=len(streams),sources=m['sources'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--resolution',required=True)
    p.add_argument('--affine');p.add_argument('--output',required=True);a=p.parse_args()
    results=[audit(a.target,'target'),audit(a.resolution,'resolution')]
    if a.affine:results.append(audit(a.affine,'adaptive'))
    atomic_json(a.output,results);print('AUDIT PASS',sum(len(x['records']) for x in results),'checkpoints')
