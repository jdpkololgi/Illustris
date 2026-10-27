"""Actual fresh/resumed endpoint trainer, disposable two-update comparisons."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
root=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/restart_repair_20261026_v1/endpoint_entrypoint')
root.mkdir(parents=True,exist_ok=False)
base=[sys.executable,'-m','workflows.sbi.e2e_wholefield_pilot_train','--architecture','unet','--prediction','x',
      '--seed','17','--seconds','120','--source-revision','configs/field_restart_numerics_20261026.json']
for name,steps in [('restart',1),('restart',2),('uninterrupted',2)]:
    with (root/f'{name}_{steps}.log').open('x') as log:
        subprocess.run(base+['--output-root',str(root/name),'--stop-step',str(steps)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
a,b=[torch.load(root/name/'unet_x_17/LATEST.pt',map_location='cpu',weights_only=False) for name in ('restart','uninterrupted')]
diff={kind:max(float((a[kind][key]-b[kind][key]).abs().max()) for key in a[kind]) for kind in ('model','ema')}
report={'pass':a['step']==b['step']==2 and max(diff.values())<=2e-6,'difference':diff,'steps':[a['step'],b['step']],
        'scope':'actual trainer fresh and resumed, fixed selected inputs; disposable updates only'}
(root/'REPLAY.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
if not report['pass']:raise RuntimeError('Actual trainer replay failed')
