"""Bounded strict-determinism diagnostic; disposable updates only."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
p = argparse.ArgumentParser()
p.add_argument('--scope', choices=['endpoint', 'encoder'], required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(4)
report = {'scope': a.scope, 'strict_determinism': True, 'pass': False}
try:
    if a.scope == 'endpoint':
        from workflows.sbi import e2e_wholefield_unet_readiness as audit
        from workflows.sbi import e2e_coupled_views as views
        meta = audit.metadata()
        report['result'] = audit.endpoint_smoke(a.output, views.load_chart(meta['normalizer_sha256']))
    else:
        # Patch subprocess dispatch so the exact trainer starts in a fresh Python
        # process with strict algorithms enabled before model construction.
        from workflows.sbi import p12a_field_training_readiness as audit
        original = audit.subprocess.run
        bootstrap = "import torch,runpy; torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.set_num_threads(4); runpy.run_module('workflows.abacus_tweb.p10_train_arm_a',run_name='__main__')"
        def strict_run(command, **kwargs):
            return original([command[0], '-c', bootstrap] + command[3:], **kwargs)
        audit.subprocess.run = strict_run
        report['result'] = audit.compute(a.output, audit.metadata())
    report['pass'] = True
except Exception:
    report['traceback'] = traceback.format_exc()
finally:
    (a.output / 'DIAGNOSTIC.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
sys.exit(0 if report['pass'] else 1)
