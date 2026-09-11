"""Run the user-authorized matched 192-update research canary, without holdouts."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from workflows.sbi import e2e_wide_pipeline as p

SMOKE = p.REPO/'docs/evidence/e2e_field_v2/wide_gpu_smoke_20260911/SMOKE_COMPLETE.json'
NORMALIZATION = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/'
                     'smoke_20260911_58196582/normalization.json')


def preflight():
    c = p.read_config()
    receipt = json.loads(SMOKE.read_text())
    if receipt['technical_pass'] is not True or receipt['heldout_payloads_read'] is not False:
        raise ValueError('missing qualified engineering receipt')
    source = {k: p.sha256(p.REPO/k) for k in p.SOURCE_FILES}
    if (source != receipt['registration']['source_sha256'] or
            p.digest(c) != receipt['registration']['config_sha256'] or
            p.sha256(NORMALIZATION) != receipt['normalization_sha256']):
        raise ValueError('source/config/normalization differs from the passed GPU smoke')
    ds = p.dataset_for(c, NORMALIZATION, verify=False)
    if c['training']['maximum_updates'] != 192:
        raise ValueError('this launch is exactly the registered 192-update canary')
    return c, ds, receipt, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true', help='metadata-only preflight; no compute or writes')
    args = parser.parse_args()
    c, ds, receipt, source = preflight()
    if args.check:
        print(json.dumps({'preflight_pass': True, 'parents': len(ds), 'config_sha256': p.digest(c)}))
        return
    if args.output is None:
        parser.error('--output is required for training')
    p.runtime()
    out = p.output_path(c, args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    registration = {'job_id': os.environ['SLURM_JOB_ID'], 'node': socket.gethostname(),
                    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=p.REPO, text=True).strip(),
                    'runner_sha256': p.sha256(__file__), 'source_sha256': source,
                    'config_sha256': p.digest(c), 'normalization_sha256': p.sha256(NORMALIZATION),
                    'normalization_path': str(NORMALIZATION), 'smoke_receipt_sha256': p.sha256(SMOKE),
                    'research_canary_authorized': True, 'authorization': 'user: Begin training',
                    'updates_per_stage': 192, 'seed': 42, 'start': 'fresh; not resumed from smoke',
                    'phases': ['ph000', 'ph002', 'ph003'], 'heldout_access_authorized': False,
                    'training_ready': False, 'r0_physics_pass': False}
    p.write_json(out/'RESEARCH_CANARY_STARTED.json', registration)
    results = []
    for method in ('cfm', 'diffusion'):
        for stage in ('coarse', 'fine'):
            preflight()
            print(f'RESEARCH FIT START: {method}/{stage}, 192 updates', flush=True)
            dest = out/f'{method}_{stage}'
            p.train(c, NORMALIZATION, stage, method, dest, stop_after=192)
            done = json.loads((dest/'CANARY_COMPLETE.json').read_text())
            if done['updates'] != 192:
                raise ValueError('incomplete stage')
            results.append({'method': method, 'stage': stage, 'updates': 192,
                            'elapsed_seconds': done['elapsed_seconds'],
                            'checkpoint': str(dest/'step_000192.pt'),
                            'checkpoint_sha256': p.sha256(dest/'step_000192.pt')})
    preflight()
    p.write_json(out/'RESEARCH_CANARY_COMPLETE.json', {'registration': registration, 'stages': results,
                 'elapsed_seconds': time.monotonic()-started, 'training_complete': True,
                 'calibration_pass': None, 'training_ready': False, 'r0_physics_pass': False,
                 'claim': 'matched training-phase canary only; no held-out evaluation or production release'})
    print('RESEARCH CANARY COMPLETE: four stages; no calibration claim', flush=True)


if __name__ == '__main__':
    main()
