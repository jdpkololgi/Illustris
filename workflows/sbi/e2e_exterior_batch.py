"""One deterministic, frozen four-draw production unit; no scheduler calls."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_exterior_campaign as campaign


def check_freeze(path):
    freeze = json.loads(path.read_text())
    if freeze.get('schema') != 'e2e-exterior-selection-freeze-v1':
        raise PermissionError('Frozen selection required')
    for name, sha in freeze['sources'].items():
        if c.sha256(c.REPO/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    if c.sha256(freeze['prior']['path']) != freeze['prior']['sha256']:
        raise ValueError('Frozen prior changed')
    return freeze


def completed(path, payload_key):
    marker = path/'COMPLETE.json'
    if not marker.exists():
        if path.exists():
            raise ValueError('Partial output requires explicit reviewed recovery: '+str(path))
        return None
    record = json.loads(marker.read_text())
    if not record.get('passed'):
        raise ValueError('Failed output cannot be reused')
    item = record[payload_key]
    if c.sha256(item['path']) != item['sha256']:
        raise ValueError('Completed payload changed')
    if c.sha256(path/'BINDING.json') != record['binding']:
        raise ValueError('Completed binding changed')
    return record


def run(phase, cap, first, freeze_path):
    c.require_compute(); freeze = check_freeze(freeze_path)
    if (phase not in freeze['final_phases']+['Loa'] or phase == 'ph001'
            or cap not in ('NGC', 'SGC') or first not in range(0, freeze['draws'], 4)):
        raise PermissionError('Unregistered production unit')
    root = campaign.ROOT
    cache = root/'fullcap_inputs'/phase/cap
    density = root/'fullcap_density'/phase/cap/f'd{first:03d}'
    tides = root/'fullcap_tides'/phase/cap/f'd{first:03d}'
    if not (cache/'PREPARED.json').exists():
        raise ValueError('Qualified targetless cache is not ready')
    # Separate subprocesses release the inference model and its CUDA/host
    # caches before the field completion allocates large FFT workspaces.
    record = completed(density, 'field')
    if record is None:
        subprocess.run([sys.executable, '-u', '-m', 'workflows.sbi.e2e_exterior_fullcap', 'run',
                        '--root', str(cache), '--out', str(density), '--first', str(first),
                        '--freeze', str(freeze_path)], check=True)
        record = completed(density, 'field')
    binding = json.loads((density/'BINDING.json').read_text())
    if (record['phase'] != phase or record['cap'] != cap or record['first'] != first
            or binding['freeze_sha256'] != c.sha256(freeze_path)):
        raise ValueError('Existing density is from a different production unit')
    result = completed(tides, 'output')
    if result is None:
        subprocess.run([sys.executable, '-u', '-m', 'workflows.sbi.e2e_exterior_cap_tides',
                        '--field-receipt', str(density/'COMPLETE.json'),
                        '--prior', freeze['prior']['path'], '--out', str(tides),
                        '--method', freeze['method'], '--padding', str(freeze['fullcap_padding']),
                        '--freeze', str(freeze_path)], check=True)
        result = completed(tides, 'output')
    binding = json.loads((tides/'BINDING.json').read_text())
    if (result['phase'] != phase or result['cap'] != cap or result['first'] != first
            or result['draws'] != 4 or result['method'] != freeze['method']
            or binding['freeze_sha256'] != c.sha256(freeze_path)):
        raise ValueError('Existing tides are from a different production unit')
    if (tides/'VERIFIED.json').exists():
        checked = json.loads((tides/'VERIFIED.json').read_text())
        if (not checked['passed'] or checked['complete_receipt']['sha256'] != c.sha256(tides/'COMPLETE.json')
                or checked['verifier_sha256'] != c.sha256(c.REPO/'workflows/sbi/e2e_exterior_cap_verify.py')):
            raise ValueError('Existing verification changed')
    else:
        subprocess.run([sys.executable, '-u', '-m', 'workflows.sbi.e2e_exterior_cap_verify', '--root', str(tides)], check=True)
    print(json.dumps(dict(phase=phase, cap=cap, first=first, production_unit_verified=True,
                          posterior_qualified=False, job=os.environ['SLURM_JOB_ID'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', required=True)
    p.add_argument('--cap', choices=['NGC', 'SGC'], required=True); p.add_argument('--first', type=int, required=True)
    p.add_argument('--freeze', type=Path, required=True); a = p.parse_args(); run(a.phase, a.cap, a.first, a.freeze)
