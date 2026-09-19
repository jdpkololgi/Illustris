"""Serial target phases in fresh processes, without changing bound FFT kernels."""
import argparse
import json
import subprocess
import sys
import time

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord


def run(seconds):
    c.require_compute()
    for item in json.loads((c.REPO / 'SOURCE.json').read_text())['files']:
        if c.sha256(c.REPO / item['relative']) != item['sha256']:
            raise ValueError('fresh-target snapshot drift')
    deadline = time.monotonic() + seconds
    for phase in c.ROLES:
        marker = coord.ROOT / 'targets' / phase / 'TARGETS_COMPLETE.json'
        if marker.exists():
            coord.verify_receipt(marker, payload=False)
            continue
        if not (c.ROOT / 'matter' / phase / 'DENSITY_COMPLETE.json').exists():
            raise RuntimeError('native completion required before target recovery')
        left = deadline - time.monotonic()
        if left < 1800:
            return 75
        print(json.dumps(dict(phase=phase, fresh_process=True, seconds_left=left)), flush=True)
        code = subprocess.run([
            sys.executable, '-u', '-m', 'workflows.sbi.e2e_coupled_target_products',
            '--phase', phase, '--workers', '32', '--stop-after-seconds', str(left - 900)
        ], check=False).returncode
        if code:
            return 75 if code == 75 else 1
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.seconds))
