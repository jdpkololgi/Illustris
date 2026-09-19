"""Bounded train-only statistics collection; no partial usable normalizer."""
import argparse
import json
import time
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_normalization as norm


def run(seconds):
    c.require_compute()
    for item in json.loads((c.REPO/'SOURCE.json').read_text())['files']:
        if c.sha256(c.REPO/item['relative'])!=item['sha256']:
            raise ValueError('normalization worker source drift')
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline-900:
        result=norm.fit(collect_ready=True)
        if 'fit_phases' in result:
            print(json.dumps(dict(normalizer_complete=True,fit_phases=result['fit_phases'])),flush=True)
            return 0
        print(json.dumps(result),flush=True)
        time.sleep(60)
    return 75


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=3600)
    args=parser.parse_args(); raise SystemExit(run(args.seconds))
