"""Bounded normalized-reader qualification after train normalization appears."""
import argparse
import json
import time

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_interface_qualify as qa
from workflows.sbi import e2e_coupled_loader_benchmark as loader
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_coupled_audit_worker as audits


def finalize(publish_data_release=False):
    complete = qa.run()
    if complete['pass'] is not True:
        raise ValueError('full normalized interface qualification did not pass')
    print(json.dumps(dict(interface_complete=True, pairs=complete['pairs'],
                          offset_cases=complete['offset_cases'])), flush=True)
    if publish_data_release:
        from workflows.sbi import e2e_coupled_data_release as release
        record = release.run()
        print(json.dumps(dict(data_products_qualified=record['data_products_qualified'],
                              scientific_training_authorized=False,
                              full_preparation_complete=False)), flush=True)
    return 0


def run(seconds, publish_data_release=False):
    c.require_compute()
    for item in json.loads((c.REPO / 'SOURCE.json').read_text())['files']:
        if c.sha256(c.REPO / item['relative']) != item['sha256']:
            raise ValueError('interface worker source snapshot changed')
    deadline = time.monotonic() + seconds
    normalizer = coord.ROOT / 'normalization' / 'NORMALIZATION_COMPLETE.json'
    while time.monotonic() < deadline - 1200:
        if not normalizer.exists():
            print(json.dumps(dict(interface_waiting_for='complete_13_phase_normalizer')), flush=True)
            time.sleep(60)
            continue
        normalizer_hash = c.sha256(normalizer)
        chart = views.load_chart(normalizer_hash)
        # All reader work remains numerical preparation, including confirmation
        # roles. The loader timing probe rejects every non-training phase.
        for phase in c.ROLES:
            if time.monotonic() >= deadline - 1200:
                return 75
            if audits.qualified(phase):
                qa.qualify_phase(phase, chart, normalizer_hash)
        if time.monotonic() >= deadline - 1200:
            return 75
        try:
            measured = loader.run()
        except BlockingIOError:
            # The normalizer may still be releasing its publication lock.
            time.sleep(60)
            continue
        print(json.dumps(dict(loader_seconds_per_pair=[
            row['seconds_per_pair_including_validation'] for row in measured['measurements']])), flush=True)
        remaining = [phase for phase in c.ROLES if not audits.qualified(phase)]
        if not remaining:
            return finalize(publish_data_release)
        print(json.dumps(dict(interface_waiting_for_audits=remaining)), flush=True)
        time.sleep(60)
    return 75


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=13800)
    parser.add_argument('--publish-data-release', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run(args.seconds, args.publish_data_release))
