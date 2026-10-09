"""Bounded metadata-only full-cap inventory and resource estimate.

Reads only explicitly addressed ph012--017 JSON receipts and file metadata.
Does not load targets, field payloads, models, CUDA, or submit scheduler jobs.
Reuse candidates must still pass payload/source/parity checks on compute nodes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
BASE = ROOT/'coherent_exterior_20261002_v1'
MIX = ROOT/'cfm_model_mixture_20261006_v2'
PHASES = tuple(f'ph{i:03d}' for i in range(12, 18))


def inventory():
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    tasks, timings, inputs = [], [], []
    for phase in PHASES:
        for cap in ('NGC', 'SGC'):
            prepared_path = BASE/'fullcap_inputs'/phase/cap/'PREPARED.json'
            prepared = read(prepared_path)
            assert prepared['phase'] == phase and prepared['cap'] == cap and prepared['targetless']
            plan_path = prepared_path.parent/'PLAN.json'
            plan = read(plan_path)
            assert hashes[str(plan_path)] == prepared['plan_sha256']
            windows = len(plan['midpoints'])
            assert all(Path(item['path']).is_file() for item in prepared['outputs'])
            inputs.append(dict(phase=phase, cap=cap, windows=windows,
                               prepared=str(prepared_path), payload_hashes_verified=False))
            for seed in (17, 29):
                for first in range(0, 32, 4):
                    if seed == 17:
                        density = BASE/'fullcap_density'/phase/cap/f'd{first:03d}'
                        tides = BASE/'fullcap_tides'/phase/cap/f'd{first:03d}'
                    else:
                        parent = MIX/phase/cap/'seed29'/f'd{first:03d}'
                        density, tides = parent/'density', parent/'tides'
                    task = dict(phase=phase, cap=cap, seed=seed, first=first, windows=windows)
                    for label, path, key in [('density', density, 'field'), ('tides', tides, 'output')]:
                        marker = path/'COMPLETE.json'
                        complete = marker.is_file()
                        task[label+'_receipt'] = str(marker)
                        task[label+'_complete_candidate'] = complete
                        task[label+'_partial_directory'] = path.exists() and not complete
                        if not complete:
                            continue
                        rec = read(marker)
                        assert rec['passed'] and rec['phase'] == phase and rec['cap'] == cap
                        assert rec['first'] == first and rec['draws'] == 4
                        assert Path(rec[key]['path']).is_file()
                        assert Path(rec[key]['path']).stat().st_size == rec[key]['bytes']
                        task[label+'_bytes'] = rec[key]['bytes']
                        if label == 'density':
                            assert rec['windows'] == windows
                            seconds = rec['sampling_seconds']+rec['loading_seconds']
                        else:
                            # CapTidal's total seconds already includes setup.
                            seconds = rec['seconds']
                        timings.append(dict(phase=phase, cap=cap, seed=seed, first=first,
                                            product=label, seconds=seconds, windows=windows,
                                            bytes=rec[key]['bytes']))
                    tasks.append(task)
    rates = {}
    for cap in ('NGC', 'SGC'):
        for product in ('density', 'tides'):
            rows = [r for r in timings if r['cap'] == cap and r['product'] == product]
            rates[cap+'/'+product] = dict(
                n=len(rows),
                median_seconds_per_window=statistics.median(r['seconds']/r['windows'] for r in rows),
                max_seconds_per_window=max(r['seconds']/r['windows'] for r in rows),
                median_bytes_per_window=statistics.median(r['bytes']/r['windows'] for r in rows))
    summary = []
    totals = dict(missing_density=0, missing_tides=0,
                  estimate_median_gpu_hours=0., estimate_max_rate_gpu_hours=0.,
                  estimated_added_gib=0.)
    for phase in PHASES:
        for seed in (17, 29):
            group = [t for t in tasks if t['phase'] == phase and t['seed'] == seed]
            row = dict(phase=phase, seed=seed, density_reuse_candidates=0,
                       tidal_reuse_candidates=0, estimate_median_gpu_hours=0.,
                       estimate_max_rate_gpu_hours=0., estimated_added_gib=0.)
            for task in group:
                for product in ('density', 'tides'):
                    if task[product+'_complete_candidate']:
                        row['density_reuse_candidates' if product == 'density' else 'tidal_reuse_candidates'] += 1
                        continue
                    totals['missing_'+product] += 1
                    rate = rates[task['cap']+'/'+product]
                    row['estimate_median_gpu_hours'] += task['windows']*rate['median_seconds_per_window']/3600
                    row['estimate_max_rate_gpu_hours'] += task['windows']*rate['max_seconds_per_window']/3600
                    row['estimated_added_gib'] += task['windows']*rate['median_bytes_per_window']/2**30
            for key in ('estimate_median_gpu_hours', 'estimate_max_rate_gpu_hours', 'estimated_added_gib'):
                totals[key] += row[key]
            summary.append(row)
    return dict(schema='wholefield-fullcap-metadata-preflight-v1',
                created_utc=datetime.now(timezone.utc).isoformat(), phases=list(PHASES),
                seeds=[17, 29], draws=32, nfe=128, tasks=tasks, inputs=inputs,
                rates=rates, summary=summary, totals=totals, source_hashes=hashes,
                payload_hashes_verified=False, permission_to_launch=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                exclusions=['Payload/source/parity validation', 'Corrected-operator extraction',
                            'Gate 2 sensitivity and full-rule calibration',
                            'Allocation idle/tail time, queue delay, failures and retries',
                            'Gaussian gate and any model training'],
                estimate_note='Planning estimates from completed task timings, scaled by prepared window counts; not a measured full new run or guaranteed upper bound.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: result[k] for k in ('summary', 'totals', 'exclusions', 'estimate_note')}))
