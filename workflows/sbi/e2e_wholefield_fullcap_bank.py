"""Assemble the complete registered baseline; never score a partial panel."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from workflows.sbi import e2e_wholefield_fullcap_extract as extract
from workflows.sbi import e2e_wholefield_fullcap_run as gen
from workflows.sbi.e2e_wholefield_sensitivity import fine_vector
from workflows.sbi.e2e_vdm_context_metrics import fair_energy

ROOT = gen.ROOT
OUT = ROOT/'baseline_v1'


def terms(draws, truth):
    """Equal-pair scalar traces and proper score, retaining leading axes."""
    x, y = np.asarray(draws), np.asarray(truth)
    if x.shape[-3:] != (16, 32, 6) or y.shape != x.shape[:-2]+(6,):
        raise ValueError('Six-dimensional, 16-pair, 32-draw panel required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite baseline')
    v = x.var(-2, ddof=1).sum(-1).mean(-1)
    q = ((x.mean(-2)-y)**2).sum(-1).mean(-1)-v/32
    energy = fair_energy(np.moveaxis(x, -2, 0), y).mean(-1)
    return q, v, energy


def pooled(q, v, energy):
    q, v, energy = map(np.asarray, (q, v, energy))
    if q.shape != v.shape or energy.shape != q.shape or np.any(v <= 0):
        raise ValueError('Matching positive-variance terms required')
    # Ratio of pooled traces, not mean of phase/seed ratios.
    return dict(q=float(q.mean()), v=float(v.mean()), ratio=float(q.mean()/v.mean()),
                fair_energy=float(energy.mean()), phase_seed_terms=int(q.size))


def run():
    gen.compute(); started = time.monotonic()
    manifest = gen.read(ROOT/'MANIFEST.json'); gen.contract.validate(manifest)
    extraction = gen.read(extract.OUT/'MANIFEST.json')
    gen.verify_record(extraction['source']); gen.verify_record(extraction['generation'])
    expected = {t['key'] for t in manifest['tasks']}
    if len(expected) != 192:
        raise ValueError('Wrong task panel')
    missing = sorted(k for k in expected if not (extract.OUT/'cores'/k/'COMPLETE.json').exists())
    if missing:
        raise ValueError(f'Baseline requires all 192 tasks; missing {len(missing)}: {missing[:4]}')
    OUT.mkdir(exist_ok=False)
    sources = {str(Path(module.__file__).resolve()): gen.sha(module.__file__)
               for module in (extract, gen)}
    sources[str(Path(__file__).resolve())] = gen.sha(__file__)
    for fn in (fine_vector, fair_energy):
        path = Path(fn.__code__.co_filename).resolve(); sources[str(path)] = gen.sha(path)
    gen.write(OUT/'BINDING.json', dict(generation=gen.record(ROOT/'MANIFEST.json'),
              extraction=gen.record(extract.OUT/'MANIFEST.json'), sources=sources,
              scope='Full-cap frozen baseline, not the historical pair-local draw law',
              job=os.environ['SLURM_JOB_ID']))
    x = np.empty((6, 2, 16, 32, 6)); y = np.empty((6, 2, 16, 6))
    ids = np.empty((6, 16), dtype='U128'); inputs = []
    for pi, phase in enumerate(extract.PHASES):
        for si, seed in enumerate((17, 29)):
            for ci, cap in enumerate(('NGC', 'SGC')):
                section = slice(ci*8, (ci+1)*8)
                saved_truth = saved_ids = None
                for first in range(0, 32, 4):
                    key = extract.address(phase, cap, seed, first)
                    receipt = extract.OUT/'cores'/key/'COMPLETE.json'
                    rec = gen.read(receipt)
                    if not rec['passed'] or rec['key'] != key or rec['draws'] != 4:
                        raise ValueError('Wrong extraction receipt')
                    gen.verify_record(rec['binding']); gen.verify_record(rec['output'])
                    bound = gen.read(rec['binding']['path']); gen.verify_record(bound['extraction'])
                    if bound['extraction'] != gen.record(extract.OUT/'MANIFEST.json'):
                        raise ValueError('Extraction binding mismatch')
                    with np.load(rec['output']['path'], allow_pickle=False) as f:
                        if f['draws'].shape != (8, 4, 6) or f['truth'].shape != (8, 6):
                            raise ValueError('Wrong summary dimensions')
                        computed = fine_vector(f['density'], f['eigen'][1])[0]
                        truth = fine_vector(f['truth_density'], f['truth_eigen'])[0]
                        np.testing.assert_array_equal(computed, f['draws'])
                        np.testing.assert_array_equal(truth, f['truth'])
                        pair_ids = f['pair_id'].copy()
                        if len(set(pair_ids)) != 8 or any(not p.startswith(phase+'_'+cap+'_') for p in pair_ids):
                            raise ValueError('Duplicate or foreign pair')
                        if saved_truth is not None:
                            np.testing.assert_array_equal(saved_truth, truth)
                            np.testing.assert_array_equal(saved_ids, pair_ids)
                        saved_truth, saved_ids = truth, pair_ids
                        x[pi, si, section, first:first+4] = computed
                y[pi, si, section] = saved_truth
                if si == 0:
                    ids[pi, section] = saved_ids
                else:
                    np.testing.assert_array_equal(ids[pi, section], saved_ids)
                # All eight draw chunks have complete, independently bound addresses.
                inputs.extend(gen.record(extract.OUT/'cores'/extract.address(phase, cap, seed, first)/'COMPLETE.json')
                              for first in range(0, 32, 4))
            np.testing.assert_array_equal(y[pi, 0], y[pi, si])
        print(json.dumps(dict(stage='baseline_phase_verified', phase=phase)), flush=True)
    q, v, energy = terms(x, y)
    records = []
    for pi, phase in enumerate(extract.PHASES):
        for si, seed in enumerate((17, 29)):
            records.append(dict(phase=phase, seed=seed, pairs=16, draws=32,
                                **pooled(q[pi, si], v[pi, si], energy[pi, si])))
    reports = dict(all=pooled(q, v, energy), out_of_fit=pooled(q[2:], v[2:], energy[2:]),
                   selection=pooled(q[:2], v[:2], energy[:2]),
                   seeds={str(seed): pooled(q[:, i], v[:, i], energy[:, i]) for i, seed in enumerate((17, 29))})
    payload = OUT/'PANEL.npz'
    with payload.open('xb') as stream:
        np.savez_compressed(stream, draws=x, truth=y, pair_id=ids, phases=np.array(extract.PHASES),
                            seeds=np.array([17, 29]), q=q, v=v, fair_energy=energy)
    for path, digest in sources.items():
        if gen.sha(path) != digest:
            raise ValueError('Analysis source changed during assembly')
    gen.write(OUT/'COMPLETE.json', dict(passed=True, cases=192, phases=6, seeds=2, pairs_per_phase=16,
              draws_per_case=32, rows=records, pooled=reports, inputs=inputs,
              output=gen.record(payload), binding=gen.record(OUT/'BINDING.json'),
              generation=gen.record(ROOT/'MANIFEST.json'), seconds=time.monotonic()-started,
              job=os.environ['SLURM_JOB_ID'], gate2_passed=None, posterior_qualified=False))
    print(json.dumps(dict(stage='baseline_complete', rows=records, pooled=reports)), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
