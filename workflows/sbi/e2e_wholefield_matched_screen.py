"""Six-phase/two-seed necessary-event screen: can reject, never pass Gate 2.

The frozen minimum point improvement and seed floor are necessary for full
advancement. Existential selection over all candidates/checkpoints relaxes the
actual selector. All additional requirements can only reduce its probability.
"""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1')
BANK = ROOT/'fullcap_v1/baseline_v1'
OUT = ROOT/'matched_gate2_20261013_v1'
CONFIG = REPO/'configs/e2e_wholefield_matched_gate2_20261013.json'


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(rec):
    if record(rec['path']) != rec:
        raise ValueError('Changed input: '+rec['path'])


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def guard():
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('Compute allocation required')
    if os.environ.get('PYTHONNOUSERSITE') != '1' or any(os.environ.get(k) for k in
            ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'LD_PRELOAD')):
        raise RuntimeError('Isolated cosmic_env required')


def ratio(q, v):
    q, v = np.asarray(q), np.asarray(v)
    if q.shape != v.shape or np.any(v <= 0):
        raise ValueError('Matching positive variance required')
    return q.mean(axis=(-2, -1))/v.mean(axis=(-2, -1))


def necessary_event(base, candidates, floor, margin=.1):
    # base trial; candidates configuration, checkpoint, trial.
    a, b = np.asarray(base), np.asarray(candidates)
    if a.ndim != 1 or b.ndim != 3 or b.shape[-1] != len(a):
        raise ValueError('Incorrect event axes')
    gain = abs(a-1)[None, None]-abs(b-1)
    return ((gain >= margin*abs(a-1)[None, None]) & (gain >= floor)).any((0, 1))


def complete_rule(reference, candidates, ratio_floor, score_floor):
    """Frozen rule on per-phase/seed terms; NOT a claim of null calibration.

    Reference Q,V,energy: (mode,phase,seed). Candidate: (arm,mode,checkpoint,
    phase,seed). Mode 0=true coarse, 1=sampled coarse. Coverage and CRPS add a
    final six-coordinate axis. Every bootstrap retains both seeds and all terms.
    """
    rq, rv, re = (np.asarray(reference[k]) for k in ('q', 'v', 'energy'))
    cq, cv, ce = (np.asarray(candidates[k]) for k in ('q', 'v', 'energy'))
    if rq.shape != (2, 6, 2) or cq.shape != (2, 2, 4, 6, 2):
        raise ValueError('Full rule requires both modes and the complete panel')
    if any(a.shape != rq.shape for a in (rv, re)) or any(a.shape != cq.shape for a in (cv, ce)):
        raise ValueError('Inconsistent primary term shapes')
    if not all(np.isfinite(a).all() for a in (rq, rv, re, cq, cv, ce)) or np.any(rv <= 0) or np.any(cv <= 0):
        raise ValueError('Finite terms and positive variances required')
    for key in ('coverage', 'crps'):
        if np.shape(reference[key]) != (2, 6, 2, 6) or np.shape(candidates[key]) != (2, 2, 4, 6, 2, 6):
            raise ValueError('Coordinatewise guardrails required')
        if not np.isfinite(reference[key]).all() or not np.isfinite(candidates[key]).all():
            raise ValueError('Nonfinite guardrails')
    bootstrap = np.array(list(itertools.product(range(6), repeat=6)))
    selected = np.argmin(ce[:, 1, :, :2, :].mean((-2, -1)), axis=1)
    passed = []
    for arm, checkpoint in enumerate(selected):
        okay = True
        for h in (int(checkpoint), int(checkpoint)-1 if checkpoint else 1):
            for mode in range(2):
                ra = float(ratio(rq[mode], rv[mode])); rb = float(ratio(cq[arm, mode, h], cv[arm, mode, h]))
                gain = abs(ra-1)-abs(rb-1)
                okay &= gain >= max(.1*abs(ra-1), ratio_floor)
                okay &= re[mode].mean()-ce[arm, mode, h].mean() > score_floor
                bra = ratio(rq[mode][bootstrap], rv[mode][bootstrap])
                brb = ratio(cq[arm, mode, h][bootstrap], cv[arm, mode, h][bootstrap])
                ratio_margin = abs(bra-1)-abs(brb-1)-np.maximum(.1*abs(bra-1), ratio_floor)
                score_margin = (re[mode][bootstrap]-ce[arm, mode, h][bootstrap]).mean((-2, -1))-score_floor
                okay &= np.quantile(ratio_margin, .05/16) > 0 and np.quantile(score_margin, .05/16) > 0
                # Coverage compares distance from nominal content, not a rise in raw coverage.
                acov = np.asarray(reference['coverage'])[mode].mean((0, 1))
                bcov = np.asarray(candidates['coverage'])[arm, mode, h].mean((0, 1))
                okay &= np.all(abs(bcov-.9) <= abs(acov-.9)+.02)
                acrps = np.asarray(reference['crps'])[mode].mean((0, 1))
                bcrps = np.asarray(candidates['crps'])[arm, mode, h].mean((0, 1))
                okay &= np.all(bcrps <= 1.02*acrps)
        passed.append(bool(okay))
    return dict(selected=selected.tolist(), arms_passed=passed, passed=any(passed))


def bound(events, comparisons, error):
    n = len(events); p = float(np.mean(events))
    radius = math.sqrt(math.log(comparisons/error)/(2*n))
    return dict(successes=int(np.sum(events)), trials=n, point=p,
                lower=max(0., p-radius), upper=min(1., p+radius), radius=radius)


def fixtures(x, y, cfg, common=False):
    result = []
    for p in range(6):
        flat = x[p].transpose(0, 2, 1, 3).reshape(2, 32, 96)
        cov = sum(np.cov(z, rowvar=False) for z in flat)/2
        cov = (1-cfg['covariance_shrinkage'])*cov + cfg['covariance_shrinkage']*np.diag(np.diag(cov))
        direction = (flat.mean((0, 1))-y[p, 0].ravel())
        direction /= np.linalg.norm(direction)
        if common:
            f = cfg['common_mode_trace_fraction']
            cov = (1-f)*cov + f*np.trace(cov)*np.outer(direction, direction)
        eigen, vectors = np.linalg.eigh(cov/np.trace(cov))
        if eigen.min() < -1e-12:
            raise ValueError('Non-PSD fixture')
        eigen = np.maximum(eigen, 0)
        v = flat.var(1, ddof=1).sum(-1)
        q = ((flat.mean(1)-y[p].reshape(2, 96))**2).sum(-1)-v/32
        if np.any(q <= 0):
            raise ValueError('Positive reference error traces needed')
        result.append(dict(eigen=eigen.tolist(), direction=(direction@vectors).tolist(),
                           q=q.tolist(), v=v.tolist(), effective_rank=float(1/sum(eigen**2))))
    return result


def prepare():
    guard(); cfg = read(CONFIG); rec = read(BANK/'COMPLETE.json')
    if not rec['passed'] or (rec['cases'], rec['phases'], rec['seeds'], rec['draws_per_case']) != (192, 6, 2, 32):
        raise ValueError('Complete matched bank required')
    verify(rec['output']); verify(rec['binding']); verify(rec['generation'])
    for item in rec['inputs']:
        verify(item)
    with np.load(rec['output']['path'], allow_pickle=False) as f:
        x, y = f['draws'], f['truth']
        if x.shape != (6, 2, 16, 32, 6) or y.shape != (6, 2, 16, 6):
            raise ValueError('Wrong panel axes')
        if f['phases'].tolist() != cfg['phases'] or f['seeds'].tolist() != cfg['seeds']:
            raise ValueError('Wrong phases/seeds')
        np.testing.assert_array_equal(y[:, 0], y[:, 1])
        if len(set(f['pair_id'].ravel())) != 96:
            raise ValueError('Pair identity mismatch')
        # Independent direct formula, not the baseline assembly helper.
        v = np.mean(np.sum(np.var(x, axis=3, ddof=1), axis=3), axis=2)
        q = np.mean(np.sum((np.mean(x, axis=3)-y)**2, axis=3), axis=2)-v/32
        np.testing.assert_allclose(q, f['q'], rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(v, f['v'], rtol=1e-13, atol=1e-15)
        energy = f['fair_energy'].copy()
        # Independent fair energy evaluation on every case.
        check = np.empty((6, 2))
        for p in range(6):
            for seed in range(2):
                z = x[p, seed]; yy = y[p, seed]
                first = np.linalg.norm(z-yy[:, None], axis=-1).mean()
                dist = np.linalg.norm(z[:, :, None]-z[:, None, :], axis=-1)
                check[p, seed] = first - dist.sum()/(16*2*32*31)
        np.testing.assert_allclose(check, energy, rtol=1e-12, atol=1e-14)
    seed_r = q.mean(0)/v.mean(0)
    floor = abs(abs(seed_r[0]-1)-abs(seed_r[1]-1))/math.sqrt(2)
    score_floor = abs(energy[:, 0].mean()-energy[:, 1].mean())/math.sqrt(2)
    OUT.mkdir(exist_ok=False)
    binding = dict(config=record(CONFIG), source=record(__file__), bank=record(BANK/'COMPLETE.json'),
                   panel=rec['output'], commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                   job=os.environ['SLURM_JOB_ID'], floor=floor, score_floor=score_floor,
                   baseline_ratio=float(ratio(q, v)), seed_ratios=seed_r.tolist(),
                   out_of_fit_ratio=float(ratio(q[2:], v[2:])),
                   fixtures={family: fixtures(x, y, cfg, family == 'common_mode') for family in cfg['families']},
                   complete_rule_calibrated=False, source_hashes_verified=True)
    write(OUT/'BINDING.json', binding)
    print(json.dumps({k: binding[k] for k in ('baseline_ratio', 'seed_ratios', 'floor', 'score_floor', 'out_of_fit_ratio')}), flush=True)


def sample_stats(eps, eigen, radial):
    z = eps*np.sqrt(eigen)
    if radial is not None:
        z = z*radial[..., None]
    return z.mean(-2), z.var(-2, ddof=1).sum(-1)


def simulate(task):
    family, corr, ff, cfg, seed = task
    rng = np.random.default_rng(seed); n = cfg['replicates']; effects = cfg['effects']
    q0 = np.array([f['q'] for f in ff]); v0 = np.array([f['v'] for f in ff])
    r0 = float(ratio(q0, v0)); targets = 1+(1-np.array(effects))*(r0-1)
    scales = r0/targets
    tq = np.min(q0, axis=1)*.5
    bias_scale = np.sqrt((targets*v0.mean()-tq.mean())/(q0.mean()-tq.mean()))
    base_q = np.empty((n, 6, 2)); base_v = np.empty_like(base_q)
    cand_q = np.empty((len(effects), 2, 4, n, 6, 2)); cand_v = np.empty_like(cand_q)
    exact_b = np.empty_like(base_q); exact_c = np.empty((len(effects), n, 6, 2))
    for start in range(0, n, cfg['batch_size']):
        count = min(cfg['batch_size'], n-start); sl = slice(start, start+count)
        for p, f in enumerate(ff):
            lam = np.array(f['eigen']); direction = np.array(f['direction']); d = len(lam)
            # Shared phase realization, never independently shuffled regions/seeds.
            truth = rng.normal(size=(count, d))*np.sqrt(lam)
            if family == 'heavy_tail':
                truth *= np.sqrt(3/rng.chisquare(5, count))[:, None]
            for s in range(2):
                bias = direction*np.sqrt(q0[p, s]-tq[p]) if family == 'biased' else np.zeros(d)
                error = truth*np.sqrt(tq[p] if family == 'biased' else q0[p, s])+bias
                eps = rng.normal(size=(count, 32, d))
                radial = np.sqrt(3/rng.chisquare(5, (count, 32))) if family == 'heavy_tail' else None
                mu, var = sample_stats(eps, lam, radial)
                base_v[sl, p, s] = var*v0[p, s]
                base_q[sl, p, s] = np.sum((mu*np.sqrt(v0[p, s])-error)**2, axis=-1)-base_v[sl, p, s]/32
                exact_b[sl, p, s] = np.sum(error**2, axis=-1)
                for k in range(len(effects)):
                    err = truth*np.sqrt(tq[p])+bias*bias_scale[k] if family == 'biased' else error
                    exact_c[k, sl, p, s] = np.sum(err**2, axis=-1)
                for c in range(2):
                    shared = rng.normal(size=(count, 32, d))
                    for h in range(4):
                        independent = rng.normal(size=(count, 32, d))
                        z = corr*eps + np.sqrt(1-corr*corr)*(np.sqrt(.8)*shared+np.sqrt(.2)*independent)
                        bm, bv = sample_stats(z, lam, radial)
                        for k in range(len(effects)):
                            scale = 1. if family == 'biased' else scales[k]
                            err = truth*np.sqrt(tq[p])+bias*bias_scale[k] if family == 'biased' else error
                            cand_v[k, c, h, sl, p, s] = bv*v0[p, s]*scale
                            cand_q[k, c, h, sl, p, s] = np.sum((bm*np.sqrt(v0[p, s]*scale)-err)**2, axis=-1)-cand_v[k, c, h, sl, p, s]/32
    exact_v = np.broadcast_to(v0, exact_b.shape)
    candidate_exact_v = np.broadcast_to(v0, exact_c.shape).copy()
    if family != 'biased':
        candidate_exact_v *= scales[:, None, None, None]
    return dict(family=family, correlation=corr, base=ratio(base_q, base_v),
                candidates=ratio(cand_q, cand_v), oracle_base=ratio(exact_b, exact_v),
                oracle_candidates=ratio(exact_c, candidate_exact_v),
                # Ratios of aggregate MC moments check population effects.
                population_base=float(ratio(base_q.mean(0), base_v.mean(0))),
                population_candidates=ratio(cand_q.mean((1, 2, 3)), cand_v.mean((1, 2, 3))).tolist(),
                target_ratios=targets.tolist())


def run(workers):
    guard(); begin = time.monotonic(); cfg = read(CONFIG); binding = read(OUT/'BINDING.json')
    for name in ('config', 'source', 'bank', 'panel'):
        verify(binding[name])
    if not 1 <= workers <= cfg['maximum_workers']:
        raise ValueError('Worker limit')
    tasks = [(f, c, binding['fixtures'][f], cfg, cfg['seed']+100003*(i*3+j))
             for i, f in enumerate(cfg['families']) for j, c in enumerate(cfg['correlations'])]
    rows = []; arrays = {}; comparisons = len(tasks)*len(cfg['effects'])*2
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(simulate, tasks)):
            for key in ('base', 'candidates', 'oracle_base', 'oracle_candidates'):
                arrays[f'{i}_{key}'] = result[key]
            for k, effect in enumerate(cfg['effects']):
                event = necessary_event(result['base'], result['candidates'][k], binding['floor'])
                oracle = necessary_event(result['oracle_base'], result['oracle_candidates'][k][None, None], binding['floor'])
                rows.append(dict(family=result['family'], correlation=result['correlation'], effect=effect,
                                 finite=bound(event, comparisons, cfg['familywise_mc_error']),
                                 oracle=bound(oracle, comparisons, cfg['familywise_mc_error']),
                                 population_base=result['population_base'],
                                 population_candidate=result['population_candidates'][k], target=result['target_ratios'][k]))
            print(json.dumps(dict(stage='law_complete', family=result['family'], correlation=result['correlation'], seconds=time.monotonic()-begin)), flush=True)
    failures = [r for r in rows if r['effect'] in cfg['powered_effects'] and r['finite']['upper'] < cfg['power']]
    with (OUT/'SIMULATIONS.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    for name in ('config', 'source', 'bank', 'panel'):
        verify(binding[name])
    write(OUT/'COMPLETE.json', dict(passed=True, gate2_passed=False,
          decision='reject_insufficient_sensitivity' if failures else 'unresolved_requires_full_rule',
          rejecting_rows=len(failures), rows=rows, simulations=record(OUT/'SIMULATIONS.npz'),
          binding=record(OUT/'BINDING.json'), comparisons=comparisons,
          complete_rule_simulated=False, posterior_qualified=False,
          null_rates_are_upper_bounds_only=True, job=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-begin))
    print(json.dumps(dict(stage='complete', rejecting_rows=len(failures), seconds=time.monotonic()-begin)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else run(args.workers)
