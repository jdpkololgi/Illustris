"""Necessary-event power bound for Gate 2; this module can reject, never pass.

The full advancement event is contained in `optimistic_event`. Its probability
therefore cannot exceed the latter's under each explicitly defined synthetic law.
This permits a scientifically valid early stop without pretending that a partial
score audit establishes the entire gate. All covariance fitting here is fixture
construction, not inference normalization or correction of model predictions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from workflows.sbi import e2e_wholefield_sensitivity as s

CONFIG = s.REPO/'configs/e2e_wholefield_gate2_screen_v1.json'


def ratio_from_draws(x, y):
    """Batched trace ratio; orthogonal changes of summary basis leave it fixed."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 3 or y.shape != (len(x), x.shape[-1]) or x.shape[1] < 3:
        raise ValueError('Expected batch/draw/dimension and batch/dimension')
    v = x.var(1, ddof=1).sum(-1)
    if not np.all(v > 0):
        raise ValueError('Positive predictive trace required')
    return np.sum((x.mean(1)-y)**2, -1)/v - 1/x.shape[1]


def relative_gain(base, candidate):
    a, b = np.asarray(base), np.asarray(candidate)
    return (abs(a-1)-abs(b-1))/np.maximum(abs(a-1), 1e-15)


def optimistic_event(base, candidates, margin=.2):
    """base=(trial,phase); candidates=(configuration,checkpoint,trial,phase)."""
    base, candidates = np.asarray(base), np.asarray(candidates)
    if base.ndim != 2 or candidates.ndim != 4 or candidates.shape[2:] != base.shape:
        raise ValueError('Wrong phase/candidate shape')
    eligible = np.all(relative_gain(base[None, None], candidates) >= margin, -1)
    return eligible.any((0, 1))


def probability_bound(events, comparisons, error=.01):
    values = np.asarray(events, bool)
    if values.ndim != 1 or not len(values) or comparisons < 1 or not 0 < error < 1:
        raise ValueError('Nonempty trials and a valid simultaneous bound required')
    p = float(values.mean())
    radius = math.sqrt(math.log(comparisons/error)/(2*len(values)))
    return dict(successes=int(values.sum()), trials=len(values), point=p,
                lower=max(0., p-radius), upper=min(1., p+radius),
                method='one-sided Hoeffding bounds with union bound over all reported probabilities',
                radius=radius, comparisons=comparisons, familywise_mc_error=error)


def phase_fixture(x, truth, shrinkage=.1, common_fraction=0.):
    """Retain whole-phase dependence; never bootstrap regions independently."""
    x, truth = np.asarray(x), np.asarray(truth)
    if x.ndim != 3 or truth.shape != (len(x), x.shape[-1]):
        raise ValueError('Case/draw/coordinate bank required')
    flat = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    covariance = np.cov(flat, rowvar=False)
    covariance = (1-shrinkage)*covariance + shrinkage*np.diag(np.diag(covariance))
    trace = float(np.trace(covariance))
    if trace <= 0:
        raise ValueError('Positive covariance trace required')
    direction = (flat.mean(0)-truth.ravel())
    if np.linalg.norm(direction) <= 1e-12:
        direction = np.ones(len(direction))
    direction = direction/np.linalg.norm(direction)
    covariance = (1-common_fraction)*covariance + common_fraction*trace*np.outer(direction, direction)
    eigen, vectors = np.linalg.eigh(covariance/trace)
    if eigen.min() < -1e-12:
        raise ValueError('Covariance not positive semidefinite')
    eigen = np.maximum(eigen, 0)
    if not np.isclose(eigen.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError('Fixture normalization drift')
    bias_direction = direction @ vectors
    ratio = s.moments(x, truth)['ratio']
    if ratio is None or ratio <= 0:
        raise ValueError('Positive baseline ratio needed for this declared surrogate')
    return dict(eigen=eigen, direction=bias_direction, ratio=ratio,
                dimensions=len(eigen), effective_rank=float(1/np.sum(eigen**2)),
                max_correlation=float(np.max(abs(covariance/np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))-np.eye(len(eigen))))))


def sampled_moments(noise, eigen, heavy=None):
    sample = noise*np.sqrt(eigen)[None, None]
    if heavy is not None:
        sample = sample*heavy[..., None]
    return sample.mean(1), sample.var(1, ddof=1).sum(-1)


def simulate(task):
    """Synthetic repeated-panel experiment in an orthogonal covariance basis."""
    family, correlation, fixtures, cfg, seed = task
    rng = np.random.default_rng(seed)
    n = cfg['replicates']; m = 32; effects = np.array(cfg['effects'])
    base = np.empty((n, 2)); candidates = np.empty((len(effects), 2, 4, n, 2))
    oracle_base = np.empty((n, 2)); oracle_candidate = np.empty((len(effects), n, 2))
    for start in range(0, n, cfg['batch_size']):
        size = min(cfg['batch_size'], n-start); sl = slice(start, start+size)
        for phase, fixture in enumerate(fixtures):
            lam = fixture['eigen']; d = len(lam); r0 = fixture['ratio']
            bias = np.zeros(d); truth_scale = r0
            if family == 'biased':
                truth_scale = min(r0, 1.)*.5
                bias = fixture['direction']*np.sqrt(r0-truth_scale)
            truth = rng.normal(size=(size, d))*np.sqrt(lam*truth_scale)
            if family == 'heavy_tail':
                truth *= np.sqrt(3/rng.chisquare(5, size=size))[:, None]
            truth += bias
            eps = rng.normal(size=(size, m, d))
            heavy = np.sqrt(3/rng.chisquare(5, size=(size, m))) if family == 'heavy_tail' else None
            mu_a, var_a = sampled_moments(eps, lam, heavy)
            base[sl, phase] = np.sum((mu_a-truth)**2, -1)/var_a - 1/m
            oracle_base[sl, phase] = np.sum(truth**2, -1)
            for ci in range(2):
                shared = rng.normal(size=(size, m, d))
                for checkpoint in range(4):
                    independent = rng.normal(size=(size, m, d))
                    eps_b = correlation*eps + np.sqrt(1-correlation**2)*(np.sqrt(.8)*shared + np.sqrt(.2)*independent)
                    # Identical radial factors are favorable pairing for heavy tails;
                    # truth and posterior radial factors remain independent.
                    mu_b, var_b = sampled_moments(eps_b, lam, heavy)
                    for k, effect in enumerate(effects):
                        r1 = 1+(1-effect)*(r0-1)
                        mean_shift = np.zeros(d); variance_scale = r0/r1
                        if family == 'biased' and r0 > 1:
                            mean_shift = (1-np.sqrt((r1-truth_scale)/(r0-truth_scale)))*bias
                            variance_scale = 1.
                        value = np.sum((mean_shift+np.sqrt(variance_scale)*mu_b-truth)**2, -1)/(variance_scale*var_b) - 1/m
                        candidates[k, ci, checkpoint, sl, phase] = value
                        if ci == 0 and checkpoint == 0:
                            oracle_candidate[k, sl, phase] = np.sum((mean_shift-truth)**2, -1)/variance_scale
    return dict(family=family, correlation=correlation, base=base,
                candidates=candidates, oracle_base=oracle_base, oracle_candidate=oracle_candidate)


def conditional_mc(x, truth, replicates, seed):
    """Whole-draw address resampling; spatial cases are never resampled."""
    rng = np.random.default_rng(seed)
    m = x.shape[1]
    index = rng.integers(m, size=(replicates, m))
    values = np.empty(replicates)
    flat = x.transpose(1, 0, 2).reshape(m, -1)
    for first in range(0, replicates, 64):
        cut = slice(first, first+64)
        draws = flat[index[cut]]
        values[cut] = ratio_from_draws(draws, np.broadcast_to(truth.ravel(), (len(draws), truth.size)))
    return values


def run(workers):
    s.guard(); started = time.monotonic(); cfg = s.read(CONFIG); parent_cfg = s.read(s.CONFIG)
    if not 1 <= workers <= cfg['maximum_parallel_cpu_workers']:
        raise ValueError('Worker cap exceeded')
    prepared = s.read(s.OUT/'PREPARED.json'); s.verify(prepared['arrays'])
    if not prepared['passed'] or prepared['config'] != s.record(s.CONFIG):
        raise ValueError('Prepared panel/config mismatch')
    dest = s.OUT/'necessary_screen_v1'; dest.mkdir(exist_ok=False)
    prov = dict(**s.provenance(), screen_source=s.record(__file__), screen_config=s.record(CONFIG),
                prepared=s.record(s.OUT/'PREPARED.json'))
    with np.load(prepared['arrays']['path'], allow_pickle=False) as f:
        x, y = f['draws'], f['truth']
        if list(f['pair_id']) != [r['pair_id'] for r in prepared['cases']]:
            raise ValueError('Pair ID binding mismatch')
    fixtures = {}; reports = []; selections = []
    for phase in s.PHASES:
        ix = [i for i, row in enumerate(prepared['cases']) if row['phase'] == phase]
        if len(ix) != 16:
            raise ValueError('Full 16-pair phase required')
        selections.append(ix)
        q = conditional_mc(x[ix], y[ix], cfg['replicates'], cfg['seed'])
        lo, hi = np.quantile(q, cfg['fixed_panel_mc_quantiles'])
        reports.append(dict(phase=phase, point=s.moments(x[ix], y[ix])['ratio'],
                            conditional_mc_interval=[float(lo), float(hi)],
                            includes_one=bool(lo <= 1 <= hi),
                            caveat='whole-draw MC range at fixed phase truth, not population uncertainty'))
    eligible_phases = [i for i, q in enumerate(reports) if not q['includes_one']]
    if not eligible_phases:
        s.write(dest/'INCONCLUSIVE.json', dict(**prov, ratio_screen_applicable=False,
                reason='Baseline ratio indistinguishable from one: register joint-dependence alternative before running the screen',
                baseline=reports, gate2_passed=False))
        print(json.dumps(dict(stage='inconclusive', baseline=reports)), flush=True)
        return
    tasks = []
    for family in cfg['families']:
        ff = [phase_fixture(x[ix], y[ix], cfg['covariance_shrinkage'],
                            cfg['common_mode_trace_fraction'] if family == 'common_mode' else 0.) for ix in selections]
        fixtures[family] = [{k:(v.tolist() if isinstance(v, np.ndarray) else v) for k,v in q.items()} for q in ff]
        for corr in cfg['paired_noise_correlations']:
            tasks.append((family, corr, ff, cfg, cfg['seed']+len(tasks)*100003))
    # Bind all nuisance parameters, power rules and sources before synthetic output.
    s.write(dest/'BINDING.json', dict(**prov, baseline=reports, fixtures=fixtures, screen_settings=cfg,
                                    gate_config=parent_cfg, workers=workers, eligible_ratio_phase_indices=eligible_phases))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(simulate, tasks):
            results.append(result)
            print(json.dumps(dict(stage='simulated', family=result['family'], paired_noise_correlation=result['correlation'],
                                  seconds=time.monotonic()-started)), flush=True)
    comparisons = len(results)*len(cfg['effects'])*2
    rows = []; arrays = {}
    for i, result in enumerate(results):
        arrays[f'{i}_base'] = result['base']; arrays[f'{i}_candidates'] = result['candidates']
        arrays[f'{i}_oracle_base'] = result['oracle_base']; arrays[f'{i}_oracle_candidate'] = result['oracle_candidate']
        for k, effect in enumerate(cfg['effects']):
            event = optimistic_event(result['base'][..., eligible_phases], result['candidates'][k][..., eligible_phases])
            oracle_event = np.all(relative_gain(result['oracle_base'][..., eligible_phases], result['oracle_candidate'][k][..., eligible_phases]) >= .2, -1)
            rows.append(dict(family=result['family'], paired_noise_correlation=result['correlation'], effect=effect,
                             finite_draw=probability_bound(event, comparisons, cfg['familywise_mc_error']),
                             exact_moments=probability_bound(oracle_event, comparisons, cfg['familywise_mc_error'])))
    failures = [q for q in rows if q['effect'] == .2 and q['finite_draw']['upper'] < .8]
    with (dest/'SIMULATIONS.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    if s.record(__file__) != prov['screen_source'] or s.record(CONFIG) != prov['screen_config']:
        raise ValueError('Screen source/config changed while running')
    outcome = dict(**prov, passed=True, gate2_passed=False,
                   gate2_decision='reject_insufficient_sensitivity' if failures else 'inconclusive_requires_complete_rule',
                   rejection_basis='Full-rule success implies relaxed ratio-event success; simultaneous upper probability bound < 0.8' if failures else None,
                   rows=rows, rejecting_rows=len(failures), baseline=reports, eligible_ratio_phase_indices=eligible_phases,
                   arrays=s.record(dest/'SIMULATIONS.npz'), binding=s.record(dest/'BINDING.json'),
                   complete_rule_simulated=False,
                   excluded_requirements=['phases whose baseline ratio MC interval includes one', 'second training seed', 'true-coarse evaluation', 'adjacent-checkpoint stability',
                       'actual checkpoint selector', 'joint proper-score improvement', 'coverage and marginal guardrails', 'MC significance'],
                   exclusions_make_event_more_permissive=True,
                   qualification='No architecture failure or empirical population-power inference; necessary condition fails under specified synthetic laws',
                   seconds=time.monotonic()-started)
    s.write(dest/'COMPLETE.json', outcome)
    print(json.dumps(dict(stage='screen_complete', decision=outcome['gate2_decision'], rejecting_rows=len(failures),
                          registered_effect=[q for q in rows if q['effect'] == .2], seconds=outcome['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    run(parser.parse_args().workers)
