"""Independent reconciliation of the Gate 2 rejection (no new sensitivity fit)."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket

import numpy as np
from scipy.stats import beta

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
OUT = ROOT/'wholefield_comparison_20261009_v1/gate2'
SCREEN = OUT/'necessary_screen_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run():
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('Compute node required')
    prepared = read(OUT/'PREPARED.json'); report = read(SCREEN/'COMPLETE.json')
    binding = read(SCREEN/'BINDING.json'); cfg = binding['screen_settings']
    checks = 0
    def equal(a, b, **kwargs):
        nonlocal checks
        np.testing.assert_allclose(a, b, rtol=0, atol=kwargs.get('atol', 1e-12)); checks += 1
    for rec in prepared['inputs'] + [prepared['arrays'], report['arrays'], report['binding'], report['prepared'],
                                      report['source'], report['config'], report['screen_source'], report['screen_config']]:
        if sha(rec['path']) != rec['sha256']:
            raise ValueError('Hash drift: '+rec['path'])
        checks += 1
    with np.load(prepared['arrays']['path'], allow_pickle=False) as f:
        x, y = f['draws'], f['truth']
        ids = list(f['pair_id'])
    assert ids == [row['pair_id'] for row in prepared['cases']]
    checks += 1
    # Rebuild all source summaries with a boolean geometry mask and contrast
    # matrix, independently of the extraction helper/slicing implementation.
    coords = np.indices((16, 16, 16)).reshape(3, -1)
    mask = ((coords >= 2) & (coords < 14)).all(0)
    contrast = np.array([[-1., 0., 1.], [1., -2., 1.]])/np.sqrt([[2.], [6.]])
    for phase in ('ph012', 'ph013'):
        for cap in ('NGC', 'SGC'):
            ix = [i for i, q in enumerate(prepared['cases']) if q['phase'] == phase and q['cap'] == cap]
            assert len(ix) == 8
            for first in range(0, 32, 4):
                path = ROOT/'cfm_conditioning_augmentation_20261006_v3/cores'/phase/cap/'parent/s0'/f'd{first:03d}/COMPLETE.json'
                rec = read(path)
                with np.load(rec['output']['path'], allow_pickle=False) as f:
                    ev = f['eigen'][1].sum(-2)/4096
                    density = f['density'][..., mask].sum(-1)/int(mask.sum())
                    tidal = np.einsum('...j,kj->...k', ev, contrast)
                    rebuilt = np.concatenate((density[..., None], tidal), -1).reshape(8, 4, 6)
                    equal(rebuilt, x[ix, first:first+4])
                    truth_ev = f['truth_eigen'].sum(-2)/4096
                    truth_density = f['truth_density'][..., mask].sum(-1)/int(mask.sum())
                    rebuilt_truth = np.concatenate((truth_density[..., None], np.einsum('...j,kj->...k', truth_ev, contrast)), -1).reshape(8, 6)
                    equal(rebuilt_truth, y[ix])
                    assert list(f['pair_id']) == [ids[i] for i in ix]
                    checks += 1
    # Moment ratios, conditional-MC phase eligibility and fixture spectra.
    indices = np.random.default_rng(cfg['seed']).integers(32, size=(cfg['replicates'], 32))
    counts = np.array([np.bincount(v, minlength=32) for v in indices], dtype=float)
    eligible = []
    for p, phase in enumerate(('ph012', 'ph013')):
        ix = [i for i, row in enumerate(prepared['cases']) if row['phase'] == phase]
        flat = x[ix].transpose(1, 0, 2).reshape(32, -1); target = y[ix].ravel()
        residual = flat.mean(0)-target
        var = sum(np.dot(row-flat.mean(0), row-flat.mean(0)) for row in flat)/31
        value = np.dot(residual, residual)/var-1/32
        equal(value, report['baseline'][p]['point'])
        means = counts@flat/32
        raw_second = counts@(flat**2).sum(-1)
        variances = (raw_second-32*(means**2).sum(-1))/31
        ratios = ((means-target)**2).sum(-1)/variances-1/32
        interval = np.quantile(ratios, cfg['fixed_panel_mc_quantiles'])
        equal(interval, report['baseline'][p]['conditional_mc_interval'], atol=1e-10)
        if not interval[0] <= 1 <= interval[1]:
            eligible.append(p)
        for family in cfg['families']:
            fixture = binding['fixtures'][family][p]
            equal(sum(fixture['eigen']), 1.)
            equal(np.dot(fixture['direction'], fixture['direction']), 1.)
            equal(fixture['ratio'], value)
            for effect in cfg['effects']:
                target_r = 1+(1-effect)*(value-1)
                if family == 'biased' and value > 1:
                    truth_variance = .5
                    remaining_squared_bias = target_r-truth_variance
                    expected = truth_variance+remaining_squared_bias
                else:
                    posterior_variance = value/target_r
                    expected = value/posterior_variance
                equal(expected, target_r)
                equal(abs(target_r-1), (1-effect)*abs(value-1))
    assert eligible == report['eligible_ratio_phase_indices'] == binding['eligible_ratio_phase_indices']
    checks += 1
    # Recount every event using direct inequalities, not the screen's helpers.
    cp_rows = []; rejecting = 0
    with np.load(report['arrays']['path'], allow_pickle=False) as f:
        for i, (family, corr) in enumerate((fam, cor) for fam in cfg['families'] for cor in cfg['paired_noise_correlations']):
            base = f[f'{i}_base']; candidates = f[f'{i}_candidates']
            ob = f[f'{i}_oracle_base']; oc = f[f'{i}_oracle_candidate']
            assert base.shape == (4096, 2) and candidates.shape == (5, 2, 4, 4096, 2)
            checks += 1
            for k, effect in enumerate(cfg['effects']):
                row = next(q for q in report['rows'] if q['family'] == family and q['paired_noise_correlation'] == corr and q['effect'] == effect)
                success = np.zeros(4096, dtype=bool)
                for configuration in range(2):
                    for checkpoint in range(4):
                        candidate = candidates[k, configuration, checkpoint][:, eligible]
                        # The independently written condition is equivalent to
                        # >=20% reduction; tiny exact-boundary roundoff is checked.
                        ok = (abs(candidate-1) <= .8*abs(base[:, eligible]-1)).all(1)
                        success |= ok
                oracle = (abs(oc[k][:, eligible]-1) <= .8*abs(ob[:, eligible]-1)).all(1)
                for name, event in (('finite_draw', success), ('exact_moments', oracle)):
                    count = int(event.sum()); n = len(event); stat = row[name]
                    if count != stat['successes']:
                        raise ValueError(('Independent event-count mismatch', family, corr, effect, name, count, stat['successes']))
                    equal(count/n, stat['point'])
                    delta = math.sqrt(math.log(120/.01)/(2*n))
                    equal(stat['upper'], min(1., count/n+delta))
                    equal(stat['lower'], max(0., count/n-delta))
                    if name == 'finite_draw' and effect == .2:
                        cp = float(beta.ppf(1-.01/120, count+1, n-count)) if count < n else 1.
                        cp_rows.append(dict(family=family, correlation=corr, exact_binomial_upper=cp,
                                            registered_hoeffding_upper=stat['upper']))
                        if stat['upper'] < .8:
                            rejecting += 1
                            assert cp < .8
                            checks += 1
                for p in range(2):
                    if family != 'biased' or binding['fixtures'][family][p]['ratio'] < 1:
                        r0 = binding['fixtures'][family][p]['ratio']; r1 = 1+(1-effect)*(r0-1)
                        equal(oc[k, :, p], ob[:, p]*r1/r0)
                if effect == 0 and corr == 1:
                    equal(candidates[k], np.broadcast_to(base, candidates[k].shape))
    assert len(report['rows']) == 60 and rejecting == report['rejecting_rows'] == 12
    assert report['gate2_decision'] == 'reject_insufficient_sensitivity' and not report['gate2_passed']
    assert not report['complete_rule_simulated'] and report['exclusions_make_event_more_permissive']
    checks += 3
    result = dict(passed=True, independent_checks=checks, source_sha256=sha(__file__),
                  screen_complete_sha256=sha(SCREEN/'COMPLETE.json'), prepared_sha256=sha(OUT/'PREPARED.json'),
                  registered_rejection_reproduced=True, alternative_exact_binomial_upper_bounds=cp_rows,
                  full_gate_pass_claim=False, engineering_and_rejection_verification_only=True,
                  bound_scope='99% simultaneous UPPER bounds; lower bounds are descriptive and not used for the decision',
                  job=os.environ['SLURM_JOB_ID'])
    with (SCREEN/'VERIFIED.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
