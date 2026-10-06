"""Frozen pointwise conditional-Monte-Carlo oracle screen, not population inference."""
import numpy as np


def classify(point, lower, upper, lo, hi):
    """All quantities are scalar: fail on point breach, inconclusive on MC bounds."""
    if not np.isfinite([point, lower, upper, lo, hi]).all() or not lower <= upper:
        raise ValueError('finite ordered gate inputs required')
    if point < lo or point > hi:
        return 'fail'
    if lower < lo or upper > hi:
        return 'inconclusive'
    return 'pass'


def oracle_screen(power, width, truth_power, truth_width, references,
                  replicates=4096, seed=6100617):
    """M x phases x bands power; M x phases width, paired reference banks.

    Spatial reductions are performed before this call with the frozen region
    weights. Resample whole draw addresses, identically across quantities,
    regions, phases and arms. This estimates conditional MC variability only.
    Bounds are pointwise one-sided 95%; no familywise post-selection claim.
    """
    power, width, tp, tw = map(np.asarray, (power, width, truth_power, truth_width))
    if (power.shape != (32, 2, 2) or width.shape != (32, 2)
            or tp.shape != (2, 2) or tw.shape != (2,)
            or set(references) != {'parent', 'control'}
            or replicates != 4096 or seed != 6100617):
        raise ValueError('registered two-phase, two-band, 32-draw oracle panel required')
    arrays = [power, width, tp, tw]+[np.asarray(v) for v in references.values()]
    if (any(not np.isfinite(a).all() or np.any(a <= 0) for a in arrays)
            or any(np.shape(v) != width.shape for v in references.values())):
        raise ValueError('finite positive aligned oracle banks required')
    rng = np.random.default_rng(seed)
    index = rng.integers(0, 32, size=(replicates, 32))
    pr = power.mean(0)/tp
    pb = power[index].mean(1)/tp
    deficit = 100*(1-width.mean(0)/tw)
    db = 100*(1-width[index].mean(1)/tw)
    rows = []
    def record(name, point, samples, lo, hi):
        lower, upper = np.quantile(samples, [.05, .95])
        rows.append(dict(name=name, point=float(point), lower=float(lower), upper=float(upper),
                         limits=[lo, hi], decision=classify(point, lower, upper, lo, hi)))
    for phase in range(2):
        for band in range(2):
            record(f'phase{phase}/power{band}', pr[phase, band], pb[:, phase, band], .95, 1.05)
        record(f'phase{phase}/signed_deficit_pp', deficit[phase], db[:, phase], -.5, .5)
        for name, bank in references.items():
            bank = np.asarray(bank)
            rd = 100*(1-bank.mean(0)/tw)
            rb = 100*(1-bank[index].mean(1)/tw)
            # Lower is better: compare magnitude to avoid rewarding an overshoot.
            point = abs(deficit[phase])-abs(rd[phase])
            samples = abs(db[:, phase])-abs(rb[:, phase])
            # Width deficits bounded below by zero in magnitude: improvement has
            # no rejection floor. Use an explicit finite lower bound from reference.
            upper = float(np.quantile(samples, .95))
            rows.append(dict(name=f'phase{phase}/regression_vs_{name}_pp', point=float(point),
                upper=upper, limit=.2,
                decision='fail' if point > .2 else ('inconclusive' if upper > .2 else 'pass')))
    decisions = {r['decision'] for r in rows}
    decision = 'fail' if 'fail' in decisions else ('inconclusive' if 'inconclusive' in decisions else 'pass')
    return dict(decision=decision, checks=rows, replicates=replicates, seed=seed,
                scope='pointwise conditional MC only; not independent confirmation')
