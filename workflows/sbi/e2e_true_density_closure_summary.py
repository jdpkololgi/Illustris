"""Summarise the small closure receipts, without accessing new truth phases."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1/verification/true_density_closure_20261005_v1')


def summary(record):
    rows=record['cases']
    z=np.asarray([r['operator_gap_sd'] for r in rows])
    full=np.asarray([r['original_bias_sd'] for r in rows])
    matched=np.asarray([r['matched_bias_sd'] for r in rows])
    np.testing.assert_allclose(full-z,matched,atol=1e-12,rtol=0)
    better=sum(r['matched_mahalanobis']<r['original_mahalanobis'] for r in rows)
    return dict(phase=record['phase'],pairs=len(rows),components=z.size,
        fullbox_joint90=float(np.mean([r['fullbox_joint']['covered']['0.9'] for r in rows])),
        matched_joint90=float(np.mean([r['matched_joint']['covered']['0.9'] for r in rows])),
        fullbox_marginal90=float(np.mean([r['fullbox_marginal90'] for r in rows])),
        matched_marginal90=float(np.mean([r['matched_marginal90'] for r in rows])),
        operator_gap_sd_rms=float(np.sqrt(np.mean(z*z))),
        operator_gap_sd_median_abs=float(np.median(abs(z))),
        operator_gap_sd_max_abs=float(abs(z).max()),
        operator_gap_physical_mean_by_eigenvalue=np.asarray([r['operator_gap'] for r in rows]).reshape(-1,3).mean(0).tolist(),
        original_bias_sd_rms=float(np.sqrt(np.mean(full*full))),
        matched_bias_sd_rms=float(np.sqrt(np.mean(matched*matched))),
        original_mahalanobis_squared_mean=float(np.mean([r['original_mahalanobis']**2 for r in rows])),
        matched_mahalanobis_squared_mean=float(np.mean([r['matched_mahalanobis']**2 for r in rows])),
        residual_improves_pairs=better,
        original_failed_now_pass=sum(not r['fullbox_joint']['covered']['0.9'] and r['matched_joint']['covered']['0.9'] for r in rows),
        original_pass_now_failed=sum(r['fullbox_joint']['covered']['0.9'] and not r['matched_joint']['covered']['0.9'] for r in rows),
        max_trace_error=max(r['trace_maxabs'] for r in rows))


if __name__=='__main__':
    records=[json.loads((ROOT/phase/'CLOSURE.json').read_text()) for phase in ('ph012','ph013')]
    assert all(r['passed'] and len(r['cases'])==16 for r in records)
    result=dict(phases=[summary(r) for r in records],independent_phases=2,
                scope='descriptive development closure; no population confidence interval',
                attainable_joint90=29/33,posterior_qualified=False)
    print(json.dumps(result,indent=2,allow_nan=False))
