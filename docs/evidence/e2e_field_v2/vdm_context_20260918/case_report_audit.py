"""Reconcile all frozen case reports and summarize already registered metrics.

No model/data/draw/metric choices are changed. This reads case JSON only, waits
for the complete predeclared panel, and writes separate closeout artifacts.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from collections import defaultdict

import numpy as np

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def calibration_group(rows):
    rows = [r for r in rows if r['probes']]
    if not rows:
        return dict(anchors=0)
    avg = lambda key: np.mean([r[key] for r in rows], axis=0).tolist()
    out = dict(anchors=len(rows), mean_probes=float(np.mean([r['probes'] for r in rows])))
    for key in ('bias', 'crps', 'standardized_crps', 'tidal_joint_energy', 'all_six_energy'):
        out[key] = avg(key)
    for key in ('rmse_mean', 'rms_spread'):
        out[key] = np.sqrt(np.mean(np.array([r[key] for r in rows])**2, axis=0)).tolist()
    for key in ('coverage', 'width', 'attainable_coverage'):
        out[key] = {level: np.mean([r[key][level] for r in rows], axis=0).tolist()
                    for level in rows[0][key]}
    out['rank_frequencies'] = np.mean([np.array(r['rank_histogram'])/r['probes'] for r in rows], axis=0).tolist()
    return out


def spectra_group(rows):
    avg = lambda key: np.mean([r[key] for r in rows], axis=0)
    truth = avg('truth_power')
    keys = ('mean_sample_power', 'posterior_mean_power', 'posterior_residual_power_population',
            'posterior_mean_power_mc_corrected', 'posterior_residual_power_unbiased')
    out = dict(truth_power=truth.tolist())
    for key in keys:
        out[key+'_ratio'] = (avg(key)/truth).tolist()
    for key in ('correlation_posterior_mean', 'mean_sample_correlation'):
        out[key] = avg(key).tolist()
    return out


def summarize(rows):
    out = dict(anchors=len(rows), spectra=spectra_group([r['spectra'] for r in rows]), closures={})
    for closure in rows[0]['closures']:
        out['closures'][closure] = {
            estimand: {mask: calibration_group([r['closures'][closure][estimand][mask] for r in rows])
                       for mask in ('all', 'observed', 'unobserved')}
            for estimand in rows[0]['closures'][closure]}
    for prefix in ('', 'core_'):
        out[prefix+'onepoint'] = dict(
            truth=np.mean([r[prefix+'onepoint_truth'] for r in rows], axis=0).tolist(),
            draws=np.mean([np.mean(r[prefix+'onepoint_draws'], axis=0) for r in rows], axis=0).tolist())
    out['regional'] = dict(
        crps=float(np.mean([r['regional']['mean_crps'] for r in rows])),
        bias=float(np.mean([r['regional']['bias'] for r in rows])),
        rmse=float(np.sqrt(np.mean(np.array([r['regional']['bias'] for r in rows])**2))),
        rms_spread=float(np.sqrt(np.mean(np.array([r['regional']['spread'] for r in rows])**2))),
        coverage90=float(np.mean([r['regional']['coverage90'] for r in rows])),
        attainable90=float(np.mean([r['regional']['attainable90'] for r in rows])))
    wide = [r['wide'] for r in rows if r['wide'] is not None]
    out['wide'] = None if not wide else dict(**spectra_group(wide), **{
        k: float(np.mean([r[k] for r in wide])) for k in ('mean_crps', 'coverage90', 'attainable90', 'mean_bias')})
    return out


def main():
    started = time.time()
    assert socket.gethostname().startswith('nid'), 'compute node required'
    allowed_jobs = [read(ROOT/'resources/10_START.json')['job']]
    supplemental = ROOT/'resources/CASE_AUDIT_START.json'
    if supplemental.exists():
        start = read(supplemental)
        assert start['stage'] == 'case_audit' and start['gpus'] == 0
        allowed_jobs.append(start['job'])
    assert os.environ['SLURM_JOB_ID'] in allowed_jobs
    tasks = read(ROOT/'DRAW_LEDGER.json')['tasks']
    assert len(tasks) == 688
    folder = ROOT/'analysis/cases'
    # Bounded deterministic metadata wait; no new resources or scientific decisions.
    while not all((folder/(t['task_id']+'.json')).exists() for t in tasks):
        assert time.time()-started < 90*60, 'case report audit wait exceeded 90 minutes'
        time.sleep(5)
    manifest = digest(ROOT/'MANIFEST.json')
    ledger_audit = read(ROOT/'analysis/DRAW_LEDGER_INTEGRITY.json')
    assert ledger_audit['passed'] and ledger_audit['manifest_sha256'] == manifest
    receipts_by_case = defaultdict(dict)
    for name, sha in ledger_audit['fine_chunk_receipts'].items():
        receipts_by_case[Path(name).parent.name][name] = sha
    preview = read(ROOT/'analysis/density_preview/PRELIMINARY.json')
    records, hashes, reconciled = [], {}, 0
    for task in tasks:
        path = folder/(task['task_id']+'.json')
        r = read(path)
        hashes[task['task_id']] = digest(path)
        assert r['task'] == task and r['manifest_sha256'] == manifest
        assert r['draw_receipts'] == receipts_by_case[task['task_id']]
        for closure, estimands in r['closures'].items():
            for estimand, masks in estimands.items():
                assert masks['all']['probes'] == 4096
                assert masks['observed']['probes']+masks['unobserved']['probes'] == 4096
                for mask, c in masks.items():
                    if not c['probes']:
                        continue
                    assert c['draws'] == task['count']
                    assert np.shape(c['rank_histogram']) == (6, 16)
                    close(np.sum(c['rank_histogram'], axis=1), [c['probes']]*6)
                    for key in ('bias', 'crps', 'standardized_crps', 'rms_spread', 'rmse_mean'):
                        assert np.shape(c[key]) == (6,) and np.isfinite(c[key]).all()
                    for level in ('0.5', '0.68', '0.9', '0.95'):
                        assert ((np.array(c['coverage'][level]) >= 0) & (np.array(c['coverage'][level]) <= 1)).all()
                        assert (np.array(c['width'][level]) >= 0).all()
                for level in ('0.5', '0.68', '0.9', '0.95'):
                    combined = sum(np.array(c['coverage'][level])*c['probes'] for k, c in masks.items()
                                   if k != 'all' and c['probes'])/4096
                    close(combined, masks['all']['coverage'][level])
        s = r['spectra']
        close(np.array(s['mean_sample_power']), np.array(s['posterior_mean_power'])+s['posterior_residual_power_population'])
        close(np.array(s['posterior_mean_power_mc_corrected'])+s['posterior_residual_power_unbiased'], s['mean_sample_power'])
        if task['task_id'] in preview['case_receipts']:
            p = ROOT/'analysis/density_preview/cases'/path.name
            assert digest(p) == preview['case_receipts'][task['task_id']]
            old = read(p)
            assert old['task'] == task and old['draw_receipts'] == r['draw_receipts']
            assert old['spectra'] == r['spectra']
            density = r['closures']['periodic']['physical']['all']
            close([old[k] for k in ('density_crps', 'density_rmse', 'density_bias', 'rms_spread')],
                  [density[k][0] for k in ('standardized_crps', 'rmse_mean', 'bias', 'rms_spread')])
            for level in old['coverage']:
                close([old['coverage'][level], old['width'][level], old['attainable'][level]],
                      [density['coverage'][level][0], density['width'][level][0], density['attainable_coverage'][level]])
            reconciled += 1
        records.append(r)
    assert reconciled == 384
    assert set(hashes) == {p.stem for p in folder.glob('*.json')}
    groups = defaultdict(list)
    for row in records:
        t = row['task']
        if t['steps'] == 250:
            groups[t['arm'], t['replica'], t['checkpoint'], row['phase'], t['purpose'], t['coarse_mode']].append(row)
    summary = [dict(arm=k[0], replica=k[1], checkpoint=k[2], phase=k[3], purpose=k[4], coarse_mode=k[5],
                    **summarize(v)) for k, v in sorted(groups.items(), key=lambda x: str(x[0]))]
    common = dict(job=os.environ['SLURM_JOB_ID'], step=os.environ.get('SLURM_STEP_ID'), node=socket.gethostname(),
                  manifest_sha256=manifest, auditor_sha256=digest(__file__), started_unix=started, ended_unix=time.time())
    publish(ROOT/'analysis/REGISTERED_DIAGNOSTIC_SUMMARY.json', dict(**common, groups=summary,
        feature_order=['delta', 'lambda1', 'lambda2', 'lambda3', 'gap12', 'gap23'],
        onepoint_order=['mean', 'std', 'q001', 'q01', 'q10', 'q50', 'q90', 'q99', 'q999'],
        weighting='Equal anchors within each seed/phase/purpose cell; equal-draw mean within each anchor; RMS aggregation for RMSE/spread',
        scope='Only already registered case metrics; no new prediction, target access, checkpoint selection or gate changes'))
    publish(ROOT/'analysis/CASE_REPORT_AUDIT.json', dict(**common, passed=True, case_receipts=hashes,
        cases=688, preview_cases_reconciled=reconciled, rtol=1e-10, atol=1e-12,
        checks=['exact tasks and fine receipt bindings', 'all 384 preview density cases', 'spectral decomposition and finite-M identities',
                'all physical/matched closures', 'observed/unobserved coverage decomposition', 'six-feature ranks and all four coverage levels'],
        diagnostic_summary_sha256=digest(ROOT/'analysis/REGISTERED_DIAGNOSTIC_SUMMARY.json')))
    print('CASE_REPORT_AUDIT_PASS', len(records), reconciled, flush=True)


if __name__ == '__main__':
    main()
