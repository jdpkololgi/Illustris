"""Bounded JSON-only closeout checks; no field computation, targets, or fitting."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def main():
    started = time.time()
    check(digest(ROOT/'MANIFEST.json') == '67e1f9d0edc73b58c761e35f1f2dc13d4fe102e27f260104ecdb87b725a9bf10', 'manifest')
    manifest = read(ROOT/'MANIFEST.json')
    complete = read(ROOT/'EXPERIMENT_COMPLETE.json')
    result = read(ROOT/'analysis/RESULTS.json')
    ledger = read(ROOT/'DRAW_LEDGER.json')
    figures = read(ROOT/'analysis/FIGURES.json')
    draw_audit = read(ROOT/'analysis/DRAW_INTEGRITY.json')
    weight_audit = read(ROOT/'analysis/CHECKPOINT_INTEGRITY.json')
    check(complete['complete'] and not complete['production_ready'], 'completion flags')
    check(draw_audit['passed'] and weight_audit['passed'], 'payload audits')
    check(complete['results_sha256'] == figures['inputs_sha256'] == digest(ROOT/'analysis/RESULTS.json'), 'results binding')
    for row in (complete, result, figures, draw_audit, weight_audit):
        check(row['manifest_sha256'] == digest(ROOT/'MANIFEST.json'), 'manifest binding')
    for name, expected in figures['files'].items():
        check(digest(ROOT/'analysis'/name) == expected, 'figure/report hash')
    for audit in (draw_audit, weight_audit):
        check(audit['models_frozen_sha256'] == digest(ROOT/'MODELS_FROZEN.json'), 'frozen weights binding')
    check((result['central_draws'],result['coarse_draws']) == (33280,7872), 'result draw counts')
    check((draw_audit['fine_draws'],draw_audit['coarse_draws']) == (33280,7872), 'actual draw counts')
    expected_ids = {t['task_id'] for t in ledger['tasks']}
    check(len(expected_ids) == 688 and set(result['case_receipts']) == expected_ids, 'case coverage')
    check({p.stem for p in (ROOT/'analysis/cases').glob('*.json')} == expected_ids, 'extra or missing cases')
    cases = {}
    receipts = {}
    for task in ledger['tasks']:
        name = task['task_id']
        path = ROOT/'analysis/cases'/(name+'.json')
        check(digest(path) == result['case_receipts'][name], 'case hash')
        row = read(path)
        check(row['task'] == task and row['manifest_sha256'] == result['manifest_sha256'], 'case identity')
        receipts.update(row['draw_receipts'])
        cases[name] = row
    check(receipts == draw_audit['fine_receipts'], 'reported inputs differ from audited inventory')
    preview_path = ROOT/'analysis/density_preview/PRELIMINARY.json'
    check(digest(preview_path) == 'ed3e401de1b236e6f946687d69bb491bc8a3f45d61549278fa802c472acd97c3', 'preview drift')
    preview = read(preview_path)
    check(len(preview['case_receipts']) == 384, 'preview panel count')
    for name, expected in preview['case_receipts'].items():
        path = ROOT/'analysis/density_preview/cases'/(name+'.json')
        check(digest(path) == expected, 'preview case hash')
        old, new = read(path), cases[name]
        check(old['task'] == new['task'] and old['draw_receipts'] == new['draw_receipts'], 'preview input parity')
        for key in old['spectra']:
            equal(old['spectra'][key], new['spectra'][key])
        c = new['closures']['periodic']['physical']['all']
        for old_key,new_key in [('density_crps','standardized_crps'),('density_rmse','rmse_mean'),('density_bias','bias'),('rms_spread','rms_spread')]:
            equal(old[old_key], c[new_key][0])
        for level in old['coverage']:
            equal(old['coverage'][level], c['coverage'][level][0])
            equal(old['width'][level], c['width'][level][0])
            equal(old['attainable'][level], c['attainable_coverage'][level])
    for group in result['progression']:
        rows = [r for r in cases.values() if r['task']['purpose']=='main' and r['task']['steps']==250
                and all(r['task'][k] == group[k] for k in ('arm','replica','checkpoint')) and r['phase']==group['phase']]
        check(len(rows) == 16, 'progression panel')
        for key in ('density_crps','tidal_energy','density_coverage90','tidal_coverage90','tidal_coverage90_components','attainable90'):
            equal(group[key], np.mean([r['primary'][key] for r in rows], axis=0))
        equal(group['density_rmse'], np.sqrt(np.mean([r['primary']['density_rmse']**2 for r in rows])))
        ratio = np.mean([r['spectra']['mean_sample_power'] for r in rows],axis=0)/np.mean([r['spectra']['truth_power'] for r in rows],axis=0)
        equal(group['aggregate_sample_power_ratio'], ratio)
        equal(group['power_discrepancy'], np.abs(np.log(ratio)).mean())
    check(len(result['progression']) == 32, 'progression count')
    totals = complete['resource_accounting']
    check(totals['gpu_hours'] <= manifest['spec']['budget']['gpu_hours'], 'GPU cap')
    check(totals['cpu_node_hours'] <= manifest['spec']['budget']['cpu_node_hours'], 'CPU cap')
    output = dict(passed=True, started_unix=started, ended_unix=time.time(),
        auditor_sha256=digest(__file__), case_count=688, preview_cases_reconciled=384,
        progression_cells_recomputed=32, manifest_sha256=digest(ROOT/'MANIFEST.json'),
        results_sha256=digest(ROOT/'analysis/RESULTS.json'), figures_sha256=digest(ROOT/'analysis/FIGURES.json'),
        draw_integrity_sha256=digest(ROOT/'analysis/DRAW_INTEGRITY.json'),
        checkpoint_integrity_sha256=digest(ROOT/'analysis/CHECKPOINT_INTEGRITY.json'),
        experiment_complete_sha256=digest(ROOT/'EXPERIMENT_COMPLETE.json'),
        scope='JSON/receipt/count/aggregation/preview parity and recorded compute caps; scientific interpretation and scheduler review remain separate')
    with (ROOT/'analysis/CLOSEOUT_AUDIT.json').open('x') as stream:
        json.dump(output, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
