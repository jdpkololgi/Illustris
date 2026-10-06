"""Reviewed scalar summaries of saved-shape and common-noise interventions."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_conditioning_ablation as a
import e2e_conditioning_intervention as intervention
import numpy as np

b = a.b


def run():
    b.c.require_compute()
    saved_path, run_path = a.OUT/'COMPLETE.json', intervention.OUT/'COMPLETE.json'
    saved, current = (json.loads(p.read_text()) for p in (saved_path, run_path))
    assert saved['passed'] and len(saved['cases']) == 32
    assert current['passed'] and len(current['cases']) == 4 and current['draws'] == 32
    assert saved['source_sha256'] == b.c.sha256(a.__file__)
    assert current['source_sha256'] == b.c.sha256(intervention.__file__)
    lookup = {r['pair_id']: r for r in saved['cases']}
    allrows = []
    for row in current['cases']:
        prior = lookup[row['pair_id']]
        np.testing.assert_allclose(row['truth']['shear_rms'], prior['truth']['shear_rms'], rtol=1e-12)
        arms = {}
        for label, shape, native in [('oracle', 'oracleshape', 'truemass'),
                                      ('sampled', 'sampledshape', 'sampledmass')]:
            values = []
            for i in range(32):
                values.append(dict(cross={mass: prior['cross'][shape+'_'+mass][i] for mass in ('truemass', 'sampledmass')},
                    native=prior['cross'][shape+'_'+native][i],
                    log_distance=0. if label == 'oracle' else prior['native']['canvas_coarse_local_fine']['wide_log_distance'][i]))
            arms[label] = values
        for name, vals in row['arms'].items():
            arms[name] = [dict(cross=v['cross'], native=v['cross']['native'], log_distance=v['log_distance']) for v in vals]
        distance = np.array([v['log_distance'] for v in arms['sampled']])
        for name, factor in [('halfway', .5), ('reflected', 1.), ('randomized', 1.)]:
            measured = np.array([v['log_distance'] for v in arms[name]])
            np.testing.assert_allclose(measured, distance*factor, rtol=1e-6, atol=1e-8)
        allrows.append(dict(phase=row['phase'], pair_id=row['pair_id'], truth=row['truth'], arms=arms))
    tables = []
    for phase in ('ph012', 'ph013', 'pooled'):
        rows = [r for r in allrows if phase == 'pooled' or r['phase'] == phase]
        for arm in ('oracle', 'halfway', 'sampled', 'reflected', 'randomized'):
            values = [v for r in rows for v in r['arms'][arm]]
            entry = dict(phase=phase, arm=arm, pairs=len(rows),
                         mean_wide_log_distance=float(np.mean([v['log_distance'] for v in values])), decoder={})
            for mass in ('truemass', 'sampledmass', 'native'):
                metrics = [v['native'] if mass == 'native' else v['cross'][mass] for v in values]
                ratios = {}
                for metric in ('shear_rms', 'variance', 'power'):
                    prediction = np.mean([v[metric] for v in metrics], axis=0)
                    truth = np.mean([r['truth'][metric] for r in rows], axis=0)
                    ratios[metric] = np.asarray(prediction/truth).tolist()
                entry['decoder'][mass] = ratios
            response = []
            for r in rows:
                baseline = r['arms']['oracle']
                for v, ref in zip(r['arms'][arm], baseline):
                    response.append(np.log(v['cross']['truemass']['shear_rms']/ref['cross']['truemass']['shear_rms']))
            entry['paired_fixed_true_mass_log_shear_change'] = dict(mean=float(np.mean(response)),
                median=float(np.median(response)), fraction_positive=float(np.mean(np.array(response) > 0)),
                caveat='Paired common fine noise; dependent windows/draws, no independent-trial inference')
            tables.append(entry)
    result = dict(passed=True, posterior_qualified=False, tables=tables,
        max_mass_error=max(r['max_mass_error'] for r in current['cases']),
        max_endpoint_relative_replay_error=max(v for r in current['cases'] for v in r['endpoint_errors'].values()),
        sources=[b.c.file_record(p, content_hash=True) for p in (saved_path, run_path)],
        source_sha256=b.c.sha256(__file__),
        scope='Four original geometry-selected pairs, two phases; oracle interventions cannot qualify a posterior or establish training benefit.')
    b.c.atomic_json(intervention.OUT/'SUMMARY.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
