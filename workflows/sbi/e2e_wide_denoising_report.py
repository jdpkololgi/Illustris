"""Compact descriptive summaries of the registered training-only denoising audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from workflows.sbi import e2e_wide_pipeline as p


def summarize(data):
    if not data['complete']:
        raise ValueError('audit incomplete')
    out = {'registration': data['registration'], 'elapsed_seconds': data['elapsed_seconds'],
           'probe_summary': data['probe_summary'], 'components': {}, 'true_coarse': {},
           'trajectory': {}, 'phase_probes': {}, 'counts': {k: len(data[k]) for k in
           ('probes', 'components', 'true_coarse_diagnostics', 'sampler_trajectories')},
           'oracle_draw_sha256': {f'{r["anchor_id"]}/{r["method"]}': r['sample_sha256']
                                  for r in data['true_coarse_diagnostics']},
           'training_ready': False, 'calibration_pass': None}
    phases = sorted({r['phase'] for r in data['probes']})
    for phase in ['all'] + phases:
        for method in ('cfm', 'diffusion'):
            for step in (192, 384):
                rows = [r for r in data['components'] if r['method'] == method and r['step'] == step
                        and (phase == 'all' or r['phase'] == phase)]
                pred = {k: np.array([r['prediction'][k] for r in rows]) for k in ('coarse', 'fine', 'cross', 'total')}
                truth = np.array([r['truth']['total'] for r in rows])
                result = {'count': len(rows), 'power_ratio': np.median(pred['total']/truth, axis=0).tolist(),
                          'high_k_fraction': float(np.median(pred['total'][:, -1]/pred['total'].sum(axis=1))),
                          'truth_high_k_fraction': float(np.median(truth[:, -1]/truth.sum(axis=1))),
                          'median_power': {k: np.median(v, axis=0).tolist() for k, v in pred.items()},
                          'median_component_fraction': {k: np.median(v/pred['total'], axis=0).tolist() for k, v in pred.items() if k != 'total'}}
                out['components'][f'{phase}/{method}/{step}'] = result
            rows = [r for r in data['true_coarse_diagnostics'] if r['method'] == method
                    and (phase == 'all' or r['phase'] == phase)]
            pairs = [next(c for c in data['components'] if c['anchor_id'] == r['anchor_id'] and
                          c['method'] == method and c['step'] == 384 and c['draw'] == 0) for r in rows]
            oracle = np.array([r['components']['total'] for r in rows])
            actual = np.array([r['prediction']['total'] for r in pairs])
            truth = np.array([r['truth']['total'] for r in pairs])
            out['true_coarse'][f'{phase}/{method}'] = {
                'count': len(rows), 'oracle_power_ratio': np.median(oracle/truth, axis=0).tolist(),
                'actual_draw0_power_ratio': np.median(actual/truth, axis=0).tolist(),
                'oracle_over_actual_power': np.median(oracle/actual, axis=0).tolist(),
                'oracle_high_k_fraction': float(np.median(oracle[:, -1]/oracle.sum(axis=1))),
                'oracle_density_below_minus_one_fraction': float(np.median([r['density_below_minus_one_fraction'] for r in rows]))}
            for stage in ('coarse', 'fine'):
                if phase != 'all':
                    out['phase_probes'][f'{phase}/{method}/{stage}'] = {
                        f'{step}/{ratio}': {k: np.median([r['metrics'][k] for r in data['probes']
                        if r['phase'] == phase and r['method'] == method and r['stage'] == stage
                        and r['step'] == step and r['ratio'] == ratio], axis=0).tolist()
                        for k in ('power_ratio', 'gain', 'error_power', 'error_over_truth_power',
                                  'noise_leakage_over_input_sigma', 'velocity_error_band_fraction')}
                        for step in (192, 384) for ratio in (.05, .2, 1., 5., 20.)}
                else:
                    rows = [r for r in data['sampler_trajectories'] if r['method'] == method and r['stage'] == stage]
                    trajectory = []
                    for call in sorted({r['call'] for r in rows}):
                        chosen = [r for r in rows if r['call'] == call]
                        truths = [next(r['metrics']['truth_power'] for r in data['probes'] if
                                  r['anchor_id'] == s['anchor_id'] and r['method'] == method and
                                  r['stage'] == stage and r['step'] == 384) for s in chosen]
                        ratios = np.array([r['predicted_clean_power'] for r in chosen])/np.array(truths)
                        trajectory.append({'call': call, 'time': chosen[0]['time'],
                                           'clean_power_ratio': np.median(ratios, axis=0).tolist()})
                    out['trajectory'][f'{method}/{stage}'] = trajectory
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.audit.read_text())
    summary = summarize(data)
    summary.update(audit_path=str(args.audit.resolve()), audit_sha256=p.sha256(args.audit),
                   report_source_sha256=p.sha256(__file__))
    p.write_json(args.output, summary)
    print(json.dumps({k: summary[k] for k in ('counts', 'components', 'true_coarse', 'trajectory')}, indent=2))


if __name__ == '__main__':
    main()
