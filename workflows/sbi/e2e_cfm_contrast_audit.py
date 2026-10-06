"""Read-only contrast audit of the four already-exposed September-30 gallery cases.

No fitting, new draws, or phase selection. Output must be a new directory.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from workflows.sbi import e2e_cfm_pilot_evaluate as evaluate
from workflows.sbi import e2e_coupled_conditions as conditions
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_operators as op

SAVED = c.ROOT.parent / 'cfm_additional_20260926_v1'


def stats(values):
    return dict(mean=float(values.mean()), spatial_sd=float(values.std()),
                minimum=float(values.min()), maximum=float(values.max()),
                q99=float(np.quantile(values, .99)), q999=float(np.quantile(values, .999)),
                fraction_gt2=float(np.mean(values > 2)), fraction_gt4=float(np.mean(values > 4)))


def compare(draws, truth, support, owned):
    out = {}
    for label, mask in [('supported', support >= .5), ('owned_supported', (support >= .5) & owned)]:
        if not np.any(mask):
            out[label] = dict(cells=0)
            continue
        t = truth[mask].astype(float)
        d = draws[:, mask].astype(float)
        individual = [stats(row) for row in d]
        mean = d.mean(0)
        spread = np.sqrt(np.mean(d.var(0, ddof=1)))
        high = t >= np.quantile(t, .99)
        out[label] = dict(cells=int(mask.sum()), truth=stats(t), posterior_mean=stats(mean),
            individual_draws=individual,
            draw_summary={key: dict(min=min(r[key] for r in individual),
                median=float(np.median([r[key] for r in individual])),
                max=max(r[key] for r in individual)) for key in individual[0]},
            rmse_over_rms_spread=float(np.sqrt(np.mean((mean-t)**2))/spread),
            truth_top1pct_mean_prediction_over_truth=float(mean[high].mean()/t[high].mean()))
    return out


def main(output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run on the approved CPU allocation')
    output.mkdir(parents=True, exist_ok=False)
    evaluate.EVAL_PHASES = ('ph016', 'ph017')
    result = dict(job=os.environ['SLURM_JOB_ID'], seed=17, draws=32,
                  coarse_step=13312, fine_step=26624, nfe=128,
                  scope='Four fixed NGC shell-0 September-30 gallery examples; not a population calibration test',
                  caveats=['Truth-conditioned peak bias is descriptive, not a requirement that posterior means recover realized extremes.',
                           'Voxel/support and spatial-functional comparisons are not independent tests.',
                           'The display pools 2x in all axes; the gallery slab averages two z cells only.',
                           'No new posterior draws or sealed phases accessed.'], cases=[])
    for phase in evaluate.EVAL_PHASES:
        for kind in ('interior', 'boundary'):
            pair = f'{phase}_NGC_s0_{kind}_00'
            folder = SAVED / ('additional'+phase[-2:]) / 'seed17_step13312_fine26624' / phase / pair / 'nfe128'
            receipt = json.loads((folder/'COMPLETE.json').read_text())
            arrays = []
            for name, sha in sorted(receipt['chunks'].items()):
                if c.sha256(folder/name) != sha:
                    raise ValueError('Saved draw checksum changed')
                with np.load(folder/name) as f:
                    arrays.append(f['rho'].astype(float))
            draws = np.concatenate(arrays)
            if draws.shape != (32,64,48,48):
                raise ValueError(f'Unexpected draw shape: {draws.shape}')
            target, target_sha = evaluate.truth(phase, pair)
            truth = target['rho_joint']
            support = conditions.load_pair(phase,pair)['support']
            owned = np.zeros(truth.shape, dtype=bool)
            owned[16:48,16:32,16:32] = True
            pooled = np.stack([op.mean_pool(row,2) for row in draws])
            slab = lambda a: a[...,23:25].mean(-1)
            entry = dict(pair=pair,target_sha256=target_sha,draw_receipt_sha256=c.sha256(folder/'COMPLETE.json'),
                native=compare(draws,truth,support,owned),
                display_pool2=compare(pooled,op.mean_pool(truth,2),op.mean_pool(support,2),op.mean_pool(owned.astype(float),2)==1),
                gallery_slab=compare(slab(draws),slab(truth),slab(support),np.any(owned[:,:,23:25],axis=2)))
            result['cases'].append(entry)
            print(json.dumps({'pair':pair,'native':entry['native']['supported']['draw_summary'],
                              'truth':entry['native']['supported']['truth']}),flush=True)
    result['script_sha256']=c.sha256(Path(__file__))
    (output/'CONTRAST.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':str(output/'CONTRAST.json')}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
