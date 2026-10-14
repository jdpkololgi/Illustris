"""Frozen 2x2 endpoint intervention on the already trained affine CFM."""
import argparse
from pathlib import Path
import numpy as np
import torch
import e2e_exterior_cfm as cfm
import e2e_gaussian_block_sampler as block
import e2e_restricted_context as context
import e2e_true_coarse_fraction_saved as io


def run(output):
    io.compute_guard(); torch.set_num_threads(4)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path('/global/u2/d/dkololgi/diagnostics/restricted_context_cfm_20261009_v1/cfm_train/model_20000.pt')
    sources = [io.record(p) for p in (__file__, cfm.__file__, block.__file__, context.__file__, checkpoint)]
    data = cfm.GaussianTrainingData()
    model = cfm.TimeAffineCFM(8, 8, data.context_dimensions)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    learned = cfm.heun_affine(cfm.learned_coefficients(model, 512), 8)
    exact = (np.linalg.cholesky(data.exact_covariance),
             np.concatenate((data.exact_response, np.zeros((8, 8, 1))), axis=-1))
    probes, names, _ = block.probe_bank(4, np.eye(64)); rows = []
    for mean in ('exact', 'learned'):
        for covariance in ('exact', 'learned'):
            endpoint = ((exact if covariance == 'exact' else learned)[0],
                        (exact if mean == 'exact' else learned)[1])
            for ci, case in enumerate(data.heldout()):
                local = cfm.kernels(data, endpoint, case)
                global_update = block.ConditionalUpdate.make(case['precision'], case['mu'],
                    block.orthonormal_span(block.global_modes(4)))
                for order in ('forward', 'reverse'):
                    affine = block.transition(local if order == 'forward' else local[::-1])
                    for global_step in (False, True):
                        transition = global_update.compose(*affine) if global_step else affine
                        result, _, sigma = context.stationary(transition, case, probes)
                        if mean == covariance == 'exact' and not result['accuracy_passed']:
                            raise AssertionError('Exact/exact intervention control failed')
                        rows.append(dict(mean=mean, covariance=covariance, observation=ci, order=order,
                            oracle_global=global_step, stationary=result,
                            tau=block.autocorrelation(transition, sigma, probes).tolist() if result['stable'] else None))
    for source in sources:
        if io.sha(source['path']) != source['sha256']: raise AssertionError('Source changed')
    io.write(out/'COMPLETE.json', dict(engineering_complete=True, rows=rows, sources=sources,
        probe_names=names, checkpoint_update=20000, nfe=512, new_training=False,
        interpretation='Endpoint counterfactual interventions, not separately trained deployable models'))
    print('HYBRID COMPLETE', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    run(p.parse_args().output)
