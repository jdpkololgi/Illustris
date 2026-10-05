"""Independent read-only audit of the frozen full-cap development scores."""
import json
import os
from pathlib import Path
import sys

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1')
SOURCE = ROOT/'source_78156ac3417555ef269f'
sys.path.insert(0, str(SOURCE))
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_exterior_final_score as auth


def independent_rank(samples, target):
    """Direct centered-matrix covariance and Cholesky, not the score helper."""
    z = np.concatenate((np.asarray(target, float)[None], np.asarray(samples, float)))
    scores = []
    for i in range(len(z)):
        x = z[np.arange(len(z)) != i]
        mu = np.sum(x, axis=0)/len(x)
        centered = x-mu
        cov = centered.T@centered/(len(x)-1)
        diagonal = np.diag(cov).copy()
        cov *= .8
        cov.flat[::len(target)+1] += .2*diagonal+1e-8*max(diagonal.mean(), 1e-12)
        q = np.linalg.solve(np.linalg.cholesky(cov), z[i]-mu)
        scores.append(float(q@q))
    scores = np.asarray(scores)
    u = np.random.default_rng(119).uniform()
    return float(((scores < scores[0]).sum()+u*(scores == scores[0]).sum())/len(z))


def run():
    c.require_compute()
    freeze = ROOT/'SELECTION_FREEZE.json'
    layout = json.loads((SOURCE/'configs/e2e_coupled_product_layout_v1.json').read_text())
    result = dict(job=os.environ['SLURM_JOB_ID'], freeze_sha256=c.sha256(freeze),
                  script_sha256=c.sha256(__file__), phases={}, production_modified=False)
    for phase in ('ph012', 'ph013'):
        saved = json.loads((ROOT/'interim_scores'/phase/'SCORES.json').read_text())
        by_id = {v['pair_id']: v['metrics'] for v in saved['cases']}
        checks = []
        for row in auth.geometry(phase, freeze):
            values, provenance = auth.truth(phase, row['pair_id'], freeze)
            truth_rho = np.stack([values['rho_joint'][tuple(slice(a, b) for a, b in crop)]
                                  for crop in layout['owned_core_crops_in_joint']])
            tensor = values['fullbox_tensor_cores']
            matrix = np.empty((*tensor.shape[:-1], 3, 3))
            for k, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
                matrix[..., i, j] = matrix[..., j, i] = tensor[..., k]
            actual = np.linalg.eigvalsh(matrix)
            truth_trace = float(np.max(abs(actual.sum(-1)-(truth_rho-1))))
            eigen_chunks, rho_chunks = [], []
            identity_error, trace_error = 0., 0.
            for first in range(0, 32, 4):
                directory = ROOT/'fullcap_tides'/phase/row['cap']/f'd{first:03d}'
                rec = json.loads((directory/'COMPLETE.json').read_text())
                verification = json.loads((directory/'VERIFIED.json').read_text())
                assert rec['first'] == first and rec['draws'] == 4 and rec['passed']
                assert verification['passed'] and verification['complete_receipt']['sha256'] == c.sha256(directory/'COMPLETE.json')
                with h5py.File(rec['output']['path']) as f, h5py.File(ROOT/'fullcap_density'/phase/row['cap']/f'd{first:03d}'/'FIELD.h5') as raw:
                    assert f.attrs['first_draw'] == first and float(f.attrs['cell_mpc_h']) == 6.766
                    er, rr = [], []
                    for crop in layout['owned_core_crops_in_joint']:
                        # Derive each core from the layout, independently of
                        # the scorer's single slab followed by reshape.
                        raw_start = (np.asarray(row['center'])+layout['joint_raw_start_from_first_center']
                                     +2*np.asarray(crop)[:, 0])
                        physical = np.asarray(row['grid']['origin_mpc_h'])+3.383*raw_start
                        offset = (physical-np.asarray(f.attrs['origin_mpc_h']))/6.766
                        start = np.rint(offset).astype(int)
                        assert np.max(abs(start-offset)) < 1e-8
                        cut = tuple(slice(int(x), int(x)+16) for x in start)
                        pad = int(f.attrs['padding_cells'])
                        wide = tuple(slice(int(x)+pad, int(x)+pad+16) for x in start)
                        e = f['eigenvalues'][(slice(None), *cut, slice(None))]
                        r = f['rho_completed'][(slice(None), *wide)]
                        inferred = raw['rho'][(slice(None), *cut)]
                        coverage = raw['coverage'][cut] > 0
                        identity_error = max(identity_error, float(np.max(abs(r[:, coverage]-inferred[:, coverage]))))
                        trace_error = max(trace_error, float(np.max(abs(e.sum(-1)-(r-1)))))
                        er.append(e); rr.append(r)
                    eigen_chunks.append(np.stack(er, axis=1)); rho_chunks.append(np.stack(rr, axis=1))
            eig, rho = np.concatenate(eigen_chunks), np.concatenate(rho_chunks)
            assert np.isfinite(eig).all() and np.all(np.diff(eig, axis=-1) >= 0)
            features = eig.mean((2, 3, 4), dtype=np.float64).reshape(32, 6)
            target = actual.mean((1, 2, 3)).reshape(6)
            rank = independent_rank(features, target)
            stored = by_id[row['pair_id']]
            old_rank = stored['joint_tidal_regional']['rank']
            # Check marginal central intervals independently: order statistics
            # at indices 1 and 30 include 29/33 mass under exchangeability.
            ordered = np.sort(eig, axis=0)
            marginal = ((actual >= ordered[1]) & (actual <= ordered[30])).mean((0, 1, 2, 3))
            z = (features.mean(0)-target)/features.std(0, ddof=1)
            marginal_regional = ((target >= np.sort(features, axis=0)[1]) &
                                 (target <= np.sort(features, axis=0)[30]))
            assert identity_error == 0 and trace_error < 2e-6 and truth_trace < 2e-6
            assert abs(rank-old_rank) < 1e-10, (row['pair_id'], rank, old_rank)
            assert np.max(abs(marginal-[v['coverage']['0.9'] for v in stored['tidal']])) < 2/8192
            checks.append(dict(pair_id=row['pair_id'], cap=row['cap'], rank=rank,
                               saved_rank=old_rank, covered90=rank <= .9,
                               marginal90=marginal.tolist(), regional_marginal90=marginal_regional.tolist(),
                               regional_bias_in_sd=z.tolist(), truth_trace_max=truth_trace,
                               predicted_trace_max=trace_error, interior_max_difference=identity_error))
            print(json.dumps(checks[-1]), flush=True)
        result['phases'][phase] = dict(cases=checks, all32_draws=True,
            coverage90=sum(v['covered90'] for v in checks)/len(checks),
            regional_marginal90=float(np.mean([v['regional_marginal90'] for v in checks])),
            source=c.file_record(ROOT/'interim_scores'/phase/'SCORES.json', content_hash=True))
    # Uniform-rank oracle, independent from the existing production controls.
    rng = np.random.default_rng(20261005)
    oracle, under = [], []
    for i in range(1200):
        z = rng.normal(size=(33, 6)); z[:, 3:] += .85*z[:, :3]
        oracle.append(independent_rank(z[1:], z[0]) <= .9)
        if i < 200:
            under.append(independent_rank(.3*z[1:], z[0]) <= .9)
    result['independent_oracle'] = dict(cases=len(oracle), coverage90=float(np.mean(oracle)),
                                      attainable=29/33, underdispersed90=float(np.mean(under)))
    assert abs(np.mean(oracle)-29/33) < .035 and np.mean(under) < .5
    result['passed'] = True
    c.atomic_json(ROOT/'verification/score_audit_20261005_v1/AUDIT.json', result)
    print(json.dumps(result['independent_oracle']), flush=True)


if __name__ == '__main__':
    run()
