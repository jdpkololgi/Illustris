"""Small-receipt checkpoint summary and descriptive covariance decomposition."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_true_density_closure as b
import numpy as np


def main():
    root=b.ROOT/'verification';report=dict(checkpoints=[],covariance=[])
    for seed in (17,29):
        record=json.loads((root/'checkpoint_shear_20261005_v1'/f'seed{seed}'/'COMPLETE.json').read_text())
        for phase in ('ph012','ph013'):
            rows=[r for r in record['cases'] if r['phase']==phase]
            assert len(rows)==16
            for arm in ('coarse13312','coarse26624'):
                vals=[r['arms'][arm] for r in rows]
                entry=dict(seed=seed,phase=phase,arm=arm)
                for target in ('fullbox','matched'):
                    width=[v[target]['width'] for v in vals];trace=[v[target]['trace'] for v in vals]
                    means=np.concatenate([v['mean'] for v in width]);truth=np.concatenate([v['truth'] for v in width])
                    entry[target]=dict(width_mean=float(means.mean()),truth_mean=float(truth.mean()),
                        width_ratio=float(means.mean()/truth.mean()),width_sd=float(np.mean([v['sd'] for v in width])),
                        width_bias=float(np.mean(means-truth)),width_mean_z=float(np.mean([v['z'] for v in width])),
                        width90=float(np.mean([v['covered90'] for v in width])),
                        joint_width90=float(np.mean([v['joint']['covered']['0.9'] for v in width])),
                        trace_sd=float(np.mean([v['sd'] for v in trace])),trace90=float(np.mean([v['covered90'] for v in trace])))
                report['checkpoints'].append(entry)
    # Descriptive phase-specific bias / scatter / covariance split. It is NOT a
    # fit, significance test or calibration correction, because cases overlap.
    for phase in ('ph012','ph013'):
        with np.load(root/'trace_shear_20261005_v1'/(phase+'_REGIONAL_DRAWS.npz')) as f:
            x=f['eigenvalues'];y=f['matched']
        for name,draw,target in [('trace',x.sum(-1),y.sum(-1)),
            ('gaps',np.diff(x,axis=-1).reshape(16,32,4),np.diff(y,axis=-1).reshape(16,4))]:
            cov=np.mean([np.cov(v.T) for v in draw],axis=0)
            residual=draw.mean(1)-target;bias=residual.mean(0)
            scatter=np.cov(residual.T)-cov/32
            total=residual.T@residual/16-cov/32
            chol=np.linalg.cholesky(cov)
            whiten=lambda a:np.linalg.solve(chol,np.linalg.solve(chol,a).T).T
            report['covariance'].append(dict(phase=phase,feature=name,
                bias=bias.tolist(),mean_posterior_covariance=cov.tolist(),
                residual_scatter_minus_draw_mean_noise=scatter.tolist(),
                total_residual_second_moment_minus_draw_mean_noise=total.tolist(),
                covariance_whitened_scatter_eigenvalues=np.linalg.eigvalsh(whiten(scatter)).tolist(),
                squared_mean_bias_in_mean_covariance=float(bias@np.linalg.solve(cov,bias)),
                scope='16 dependent pairs, one phase; descriptive only, no significance or fitted correction'))
    dest=root/'trace_shear_20261005_v1'/'FOLLOWUP_SUMMARY.json'
    b.c.atomic_json(dest,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
