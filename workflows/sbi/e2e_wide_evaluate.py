"""Training-panel draw/optimization/sampler diagnostics; no calibration or fit."""
import argparse
import copy
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION

TRAIN_ROOT = p.SCRATCH_ROOT/'wide_pipeline_v1/research_20260911_58196924'


def ensemble_metrics(draws, truth, mask):
    """Four-draw descriptive metrics; fair marginal CRPS, not IID voxel inference."""
    x, y = np.asarray(draws)[:, mask], np.asarray(truth)[mask]
    m = len(x)
    if m < 2 or y.size == 0 or not np.isfinite(x).all():
        raise ValueError('invalid ensemble')
    mean = x.mean(axis=0)
    scatter = y.std(axis=0)
    pair = sum(np.abs(x[i]-x[j]) for i in range(m) for j in range(i))
    crps = np.abs(x-y).mean(axis=0)-pair/(m*(m-1))
    return {'mean_eigen_rmse': np.sqrt(np.mean((mean-y)**2, axis=0)).tolist(),
            'mean_eigen_rmse_over_truth_std': (np.sqrt(np.mean((mean-y)**2,axis=0))/np.maximum(scatter,1e-30)).tolist(),
            'mean_eigen_bias': (mean-y).mean(axis=0).tolist(),
            'mean_pointwise_draw_std': x.std(axis=0,ddof=1).mean(axis=0).tolist(),
            'fair_marginal_crps': crps.mean(axis=0).tolist()}


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {'median': np.median(a,axis=0).tolist(), 'min': np.min(a,axis=0).tolist(),
            'max': np.max(a,axis=0).tolist()}


def summarize(rows):
    out = {}
    for method in ('cfm','diffusion'):
        selected = [r for r in rows if r['method']==method]
        out[method] = {}
        for mask in ('complete','observed'):
            summary = {}
            for k in selected[0]['ensemble'][mask]:
                summary[k] = distribution([r['ensemble'][mask][k] for r in selected])
            dr = [d[mask] for r in selected for d in r['draws']]
            for name in ('four_class_disagreement','eigen_rmse_over_truth_std'):
                summary['draw_'+name] = distribution([d['metrics'][name] for d in dr])
            for name in ('filling_fraction','largest_void_fraction'):
                summary[name+'_abs_error'] = distribution([abs(d['science'][name]-r['truth_science'][mask][name])
                    for r in selected for d in (v[mask] for v in r['draws'])])
            summary['connection_changed_draw_fraction'] = float(np.mean([
                d[mask]['science']['connections_xyz'] != r['truth_science'][mask]['connections_xyz']
                for r in selected for d in r['draws']]))
            summary['pair_probability_abs_error'] = distribution([
                [abs(a['value']-b['value']) for a,b in zip(d[mask]['science']['pair'],r['truth_science'][mask]['pair'])]
                for r in selected for d in r['draws']])
            out[method][mask] = summary
    return out


@torch.no_grad()
def run(output):
    device = p.runtime()
    c, _, smoke, source = preflight()
    ds = p.dataset_for(c, NORMALIZATION)
    binding = p.provenance(c, ds)
    out = p.output_path(c, output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    cfg = json.loads((p.REPO/c['diagnostic_config']).read_text())
    checkpoints, models = {}, {}
    for method in ('cfm','diffusion'):
        for stage in ('coarse','fine'):
            for step in (96,144,192):
                key = (method,stage,step)
                path = TRAIN_ROOT/f'{method}_{stage}'/f'step_{step:06d}.pt'
                state = p.load_checkpoint(path,binding,stage,method)
                if state['step'] != step:
                    raise ValueError('wrong checkpoint update')
                models[key] = p.build_model(c,stage,device).eval()
                models[key].load_state_dict(state['model'])
                checkpoints['/'.join(map(str,key))] = p.sha256(path)
                del state
    # One per phase x shell x support; alternate cap deterministically.
    refinement = []
    for phase in ('ph000','ph002','ph003'):
        for shell in sorted({r['shell'] for r in ds.rows}):
            for j,support in enumerate(sorted({r['support_stratum'] for r in ds.rows})):
                candidates = sorted([i for i,r in enumerate(ds.rows) if r['phase']==phase and
                    r['shell']==shell and r['support_stratum']==support],key=lambda i: ds.rows[i]['anchor_id'])
                cap = ('NGC','SGC')[(int(str(shell).lstrip('s'))+j)%2]
                refinement.append(next(i for i in candidates if ds.rows[i]['cap']==cap))
    registration = {'checkpoint_sha256':checkpoints,'binding_sha256':p.digest(binding),
        'evaluator_sha256':p.sha256(__file__), 'draws_per_anchor':4,'training_anchors':len(ds),
        'loss_checkpoints':[96,144,192],'fixed_loss_replicates':4,
        'refinement_anchors':[ds.rows[i]['anchor_id'] for i in refinement],
        'refinement':'same draw 0, CFM16->32 Heun / DIFF32->64 DDIM, diagnostic only',
        'claim':'training panel only; four draws are not a calibration/power study',
        'heldout_payloads_read':False,'training_ready':False}
    p.write_json(out/'EVALUATION_STARTED.json',registration)
    rows, loss_rows, refined = [], [], []
    core=(slice(32,64),)*3
    for index, row in enumerate(ds.rows):
        item=ds[index]  # teacher-forced targets/conditions for objective checks only
        for stage in ('coarse','fine'):
            target=p.tensor(item[stage+'_target'],device)
            condition=p.tensor(item[stage+'_condition'],device)
            wide=p.tensor(item['coarse_condition'],device) if stage=='fine' else None
            for method in ('cfm','diffusion'):
                lossfn=p.flow_matching_loss if method=='cfm' else p.diffusion_loss
                losses={}
                for step in (96,144,192):
                    values=[]
                    for rep in range(4):
                        rng=torch.Generator(device=device).manual_seed(p.seed_for(904,row['anchor_id'],rep,stage))
                        values.append(float(lossfn(models[method,stage,step],target,condition,rng,wide_condition=wide)))
                    if not np.isfinite(values).all():
                        raise ValueError('nonfinite fixed-probe loss')
                    losses[str(step)]=float(np.mean(values))
                loss_rows.append({'anchor_id':row['anchor_id'],'phase':row['phase'],'method':method,'stage':stage,'losses':losses})
            del target,condition,wide
        del item
        obs=ds.inference_conditions(index)  # no targets feed generation
        with h5py.File(row['shard'],'r') as f:
            g=f[row['group']]
            truth=g['tensor_spectral'][core]
            delta_truth=g['delta_r7_gaussian'][core]
            observed=g['masks/observed_parent'][core].astype(bool)
        eigen_truth=p.eigs(truth)
        masks={'complete':np.ones((32,)*3,bool),'observed':observed}
        truth_science={k:p.science(eigen_truth,m,c['fine_cell_mpc_h'],cfg) for k,m in masks.items()}
        for method in ('cfm','diffusion'):
            draws,eigen_draws,density_draws=[],[],[]
            path=out/f'{row["anchor_id"]}_{method}.h5'
            with h5py.File(path,'x') as f:
                f.attrs['registration_sha256']=p.digest(registration)
                f.attrs['anchor_id']=row['anchor_id']; f.attrs['method']=method
                for draw in range(4):
                    co,fi,seeds=p.generate_pair(c,ds,obs,models[method,'coarse',192],models[method,'fine',192],method,f'eval-{draw}',device)
                    result=p.reconstruct(co,fi)
                    group=f.create_group(str(draw)); group.attrs['seeds_json']=json.dumps(seeds)
                    for name,value in {'coarse_delta':co,'fine_residual':fi,**result}.items():
                        group.create_dataset(name,data=value)
                    eigen_draws.append(result['eigen_core'])
                    density_draws.append(result['delta_local96'][core])
                    draws.append({k:{'metrics':p.tensor_metrics(result['tensor_core'],truth,m),
                        'science':p.science(result['eigen_core'],m,c['fine_cell_mpc_h'],cfg)} for k,m in masks.items()})
                    if draw==0 and index in refinement:
                        refined_config=copy.deepcopy(c)
                        refined_config['sampling']['cfm_steps']*=2
                        refined_config['sampling']['diffusion_steps']*=2
                        rc,rf,_=p.generate_pair(refined_config,ds,obs,models[method,'coarse',192],models[method,'fine',192],method,'eval-0',device)
                        rr=p.reconstruct(rc,rf)
                        rg=f.create_group('refined_0')
                        for name,value in {'coarse_delta':rc,'fine_residual':rf,**rr}.items():
                            rg.create_dataset(name,data=value)
                        refined.append({'anchor_id':row['anchor_id'],'phase':row['phase'],'method':method,
                            'masks':{k:p.tensor_metrics(rr['tensor_core'],result['tensor_core'],m) for k,m in masks.items()}})
            ens={k:ensemble_metrics(eigen_draws,eigen_truth,m) for k,m in masks.items()}
            for k,m in masks.items():
                density=np.asarray(density_draws)[:,m]
                ens[k]['density_mean_rmse']=float(np.sqrt(np.mean((density.mean(axis=0)-delta_truth[m])**2)))
                ens[k]['density_draw_std']=float(density.std(axis=0,ddof=1).mean())
                ens[k]['density_below_minus_one_fraction']=float(np.mean(density < -1))
            record={k:row[k] for k in ('anchor_id','phase','cap','shell','support_stratum')}
            record.update(method=method,draws=draws,ensemble=ens,truth_science=truth_science,
                          sample_file=str(path),sample_sha256=p.sha256(path))
            rows.append(record)
            p.write_json(path.with_suffix('.json'),record)
        print(f'EVALUATED {index+1}/96 {row["anchor_id"]}',flush=True)
        p.write_json(out/f'loss_{index:03d}.json',loss_rows[-4:])
    preflight()
    p.write_json(out/'EVALUATION_COMPLETE.json',{'registration':registration,'elapsed_seconds':time.monotonic()-start,
        'summary':summarize(rows),'by_phase':{phase:summarize([r for r in rows if r['phase']==phase]) for phase in ('ph000','ph002','ph003')},
        'by_shell_support':{f'{shell}/{support}':summarize([r for r in rows if r['shell']==shell and r['support_stratum']==support])
            for shell,support in sorted({(r['shell'],r['support_stratum']) for r in rows})},
        'fixed_loss_probes':loss_rows,'sampler_refinement':refined,'training_ready':False,
        'calibration_pass':None,'evaluation_complete':True})
    print('EVALUATION COMPLETE: training diagnostics only',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
