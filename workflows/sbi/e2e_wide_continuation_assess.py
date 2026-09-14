"""Apply preregistered diagnostic gates; bounded training-only failure attribution."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_continue import checked_binding
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION


def gates(report, spec):
    g=spec['diagnostic_gates_not_release_criteria']
    checks={}
    for name,stats in report['optimization'].items():
        curves={'all':stats['means_288_336_384'],**stats['by_phase']}
        checks[name+'/loss_plateau']=all(np.all(np.abs(np.diff(v)/np.asarray(v[:-1])) <=
            g['plateau_absolute_relative_loss_change_max_each_of_last_two_intervals']) for v in curves.values())
    for method in ('cfm','diffusion'):
        checks[method+'/eigen_drift']=bool(np.max(report['late_checkpoint_draw_drift']
            ['median_eigen_rms_change_over_draw_std'][method]) <= g['paired_checkpoint_median_eigen_rms_over_draw_std_max_each_component'])
        lo,hi=g['physics_band_power_ratio_median_range_each_band']
        power=np.asarray(report['parent_density_power']['median_ratio'][method])
        checks[method+'/power']=bool(np.all((power>=lo)&(power<=hi)))
        for mask in ('observed','complete'):
            checks[f'{method}/{mask}/density_support']=bool(report['science'][method][mask]
                ['density_below_minus_one_fraction']['median'] <= g['physics_density_below_minus_one_anchor_median_max'])
            rows=[x[mask+'_functionals'] for x in report['late_checkpoint_draw_drift']['rows'] if x['method']==method]
            checks[f'{method}/{mask}/void_drift']=bool(np.median([x['largest_void_abs_change'] for x in rows]) <=
                g['paired_checkpoint_median_largest_void_change_max'])
            checks[f'{method}/{mask}/connection_drift']=bool(sum(x['connection_changed'] for x in rows) <=
                g['paired_checkpoint_connection_changes_max'])
    return checks


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    device=p.runtime()
    report=json.loads((args.root/'INTERPRETATION_SUMMARY.json').read_text())
    evaluation=json.loads((args.root/'EVALUATION_COMPLETE.json').read_text())
    c,_,_,_=preflight()
    ds=p.dataset_for(c,NORMALIZATION)
    train_root=Path(evaluation['registration']['continuation_root'])
    binding=checked_binding(train_root,p.provenance(c,ds))
    spec=binding['continuation']['specification']
    checks=gates(report,spec)
    result={'gates':checks,'all_diagnostic_gates_pass':all(checks.values()),
        'stop_update':384,'automatic_extension':False,'training_ready':False,
        'heldout_payloads_read':False,'calibration_pass':None,'source_sha256':p.sha256(__file__)}
    models={}
    controls=[]
    if not all(checks.values()):
        for method in ('cfm','diffusion'):
            for stage in ('coarse','fine'):
                state=p.load_checkpoint(train_root/f'{method}_{stage}/step_000384.pt',binding,stage,method)
                model=p.build_model(c,stage,device).eval();model.load_state_dict(state['model'])
                models[method,stage]=model
        anchors=evaluation['registration']['refinement_anchors']
        for index,row in enumerate(ds.rows):
            if row['anchor_id'] not in anchors:
                continue
            item=ds[index]
            partner=next(i for i,r in enumerate(ds.rows) if r['phase']==row['phase'] and
                r['shell']==row['shell'] and r['support_stratum']==row['support_stratum'] and r['cap']!=row['cap'])
            shuffled=ds.inference_conditions(partner)
            for method in ('cfm','diffusion'):
                path=args.root/f'{row["anchor_id"]}_{method}.h5'
                saved=json.loads(path.with_suffix('.json').read_text())
                if p.sha256(path)!=saved['sample_sha256']:
                    raise ValueError('generated parent checksum changed')
                with h5py.File(path,'r') as f:
                    generated_coarse=f['0/coarse_delta'][:]
                generated_shared=ds.normalize_targets(p.coarse_to_fine(generated_coarse,fine_side=96,factor=4),'coarse')
                for stage in ('coarse','fine'):
                    target=p.tensor(item[stage+'_target'],device)
                    modes={}
                    for mode in ('matched','shuffled_observations','generated_coarse'):
                        if stage=='coarse' and mode=='generated_coarse':
                            continue
                        cond=item[stage+'_condition'].copy()
                        wide=item['coarse_condition']
                        if mode=='shuffled_observations':
                            if stage=='coarse':
                                cond=shuffled['coarse_condition']
                            else:
                                cond[:-1]=shuffled['fine_condition'][:-1]
                            wide=shuffled['coarse_condition']
                        if mode=='generated_coarse':
                            cond[-1]=generated_shared
                        values=[]
                        for rep in range(4):
                            gen=torch.Generator(device=device).manual_seed(p.seed_for(904,row['anchor_id'],rep,stage))
                            fn=p.flow_matching_loss if method=='cfm' else p.diffusion_loss
                            values.append(float(fn(models[method,stage],target,p.tensor(cond,device),gen,
                                wide_condition=p.tensor(wide,device) if stage=='fine' else None)))
                        modes[mode]=float(np.mean(values))
                    baseline=next(x['losses']['384'] for x in evaluation['fixed_loss_probes'] if
                        x['anchor_id']==row['anchor_id'] and x['method']==method and x['stage']==stage)
                    if not np.isfinite(list(modes.values())).all() or not np.isclose(modes['matched'],baseline,rtol=1e-6,atol=1e-7):
                        raise ValueError('control matched-noise baseline mismatch')
                    controls.append({'anchor_id':row['anchor_id'],'phase':row['phase'],'method':method,'stage':stage,
                        'shuffled_anchor':ds.rows[partner]['anchor_id'],'losses':modes})
            print(f'CONDITION CHECK {anchors.index(row["anchor_id"])+1}/24 {row["anchor_id"]}',flush=True)
    result['controls']=controls
    result['control_scope']='Training-panel stress tests, not held-out information gain. Fine shuffled-observation control retains true coarse; generated-coarse control changes only the shared-coarse condition using saved draw zero.'
    result['control_summary']={}
    for method in ('cfm','diffusion'):
        for stage in ('coarse','fine'):
            rows=[x for x in controls if x['method']==method and x['stage']==stage]
            if rows:
                means={k:float(np.mean([x['losses'][k] for x in rows])) for k in rows[0]['losses']}
                result['control_summary'][method+'/'+stage]={'mean_losses':means,
                    'relative_change_vs_matched':{k:float(v/means['matched']-1) for k,v in means.items() if k!='matched'}}
    preflight()
    if checked_binding(train_root,p.provenance(c,ds)) != binding:
        raise ValueError('continuation binding changed')
    p.write_json(args.root/'CONTINUATION_ASSESSMENT.json',result)
    print(json.dumps({'gates':checks,'controls':result['control_summary']},indent=2),flush=True)


if __name__=='__main__':
    main()
