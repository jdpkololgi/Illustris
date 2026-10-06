"""Phase-aware readout and existing TARP diagnostics for the fixed mock panel."""
import argparse
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_global_mock_evaluate import load_arrays, density_statistics, metric_summary
from workflows.sbi.e2e_global_mock_prepare import guard
from workflows.sbi.e2e_cfm_calibration_figures import tarp_curves

METHODS=('original','independent_tiled','shared_state')
METRICS=('density','core_mass','block_mass','fine_mass','seam_density',
         'seam_mass','seam_sum_difference','seam_average','seam_contrast','tidal','eigengap')


def aggregate(rows, draws=32):
    result={}
    for method in METHODS:
        values=[v['methods'][method] for v in rows]
        metrics={}
        for metric in METRICS:
            selected=[v[metric] for v in values if metric in v]
            if not selected:continue
            metrics[metric]={key:float(np.mean([v[key] for v in selected]))
                             for key in ('crps','bias','mse','variance','width90')}
            metrics[metric]['coverage']={p:float(np.mean([v['coverage'][p] for v in selected]))
                                        for p in ('0.5','0.68','0.9','0.95')}
            metrics[metric]['attainable']=selected[0]['attainable']
            histogram=np.sum([v['rank_histogram'] for v in selected],axis=0)
            metrics[metric]['rank_histogram']=histogram.tolist()
            metrics[metric]['rank_relative_uniform']=(histogram/histogram.mean()).tolist()
            variance=metrics[metric]['variance']
            metrics[metric]['rmse_over_sd']=(float(np.sqrt(metrics[metric]['mse']/variance))
                                            if variance>0 else None)
            corrected=metrics[metric]['mse']-metrics[metric]['variance']/draws
            metrics[metric]['mse_minus_mean_monte_carlo_variance']=corrected
            metrics[metric]['mc_corrected_rmse_over_sd']=(float(np.sqrt(corrected/metrics[metric]['variance']))
                                                         if corrected>=0 and variance>0 else None)
        for metric in ('seam_energy','seam_variogram','seam_variogram_shuffled','class_brier_0.0','class_brier_0.2'):
            selected=[v[metric] for v in values if metric in v]
            if selected:metrics[metric]=float(np.mean(selected))
        result[method]=metrics
    return result


def summarize(inputs,samples,scores,out):
    c.require_compute();out.mkdir(parents=True,exist_ok=False)
    result=json.loads(scores.read_text());rows=result['cases']
    if not result['panel_complete'] or len(rows)!=32 or {v['phase'] for v in rows}!={'ph016','ph017'}:
        raise ValueError('Do not summarize a partial panel as complete')
    groups={'all':rows}
    for key in ('phase','cap','shell','support_stratum'):
        for value in sorted({v[key] for v in rows}):
            groups[f'{key}={value}']=[v for v in rows if v[key]==value]
    for label,select in [('both_ge95',lambda v:min(v['seam_support'])>=.95),
                          ('some_below95',lambda v:min(v['seam_support'])<.95)]:
        selected=[v for v in rows if select(v)]
        if selected:groups['seam_support='+label]=selected
    prepared=json.loads((inputs/'PREPARED.json').read_text())
    if c.sha256(inputs/'PREPARED.json')!=result['binding']['inputs']:
        raise ValueError('Input/scoring provenance mismatch')
    arrays={method:[] for method in METHODS};truth=[];identities=[];case_records=[]
    for case in prepared['cases']:
        guard(case['phase']);identities.append(case['phase'])
        target=load_arrays(case['truth']);value=density_statistics(target['rho_joint'])
        truth.append(np.r_[value['regional'],value['seam_mass']])
        marker=samples/case['pair_id']/'COMPLETE.json'
        record=json.loads(marker.read_text())
        if record['binding']!=c.digest(result['binding']):raise ValueError('Case provenance mismatch')
        case_records.append(c.file_record(marker,content_hash=True))
        for item in record['outputs']:
            data=load_arrays(item);name=Path(item['path']).stem
            arrays[name].append(np.concatenate([data['regional'],data['seam_mass']],axis=1))
            row=next(v for v in rows if v['pair_id']==case['pair_id'])
            for index,label in enumerate(('seam_average','seam_contrast')):
                row['methods'][name][label]=metric_summary(data['seam_sum_difference'][:,index],
                                                          value['seam_sum_difference'][index])
    aggregate_rows={name:dict(cases=len(v),statistics=aggregate(v,result['draws'])) for name,v in groups.items()}
    truth=np.asarray(truth);identities=np.asarray(identities)
    tarp={}
    for name,values in arrays.items():
        values=np.asarray(values)
        for start,stop,label in ((0,2,'core'),(2,10,'block'),(10,18,'fine'),(18,20,'seam')):
            for width in (.25,1.):
                phase_curves=[]
                for phase in ('ph016','ph017'):
                    use=identities==phase
                    alpha,curves=tarp_curves(values[use,:,start:stop],truth[use,start:stop],seed=71,width=width)
                    tarp[f'{name}/{phase}/{label}/sd{width}']=curves.mean(0).tolist()
                    phase_curves.append(curves)
                tarp[f'{name}/all/{label}/sd{width}']=np.concatenate(phase_curves).mean(0).tolist()
    complete=dict(groups=aggregate_rows,cases=rows,tarp=dict(alpha=alpha.tolist(),curves=tarp),
                  scores=c.file_record(scores,content_hash=True),case_records=case_records,
                  draws=result['draws'],independent_phases=2,
                  limits='Two already-exposed training-unseen phases; correlated stratified anchors. '
                         'No voxel-iid uncertainty, no full-field TARP, no full-survey validation.',
                  posterior_qualified=False)
    c.atomic_json(out/'SUMMARY.json',complete)
    for group in ('all','phase=ph016','phase=ph017'):
        brief={name:{metric:dict(coverage90=v['coverage']['0.9'],crps=v['crps'],width90=v['width90'])
                     for metric,v in metrics.items() if metric in METRICS}
               for name,metrics in aggregate_rows[group]['statistics'].items()}
        print(json.dumps(dict(group=group,results=brief)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--samples',required=True,type=Path);p.add_argument('--scores',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    summarize(a.inputs,a.samples,a.scores,a.out)
