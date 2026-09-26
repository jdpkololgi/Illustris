"""Equal-phase descriptive error/spread comparisons; no automatic calibration."""
import argparse
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_coarse_controls import BASE,OPEN,panel

def metrics(rows,group,weights=None):
    sl=slice(0,2) if group=='core_mass' else slice(2,10)
    error=np.asarray([r['mean'] for r in rows])[:,sl]-np.asarray([r['truth'] for r in rows])[:,sl]
    variance=np.asarray([r['variance'] for r in rows])[:,sl]
    avg=lambda x:float(np.average(np.asarray(x).reshape(len(rows),-1).mean(1),weights=weights))
    bias=avg(error);mse=avg(error**2);spread=float(np.sqrt(avg(variance)))
    return dict(bias=bias,rmse=float(np.sqrt(mse)),centered_rmse=float(np.sqrt(avg((error-bias)**2))),
        rms_spread=spread,error_spread_ratio=float(np.sqrt(mse)/spread),
        mean_error_mse_mc_corrected=mse-avg(variance)/32,
        crps=avg([r['scores'][group]['crps'] for r in rows]),
        coverage90=avg([r['scores'][group]['coverage90'] for r in rows]),
        width90=avg([r['scores'][group]['width90'] for r in rows]))

def report(root):
    root=Path(root);ids=panel();rows=[];input_hashes={}
    for seed in (17,29):
        for step in (13312,26624):
            folder=root/f'neural_{seed}_{step}'
            done=json.loads((folder/'COMPLETE.json').read_text())
            if done['cases']!=52:raise ValueError('incomplete neural worker')
            for phase,pair in ids:
                if phase in c.TRAIN:
                    path=folder/f'{pair}.json';v=json.loads(path.read_text())
                    if v['binding']!=done['binding']:raise ValueError('neural case binding')
                else:
                    name=('cfm_replication_20260925_v1' if phase in ('ph014','ph015') else
                          'cfm_pilot_eval_20260924_v1' if step==13312 else 'cfm_pilot_final_eval_20260925_v1')
                    path=BASE/name/'results'/f'seed{seed}_step{step}'/phase/pair/'nfe128/COMPLETE.json'
                    saved=json.loads(path.read_text());s=saved['scores']
                    if saved['draws']!=32 or saved['phase']!=phase:raise ValueError('saved panel mismatch')
                    v=dict(phase=phase,pair=pair,truth_sha256=saved['truth_sha256'],truth=s['region_truth'][:10],
                        mean=s['region_mean'][:10],variance=(np.asarray(s['region_std'][:10])**2).tolist(),
                        scores={g:dict(crps=s[g]['crps'],coverage90=s[g]['coverage']['0.9'],width90=s[g]['width90'])
                            for g in ('core_mass','block_mass')},
                        rank_histograms={g:s[g]['rank_histogram'] for g in ('core_mass','block_mass')})
                input_hashes[str(path)]=c.sha256(path)
                rows.append(dict(model=f'CFM_{seed}_{step}',**v))
    fit_sha=c.sha256(root/'FIT.json')
    for phase,pair in ids:
        path=root/'classical'/f'{pair}.json';v=json.loads(path.read_text())
        if v['fit_sha256']!=fit_sha:raise ValueError('classical fit changed')
        rows.append(dict(model='classical',**v));input_hashes[str(path)]=c.sha256(path)
    for phase,pair in ids:
        selected=[r for r in rows if r['phase']==phase and r['pair']==pair]
        if len(selected)!=5 or len({r['truth_sha256'] for r in selected})!=1:raise ValueError('unmatched truth')
    summaries=[]
    for model in sorted({r['model'] for r in rows}):
        for phase in c.TRAIN+OPEN:
            selected=[r for r in rows if r['model']==model and r['phase']==phase]
            summaries.append(dict(model=model,phase=phase,panel='train' if phase in c.TRAIN else 'development',
                groups={g:metrics(selected,g) for g in ('core_mass','block_mass')}))
    regimes=[]
    for model in sorted({r['model'] for r in rows}):
        for phase in c.TRAIN+OPEN:
            for key in ('cap','shell','support'):
                index={'cap':1,'shell':2,'support':3}[key]
                relevant=[r for r in rows if r['model']==model and r['phase']==phase]
                for value in sorted({r['pair'].split('_')[index] for r in relevant}):
                    selected=[r for r in relevant if r['pair'].split('_')[index]==value]
                    regimes.append(dict(model=model,phase=phase,regime=key,value=value,anchors=len(selected),
                        groups={g:metrics(selected,g) for g in ('core_mass','block_mass')}))
    matched=[]
    for model in sorted({r['model'] for r in rows}):
        for side in ('train','development'):
            selected=[r for r in rows if r['model']==model and (r['phase'] in c.TRAIN)==(side=='train')]
            keys=['_'.join(r['pair'].split('_')[1:4]) for r in selected]
            if len(set(keys))!=16:raise ValueError('missing matched observational stratum')
            weights=np.array([1/keys.count(k) for k in keys])
            matched.append(dict(model=model,panel=side,groups={g:metrics(selected,g,weights) for g in ('core_mass','block_mass')}))
    result=dict(phase_summaries=summaries,matched_strata=matched,regimes=regimes,rows=rows,input_hashes=input_hashes,
        interpretation='descriptive train versus development; correlated regions; no universal calibration claim',
        classical_limit='coarse compressed observations; empirical Gaussian likelihood; not all neural information',sealed=['ph016','ph017','ph018','ph019'])
    c.atomic_json(root/'COMPLETE.json',result,replace=True)
    lines=['# Coarse error/spread and classical comparison','',
        'Equal-phase averages of phase diagnostics. Training52 anchors/13phases; development64 anchors/4phases.',
        'Classical fit is in-sample on training anchors; it is not cross-validated there.',
        'Coverage benchmark87.8788%. No automatic model promotion.','',
        '|Model|Panel|Group|RMSE|RMS spread|Coverage|CRPS|','|---|---|---|---:|---:|---:|---:|']
    for model in sorted({r['model'] for r in rows}):
        for side in ('train','development'):
            for g in ('core_mass','block_mass'):
                selected=[r['groups'][g] for r in summaries if r['model']==model and r['panel']==side]
                vals=[np.mean([r[k] for r in selected]) for k in ('rmse','rms_spread','coverage90','crps')]
                lines.append('|'+ '|'.join([model,side,g]+[f'{v:.5g}' for v in vals])+'|')
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);report(Path(p.parse_args().root))
