import argparse
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_coarse_controls import BASE,OPEN
from workflows.sbi.e2e_coarse_regularization import evaluation_panel
from workflows.sbi.e2e_coarse_controls_report import metrics

def run(root):
    root=Path(root);old=json.loads((BASE/'coarse_controls_20260926_v1/COMPLETE.json').read_text())
    rows=[];hashes={}
    for arm in ('baseline','decay'):
        for seed in (17,29):
            folder=root/f'{arm}_{seed}'
            if not (folder/'COMPLETE.json').exists():raise ValueError('worker incomplete')
            for step in (13312,19968,26624):
                for phase,pid in evaluation_panel():
                    if step==13312:
                        v=next(r for r in old['rows'] if r['model']==f'CFM_{seed}_13312' and r['pair']==pid)
                    else:
                        path=folder/f'eval{step}'/f'{pid}.json';v=json.loads(path.read_text());hashes[str(path)]=c.sha256(path)
                    rows.append(dict(arm=arm,checkpoint=step,replica=seed,**v))
    summary=[]
    for arm in ('baseline','decay'):
        for step in (13312,19968,26624):
            for phase in sorted({r['phase'] for r in rows}):
                selected=[r for r in rows if r['arm']==arm and r['checkpoint']==step and r['phase']==phase]
                summary.append(dict(arm=arm,step=step,phase=phase,groups={g:metrics(selected,g) for g in ('core_mass','block_mass')}))
    # Exploratory phase-swap robustness, not fresh confirmation: all four phases
    # have already informed programme decisions. Lower coverage error ranks first;
    # CRPS cannot rescue excessive narrowing. Earlier checkpoint breaks ties.
    swaps=[]
    for arm in ('baseline','decay'):
        for select,test in ((OPEN[:2],OPEN[2:]),(OPEN[2:],OPEN[:2])):
            ranked=[]
            for step in (13312,19968,26624):
                values=[r for r in summary if r['arm']==arm and r['step']==step and r['phase'] in select]
                error=max(abs(np.mean([r['groups'][g]['coverage90'] for r in values])-29/33) for g in ('core_mass','block_mass'))
                ranked.append((float(error),step))
            error,step=min(ranked)
            tested=[r for r in summary if r['arm']==arm and r['step']==step and r['phase'] in test]
            swaps.append(dict(arm=arm,selection_phases=select,assessment_phases=test,chosen_step=step,
                selection_coverage_error=error,assessment=tested))
    c.atomic_json(root/'COMPLETE.json',dict(rows=rows,summary=summary,phase_swaps=swaps,input_hashes=hashes,
        scientific_pass=False,interpretation='small-panel exploratory screen; no sealed phase access'),replace=True)
    lines=['# Coarse regularization exploratory screen','',
        'Two seeds; eight training and sixteen development anchors. No calibration qualification.',
        'Phase swaps reuse exposed development phases, not independent confirmation.','',
        '|Arm|Step|Panel|Block RMSE|Block spread|Block coverage|Core coverage|',
        '|---|---:|---|---:|---:|---:|---:|']
    for arm in ('baseline','decay'):
        for step in (13312,19968,26624):
            for side in ('train','development'):
                selected=[r for r in rows if r['arm']==arm and r['checkpoint']==step and (r['phase'] in c.TRAIN)==(side=='train')]
                b=metrics(selected,'block_mass');co=metrics(selected,'core_mass')
                lines.append(f"|{arm}|{step}|{side}|{b['rmse']:.5g}|{b['rms_spread']:.5g}|{b['coverage90']:.4f}|{co['coverage90']:.4f}|")
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
