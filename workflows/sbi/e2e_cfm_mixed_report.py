"""Matched comparison of saved same-step and new mixed-step EMA ensembles."""
import argparse
import json
from pathlib import Path
import statistics as st
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_cfm_pilot_evaluate import TRAIN_ROOT

def report(root):
    root=Path(root);rows=[];hashes={};refinements=[]
    for panel,phases in (('development',('ph012','ph013')),('replication',('ph014','ph015'))):
        for seed in (17,29):
            folder=root/panel/f'seed{seed}_step13312_fine26624'
            done=json.loads((folder/'COMPLETE.json').read_text());binding=json.loads((folder/'BINDING.json').read_text())
            if len(done['cases'])!=34 or done['binding']!=c.digest(binding):raise ValueError('mixed worker incomplete')
            if binding['step']!=13312 or binding['fine_step']!=26624:raise ValueError('wrong checkpoint pair')
            for name in done['cases']:
                path=Path(name)/'COMPLETE.json';mixed=json.loads(path.read_text())
                if mixed['phase'] not in phases or mixed['binding']!=done['binding']:raise ValueError('mixed case binding')
                hashes[str(path)]=c.sha256(path)
                if mixed['nfe']==256:
                    a=mixed['scores'];b=mixed['paired_base8_scores']
                    refinements.append(dict(seed=seed,phase=mixed['phase'],field_rms=mixed['paired_field_rms'],
                        spectral_relative_change=[(x['sample_power']-y['sample_power'])/y['sample_power'] for x,y in zip(a['spectra'],b['spectra'])]))
                    continue
                phase=mixed['phase'];pid=mixed['pair_id']
                for arm,step in (('13k',13312),('26k',26624),('mixed',None)):
                    if step is None:v=mixed
                    else:
                        old='cfm_replication_20260925_v1' if panel=='replication' else ('cfm_pilot_eval_20260924_v1' if step==13312 else 'cfm_pilot_final_eval_20260925_v1')
                        source=TRAIN_ROOT.parent/old/'results'/f'seed{seed}_step{step}'/phase/pid/'nfe128/COMPLETE.json'
                        v=json.loads(source.read_text());hashes[str(source)]=c.sha256(source)
                    if v['truth_sha256']!=mixed['truth_sha256'] or v['draws']!=32:raise ValueError('unmatched comparison')
                    if arm=='13k':
                        for key in ('region_mean','region_std','region_truth'):
                            if not np.allclose(np.asarray(v['scores'][key])[:10],np.asarray(mixed['scores'][key])[:10],rtol=2e-5,atol=1e-7):
                                raise ValueError('coarse-controlled regional mass changed')
                    rows.append(dict(arm=arm,seed=seed,phase=phase,pair=pid,scores=v['scores']))
    if len(rows)!=384:raise ValueError('main ledger incomplete')
    summaries=[]
    for arm in ('13k','26k','mixed'):
        for seed in (17,29):
            for phase in ('ph012','ph013','ph014','ph015'):
                s=[r['scores'] for r in rows if (r['arm'],r['seed'],r['phase'])==(arm,seed,phase)]
                if len(s)!=16:raise ValueError('phase panel incomplete')
                groups={g:{'crps':st.mean(x[g]['crps'] for x in s),'coverage90':st.mean(x[g]['coverage']['0.9'] for x in s),
                    'width90':st.mean(x[g]['width90'] for x in s)} for g in ('density','core_mass','block_mass','fine_mass','tidal','eigengap')}
                summaries.append(dict(arm=arm,seed=seed,phase=phase,groups=groups,
                    joint_energy=st.mean(x['joint_energy'] for x in s),variogram=st.mean(x['matched_variogram'] for x in s),
                    brier=st.mean(x['class_brier'] for x in s),
                    power_ratios=[sum(x['spectra'][i]['sample_power'] for x in s)/sum(x['spectra'][i]['truth_power'] for x in s) for i in range(5)]))
    c.atomic_json(root/'COMPLETE.json',dict(rows=rows,summaries=summaries,refinements=refinements,input_hashes=hashes,
        scientific_pass=False,interpretation='four exposed phases; coarse mass agreement is algebraic, not a new calibration result'),replace=True)
    lines=['# Mixed coarse13k / fine26k development comparison','',
        'Same coarse mass by construction; examine fine-sensitive statistics and compatibility. No confirmation opened.','',
        '|Arm|Seed|Phase|Density CRPS|Core coverage|Fine-region coverage|Joint energy|Brier|Top power ratio|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        g=s['groups'];lines.append(f"|{s['arm']}|{s['seed']}|{s['phase']}|{g['density']['crps']:.5g}|{g['core_mass']['coverage90']:.4f}|{g['fine_mass']['coverage90']:.4f}|{s['joint_energy']:.5g}|{s['brier']:.5g}|{s['power_ratios'][-1]:.4f}|")
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);report(p.parse_args().root)
