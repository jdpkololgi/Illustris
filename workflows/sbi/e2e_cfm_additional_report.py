"""Frozen mixed-model assessment on the newly authorized ph016/ph017 panel."""
import argparse
import json
from pathlib import Path
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_cfm_regime_check import summarize

def report(root):
    root=Path(root);rows=[];hashes={};refinements=[]
    for seed in (17,29):
        for panel,phase in [('additional16','ph016'),('additional17','ph017')]:
            worker=root/panel/f'seed{seed}_step13312_fine26624'
            binding=json.loads((worker/'BINDING.json').read_text())
            done=json.loads((worker/'COMPLETE.json').read_text())
            if done['binding']!=c.digest(binding) or len(done['cases'])!=17:raise ValueError('incomplete worker')
            if (binding['step'],binding['fine_step'],binding['weights'])!=(13312,26624,'ema'):raise ValueError('configuration changed')
            if binding['evaluation_phases']!=[phase] or binding['seed']!=seed:raise ValueError('panel changed')
            seen=set()
            for folder in done['cases']:
                path=Path(folder)/'COMPLETE.json';v=json.loads(path.read_text());hashes[str(path)]=c.sha256(path)
                if v['binding']!=done['binding'] or v['phase']!=phase:raise ValueError('case binding mismatch')
                for name,checksum in v['chunks'].items():
                    if c.sha256(path.parent/name)!=checksum:raise ValueError('draw checksum mismatch')
                if v['nfe']==256:
                    if v['draws']!=8:raise ValueError('refinement draws')
                    refinements.append(dict(seed=seed,phase=phase,field_rms=v['paired_field_rms'],
                        power_change=[a['sample_power']/b['sample_power']-1 for a,b in zip(v['scores']['spectra'],v['paired_base8_scores']['spectra'])]))
                    continue
                if v['nfe']!=128 or v['draws']!=32 or v['pair_id'] in seen:raise ValueError('main panel mismatch')
                seen.add(v['pair_id']);rows.append(dict(arm='mixed',seed=seed,phase=phase,pair=v['pair_id'],scores=v['scores']))
            if len(seen)!=16:raise ValueError('anchor count')
    if len(rows)!=64 or len(refinements)!=4:raise ValueError('incomplete assessment')
    summaries={phase:summarize([r for r in rows if r['phase']==phase]) for phase in ('ph016','ph017')}
    by_seed={f'{phase}_{seed}':summarize([r for r in rows if (r['phase'],r['seed'])==(phase,seed)]) for phase in ('ph016','ph017') for seed in (17,29)}
    c.atomic_json(root/'COMPLETE.json',dict(rows=rows,summaries=summaries,by_seed=by_seed,refinements=refinements,input_hashes=hashes,
        interpretation='Frozen candidate replication, not a proof of joint calibration. ph018/ph019 remain sealed.'),replace=True)
    lines=['# Frozen mixed CFM additional-phase replication','','|Phase|Core coverage|Block coverage|Fine coverage|Fine RMSE/spread|Tidal coverage|Eigengap coverage|','|---|---:|---:|---:|---:|---:|---:|']
    for phase,s in summaries.items():
        values=[s[g]['coverage90'] for g in ('core_mass','block_mass','fine_mass')]+[s['fine_mass']['rmse_over_spread']]+[s[g]['coverage90'] for g in ('tidal','eigengap')]
        lines.append('|'+phase+'|'+'|'.join(f'{v:.5f}' for v in values)+'|')
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);report(p.parse_args().root)
