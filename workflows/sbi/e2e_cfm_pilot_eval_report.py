"""Small JSON-only collector: no fitting, payload access or significance claims."""
import argparse
import json
from pathlib import Path
import statistics
from workflows.sbi.e2e_coupled_contract import atomic_json,digest


def collect(root,baseline_root=None,panel='development'):
    root=Path(root);rows=[]
    steps=(13312,26624) if baseline_root or panel=='replication' else (6656,13312)
    phases=('ph014','ph015') if panel=='replication' else ('ph012','ph013')
    for seed in (17,29):
        for step in steps:
            source=Path(baseline_root) if baseline_root and step==13312 else root
            folder=source/f'seed{seed}_step{step}'
            bound=json.loads((folder/'BINDING.json').read_text())
            done=json.loads((folder/'COMPLETE.json').read_text())
            if done['binding']!=digest(bound) or len(done['cases'])!=34:raise ValueError('worker incomplete or changed')
            for name in done['cases']:
                path=Path(name)
                if not path.is_relative_to(folder):raise ValueError('case outside worker root')
                case=json.loads((path/'COMPLETE.json').read_text())
                if case['binding']!=done['binding'] or case['phase'] not in phases:
                    raise ValueError('invalid case binding or phase')
                rows.append(dict(seed=seed,step=step,**case))
    main=[r for r in rows if r['nfe']==128]
    if len(main)!=128 or len({(r['seed'],r['step'],r['phase'],r['pair_id']) for r in main})!=128:
        raise ValueError('main ledger mismatch')
    for seed in (17,29):
        panels=[{(r['phase'],r['pair_id']):r for r in main if r['seed']==seed and r['step']==step} for step in steps]
        if panels[0].keys()!=panels[1].keys():raise ValueError('checkpoint panels differ')
        for key in panels[0]:
            left,right=(p[key] for p in panels)
            if left['truth_sha256']!=right['truth_sha256'] or left['draws']!=right['draws']:
                raise ValueError('checkpoint truths or draw counts differ')
    records=[]
    for seed in (17,29):
        for step in steps:
            for phase in phases:
                selected=[r for r in main if (r['seed'],r['step'],r['phase'])==(seed,step,phase)]
                if len(selected)!=16:raise ValueError('phase panel incomplete')
                mean=lambda f:statistics.mean(f(r['scores']) for r in selected)
                records.append(dict(seed=seed,step=step,phase=phase,
                    groups={g:dict(crps=mean(lambda s:s[g]['crps']),coverage90=mean(lambda s:s[g]['coverage']['0.9']),
                        width90=mean(lambda s:s[g]['width90']),bias=mean(lambda s:s[g]['bias']))
                        for g in ('density','core_mass','block_mass','fine_mass','tidal','eigengap')},
                    joint_energy=mean(lambda s:s['joint_energy']),variogram=mean(lambda s:s['matched_variogram']),
                    class_brier=mean(lambda s:s['class_brier'])))
    refinements=[]
    for r in rows:
        if r['nfe']!=256:continue
        new=r['scores'];old=r['paired_base8_scores']
        refinements.append(dict(seed=r['seed'],step=r['step'],phase=r['phase'],pair_id=r['pair_id'],
            paired_field_rms=r['paired_field_rms'],density_crps_delta=new['density']['crps']-old['density']['crps'],
            core_mass_crps_delta=new['core_mass']['crps']-old['core_mass']['crps'],
            class_brier_delta=new['class_brier']-old['class_brier'],
            spectral_relative_change=[(a['sample_power']-b['sample_power'])/max(b['sample_power'],1e-30)
                for a,b in zip(new['spectra'],old['spectra'])]))
    result=dict(records=records,refinements=refinements,main_cases=128,refinement_cases=8,
        scientific_pass_not_assigned=True,interpretation='two development phases; paired descriptive comparisons only')
    atomic_json(root/'COMPLETE.json',result,replace=True)
    lines=['# Coupled CFM development assessment','',
        'Completed ledger, not a calibration qualification. Nominal90%attainable=87.8788% at32draws.',
        'Two development phases; seeds, voxels and overlapping probes are not independent universes.','',
        '| Seed | Step | Phase | Density CRPS | Core-mass CRPS | Core coverage | Block coverage | Joint energy | Brier |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        g=r['groups'];lines.append(f"| {r['seed']} | {r['step']} | {r['phase']} | {g['density']['crps']:.5g} | {g['core_mass']['crps']:.5g} | {g['core_mass']['coverage90']:.4f} | {g['block_mass']['coverage90']:.4f} | {r['joint_energy']:.5g} | {r['class_brier']:.5g} |")
    lines+=['','Sampler refinement uses identical8draws/condition at128and256NFE; no tight coverage gate on8draws.',
        'See COMPLETE.json for paired refinement deltas and all physical marginal summaries.',
        'ph014/ph015 are now replication/development; ph016-ph019 remain sealed.' if panel=='replication' else
        'No confirmation was opened; no automatic training continuation or alpha selection.']
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--baseline-root')
    parser.add_argument('--panel',choices=('development','replication'),default='development')
    args=parser.parse_args();collect(args.root,args.baseline_root,args.panel)
