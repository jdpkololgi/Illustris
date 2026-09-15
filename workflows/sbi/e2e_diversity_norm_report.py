"""Verify and summarize the registered diversity/normalization experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_diversity_norm import source_hashes, field_for, overlap


def med(values):
    return float(np.median(values)) if len(values) else None


def groups(prepared,n):
    train=[r['anchor_id'] for r in prepared['selection']['train']]
    return dict(common_fit=train[:3],exposed=train[:n],unused=train[n:],
                transfer=[r['anchor_id'] for r in prepared['selection']['transfer']])


def summarize(rows,anchors,parent):
    rows=[r for r in rows if r['anchor_id'] in anchors];out={}
    for ratio in sorted({r['ratio'] for r in rows}):
        clean=[r for r in rows if r['kind']=='clean' and r['ratio']==ratio]
        noisy=[r for r in rows if r['kind']=='noisy' and r['ratio']==ratio]
        record={}
        if clean:
            record.update(clean_rms=med([r['rms'] for r in clean]),clean_abs_bias=med([abs(r['bias']) for r in clean]),
                clean_max_abs=max(r['max_abs'] for r in clean),clean_highk_error=med([r['metrics']['error_power'][-1] for r in clean]),
                clean_gain=np.median([r['metrics']['gain'] for r in clean],axis=0).tolist(),
                clean_rms_over_injected=med([r['rms_over_injected'] for r in clean]) if ratio else None)
        if noisy:
            checks=[]
            for anchor in anchors:
                rr=[r for r in noisy if r['anchor_id']==anchor]
                if not rr:continue
                noise=med([abs(r['metrics']['noise_amplitude'][-1]) for r in rr])
                err=med([r['metrics']['error_power'][-1]/parent[r['anchor_id'],ratio,r['rep']]['metrics']['error_power'][-1] for r in rr])
                gain=np.median([r['metrics']['gain'][:3] for r in rr],axis=0)
                checks.append(dict(anchor_id=anchor,noise_left=noise,error_vs_parent=err,lower_gain=gain.tolist(),
                    passed=bool(noise<=.2 and err<=.25 and np.all((gain>=.9)&(gain<=1.1)))))
            record.update(noise_left=med([r['noise_left'] for r in checks]),
                error_vs_parent=med([r['error_vs_parent'] for r in checks]),
                noisy_highk_error=med([r['metrics']['error_power'][-1] for r in noisy]),
                noisy_gain=np.median([r['metrics']['gain'] for r in noisy],axis=0).tolist(),
                checks=checks,passed=sum(r['passed'] for r in checks),total=len(checks))
        out[str(ratio)]=record
    return out


def ranks(x):
    x=np.asarray(x);return np.array([(np.sum(x<v)+.5*(np.sum(x==v)-1)) for v in x],dtype=float)


def correlation(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if np.std(x)<1e-14 or np.std(y)<1e-14:return None
    return float(np.corrcoef(x,y)[0,1])


def associations(rows,prepared,n):
    metadata={r['anchor_id']:r for r in prepared['metadata']};g=groups(prepared,n)
    rr=[r for r in rows if r['kind']=='clean' and r['ratio']==.05 and r['anchor_id'] in g['transfer']]
    phases=np.array([r['phase'] for r in rr]);out=[]
    def centered(x):
        x=np.array(x,dtype=float)
        return np.array([v-np.mean(x[phases==phase]) for v,phase in zip(x,phases)])
    for key in metadata[rr[0]['anchor_id']]['stats']:
        fit_mean=np.mean([metadata[a]['stats'][key] for a in g['exposed']])
        raw=np.array([metadata[r['anchor_id']]['stats'][key] for r in rr],dtype=float)
        for transform,x in [('value',raw),('absolute_distance_from_fit_mean',abs(raw-fit_mean))]:
            for outcome,y in [('rms',[r['rms'] for r in rr]),('signed_bias',[r['bias'] for r in rr]),
                              ('highk_error',[r['metrics']['error_power'][-1] for r in rr])]:
                out.append(dict(feature=key,transform=transform,outcome=outcome,n=len(rr),
                    pearson=correlation(x,y),spearman=correlation(ranks(x),ranks(y)),
                    within_phase_pearson=correlation(centered(x),centered(y))))
    return out


def verify(root):
    prep=json.loads((root/'PREPARED.json').read_text());frozen=json.loads((root/'FROZEN.json').read_text())
    digest=p.sha256(root/'PREPARED.json');cfg=prep['config'];train=[r['anchor_id'] for r in prep['selection']['train']]
    assert prep['source_sha256']==source_hashes()
    assert p.sha256(root/'cache.h5')==prep['cache_sha256']
    assert len(train)==15 and len(prep['selection']['transfer'])==12 and not prep['heldout_payloads_read']
    assert not any(overlap(a,b) for a in prep['selection']['train'] for b in prep['selection']['transfer'])
    assert frozen['complete'] and frozen['prepared_sha256']==digest and frozen['source_sha256']==source_hashes()
    assert len(frozen['roundtrip'])==135 and len(frozen['results'])==5
    for rows in list(frozen['results'].values())+[frozen['parent384']]:assert len(rows)==297
    cells=[];hashes={};inputs={str(root/'PREPARED.json'):digest,str(root/'FROZEN.json'):p.sha256(root/'FROZEN.json')}
    for replica in range(2):
        folder=root/f'replica_{replica}';path=folder/'MATRIX_COMPLETE.json';receipt=json.loads(path.read_text());inputs[str(path)]=p.sha256(path)
        assert receipt['complete'] and not receipt['heldout_payloads_read'] and not receipt['training_ready']
        reg=receipt['registration'];assert reg['prepared_sha256']==digest and reg['source_sha256']==source_hashes()
        assert len(receipt['checkpoints'])==16 and len(receipt['results'])==8
        for rel,digest_ckpt in receipt['checkpoints'].items():
            assert p.sha256(folder/rel)==digest_ckpt;hashes[str(folder/rel)]=digest_ckpt
        seen=set();reference=None
        for cell in receipt['results']:
            n=cell['n'];norm=cell['normalization'];assert (n,norm) not in seen;seen.add((n,norm))
            assert n in cfg['field_counts'] and norm in cfg['normalizations'] and cell['replica']==replica
            history=cell['history'];assert len(history)==3072
            schedule=[(r['noise_seed'],r['time'],r['ratio']) for r in history]
            if reference is None:reference=schedule
            assert schedule==reference
            assert all(r['anchor_id']==field_for(i,n,train) and r['update']==i+1 for i,r in enumerate(history))
            assert np.isfinite([r['loss'] for r in history]).all()
            assert [x['update'] for x in cell['curve']]==[0,1536,3072]
            for point in cell['curve']:
                assert len(point['rows'])==297
                assert len({(r['anchor_id'],r['kind'],r['ratio'],r.get('rep')) for r in point['rows']})==297
                for r in point['rows']:
                    if r['kind']=='clean' and r['ratio']==0:assert r['max_abs']<=1e-6
                    if r['kind']=='noisy':assert r['seed']==p.seed_for(cfg['evaluation_seed'],r['anchor_id'],f"evaluation-{r['rep']}",'fine')
            cells.append(cell)
        assert seen=={(n,norm) for n in cfg['field_counts'] for norm in cfg['normalizations']}
    assert cells[0]['replay'] is not None
    return prep,frozen,cells,hashes,inputs


def main(root,out):
    prep,frozen,cells,hashes,inputs=verify(root);out.mkdir(parents=True,exist_ok=False)
    parent={(r['anchor_id'],r['ratio'],r['rep']):r for r in frozen['parent384'] if r['kind']=='noisy'}
    records=[]
    for cell in cells:
        record={k:cell[k] for k in ('n','normalization','replica','replay','elapsed_seconds')}
        record['curve']=[dict(update=point['update'],groups={name:summarize(point['rows'],aa,parent) for name,aa in groups(prep,cell['n']).items()}) for point in cell['curve']]
        h=cell['history'];record['loss_blocks']=[float(np.mean([r['loss'] for r in h[i:i+512]])) for i in range(0,3072,512)]
        record['clipped_fraction']=float(np.mean([r['gradient_norm']>1 for r in h]))
        record['presentations']={a:sum(r['anchor_id']==a for r in h) for a in groups(prep,cell['n'])['exposed']}
        record['associations']=associations(cell['curve'][-1]['rows'],prep,cell['n']);records.append(record)
    frozen_summary={name:{group:summarize(rows,aa,parent) for group,aa in groups(prep,3).items()} for name,rows in frozen['results'].items()}
    summary=dict(cells=records,frozen=frozen_summary,normalizations=prep['normalizations'],metadata=prep['metadata'],
        roundtrip_max_abs=max(r['max_abs'] for r in frozen['roundtrip']),checkpoints=hashes,inputs=inputs,
        counts=dict(fits=16,updates=49152,checkpoints=32,matrix_probes=14256,frozen_probes=1782,roundtrip_comparisons=135),
        source_sha256=source_hashes(),report_source_sha256=p.sha256(__file__),heldout_payloads_read=False,training_ready=False,
        caveats=['Three phases only; transfer regions and wide contexts can be correlated.',
          'Strict scaler uses fixed largest training pool even for tiny fit; no transfer moments.',
          'Finite nominal noise clean identity is diagnostic, not required Bayes identity.',
          'Two seeds describe run variability, not independent-cosmology uncertainty.',
          'Fixed updates reduce exposures per field as diversity grows; no convergence guarantee.',
          'Direct frozen scaler swaps change the predictor; only compensated roundtrip is function-equivalent.'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    measures=[('clean_rms','Clean-input RMS (physical units)'),('noise_left','High-k injected noise amplitude left'),('error_vs_parent','Noisy high-k error / 384 parent')]
    for row,norm in enumerate(prep['config']['normalizations']):
        for col,(metric,label) in enumerate(measures):
            ax=axes[row,col]
            for group,color in [('common_fit','C0'),('transfer','C1')]:
                for seed in (0,1):
                    rr=sorted([r for r in records if r['normalization']==norm and r['replica']==seed],key=lambda r:r['n'])
                    ax.plot([r['n'] for r in rr],[r['curve'][-1]['groups'][group]['0.05'][metric] for r in rr],
                            'o-' if seed==0 else 's--',color=color,label=f'{group}, seed {seed}',alpha=.8)
            if metric in ('noise_left','error_vs_parent'):ax.axhline(.2 if metric=='noise_left' else .25,color='gray',ls=':')
            ax.set(xlabel='Distinct fitting regions',ylabel=label,title=norm,xticks=[3,6,9,15]);ax.grid(alpha=.2)
            if row==0 and col==0:ax.legend(fontsize=8)
    fig.suptitle('Matched 3,072-update learning curves; nominal noise ratio 0.05');fig.savefig(out/'learning_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,norm in zip(axes,prep['config']['normalizations']):
        for i,n in enumerate(prep['config']['field_counts']):
            rr=[r for r in records if r['n']==n and r['normalization']==norm];ratios=prep['config']['clean_ratios'][1:]
            y=[med([r['curve'][-1]['groups']['transfer'][str(ratio)]['clean_rms'] for r in rr]) for ratio in ratios]
            ax.loglog(ratios,y,'o-',label=f'{n} fields',color=f'C{i}')
        ax.set(xlabel='Nominal noise ratio (no noise injected)',ylabel='Transfer clean RMS',title=norm);ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'clean_limit.png',dpi=150);plt.close(fig)
    lines=['# Diversity / normalization results','','All rows: median over the same 12 transfer regions and two fitted models; no confidence intervals.','',
           '| Fields | Normalization | Clean RMS at .05 | .05 noise left | .05 error / parent | .2 noise left |',
           '|---:|---|---:|---:|---:|---:|']
    for n in prep['config']['field_counts']:
        for norm in prep['config']['normalizations']:
            rr=[r['curve'][-1]['groups']['transfer'] for r in records if r['n']==n and r['normalization']==norm]
            values=[med([r['0.05'][m] for r in rr]) for m in ('clean_rms','noise_left','error_vs_parent')]+[med([r['0.2']['noise_left'] for r in rr])]
            lines.append(f'| {n} | {norm} | '+' | '.join(f'{v:.6g}' for v in values)+' |')
    lines+=['','## Frozen model: same weights, direct scaler interventions','',
        '| Scaler | Transfer clean RMS .05 | Transfer noise left .05 | Transfer error / parent .05 |',
        '|---|---:|---:|---:|']
    for name,g in frozen_summary.items():
        r=g['transfer']['0.05'];lines.append(f"| {name} | {r['clean_rms']:.6g} | {r['noise_left']:.6g} | {r['error_vs_parent']:.6g} |")
    lines+=['',f"Compensated frozen coordinate roundtrip: maximum absolute normalized difference {summary['roundtrip_max_abs']:.6g}.",
        '', '## Caveats','']+['- '+x for x in summary['caveats']]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    p.write_json(out/'SUMMARY.json',summary);print('\n'.join(lines),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();main(args.root,args.output)
