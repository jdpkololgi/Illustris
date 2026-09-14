"""Summarize completed training-panel diagnostics; plotting remains on compute."""
import argparse
import json
from pathlib import Path
import numpy as np


def band_power(delta,cell=3.383):
    """Matched Hann-windowed parent power; analysis only, not another smoothing."""
    x=np.asarray(delta,dtype=float)
    n=x.shape[0]
    w=np.hanning(n)
    window=w[:,None,None]*w[None,:,None]*w[None,None,:]
    centered=x-float(np.sum(x*window)/window.sum())
    power=np.abs(np.fft.rfftn(centered*window))**2
    k=np.fft.fftfreq(n,d=cell)*2*np.pi
    kz=np.fft.rfftfreq(n,d=cell)*2*np.pi
    radius=np.sqrt(k[:,None,None]**2+k[None,:,None]**2+kz[None,None,:]**2)
    weights=np.full(kz.shape,2.); weights[0]=1
    if n%2==0:
        weights[-1]=1
    power*=weights
    edges=[0,.08,.16,.32,np.inf]
    return np.array([power[(radius>lo)&(radius<=hi)].sum() for lo,hi in zip(edges[:-1],edges[1:])])


def summarize(report, root):
    losses=report['fixed_loss_probes']
    steps=report['registration']['loss_checkpoints']
    means_key='means_'+'_'.join(map(str,steps))
    optimization={}
    for method in ('cfm','diffusion'):
        for stage in ('coarse','fine'):
            rows=[r for r in losses if r['method']==method and r['stage']==stage]
            x=np.array([[r['losses'][str(s)] for s in steps] for r in rows])
            means=x.mean(axis=0)
            optimization[f'{method}/{stage}']={means_key:means.tolist(),
                'checkpoint_steps':steps,
                'late_relative_improvement':float(1-means[2]/means[1]),
                'fraction_anchors_improving_late':float(np.mean(x[:,2]<x[:,1])),
                'by_phase':{ph:np.mean([[r['losses'][str(s)] for s in steps] for r in rows if r['phase']==ph],axis=0).tolist()
                            for ph in ('ph000','ph002','ph003')}}
    refinement={}
    for method in ('cfm','diffusion'):
        values=[r for r in report['sampler_refinement'] if r['method']==method]
        refinements=[]
        for r in values:
            ensemble=json.loads((root/f'{r["anchor_id"]}_{method}.json').read_text())['ensemble']['observed']
            rms=np.array(r['masks']['observed']['eigen_rmse'])
            spread=np.array(ensemble['mean_pointwise_draw_std'])
            refinements.append(rms/np.maximum(spread,1e-30))
        refinement[method]={'median_eigen_rms_change_over_draw_std':np.median(refinements,axis=0).tolist(),
            'max_eigen_rms_change_over_draw_std':np.max(refinements,axis=0).tolist()}
    return {'optimization':optimization,'sampler_refinement':refinement,
            'science':report['summary'],'by_phase':report['by_phase'],
            'by_shell_support':report['by_shell_support'],
            'scope':'training panel; no held-out calibration; four draws per anchor'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.root/'EVALUATION_COMPLETE.json').read_text())
    from workflows.sbi import e2e_wide_pipeline as p
    device=p.runtime()
    result=summarize(report,args.root)
    import h5py
    from workflows.sbi.e2e_wide_research_canary import preflight,NORMALIZATION
    from workflows.sbi.e2e_wide_evaluate import TRAIN_ROOT
    c,ds,_,_=preflight()
    ds=p.dataset_for(c,NORMALIZATION)
    binding=p.provenance(c,ds)
    steps=report['registration']['loss_checkpoints']
    earlier_step,current_step=steps[-2:]
    train_root=Path(report['registration'].get('train_root',TRAIN_ROOT))
    continuation_root=report['registration'].get('continuation_root')
    if continuation_root:
        from workflows.sbi.e2e_wide_continue import checked_binding
        binding=checked_binding(continuation_root,binding)
    if p.digest(binding)!=report['registration']['binding_sha256']:
        raise ValueError('report checkpoint binding differs from evaluation')
    older={}
    for method in ('cfm','diffusion'):
        for stage in ('coarse','fine'):
            state=p.load_checkpoint(train_root/f'{method}_{stage}'/f'step_{earlier_step:06d}.pt',binding,stage,method)
            older[method,stage]=p.build_model(c,stage,device).eval()
            older[method,stage].load_state_dict(state['model'])
    cfg=json.loads((p.REPO/c['diagnostic_config']).read_text())
    def functional_delta(a,b,mask):
        sa,sb=[p.science(p.eigs(t),mask,c['fine_cell_mpc_h'],cfg) for t in (a,b)]
        return {'filling_abs_change':abs(sa['filling_fraction']-sb['filling_fraction']),
                'largest_void_abs_change':abs(sa['largest_void_fraction']-sb['largest_void_fraction']),
                'connection_changed':sa['connections_xyz']!=sb['connections_xyz'],
                'pair_abs_change':[abs(x['value']-y['value']) for x,y in zip(sa['pair'],sb['pair'])]}
    power_rows=[]; drift=[]; sampler_functionals=[]
    for index,row in enumerate(ds.rows):
        with h5py.File(row['shard'],'r') as f:
            truth=f[row['group']]['delta_r7_gaussian'][:]
            mask=f[row['group']]['masks/observed_parent'][32:64,32:64,32:64].astype(bool)
        reference=band_power(truth)
        for method in ('cfm','diffusion'):
            path=args.root/f'{row["anchor_id"]}_{method}.h5'
            record=json.loads(path.with_suffix('.json').read_text())
            if p.sha256(path)!=record['sample_sha256']:
                raise ValueError('draw file checksum changed')
            with h5py.File(path,'r') as f:
                powers=[band_power(f[str(i)]['delta_local96'][:]) for i in range(4)]
                current=f['0/tensor_core'][:]
                refined=f['refined_0/tensor_core'][:] if 'refined_0' in f else None
            power_rows.append({'anchor_id':row['anchor_id'],'phase':row['phase'],'method':method,
                               'truth_min_delta':float(truth.min()),
                               'truth_below_minus_one_fraction':float(np.mean(truth < -1)),
                               'truth_observed_below_minus_one_fraction':float(np.mean(truth[32:64,32:64,32:64][mask] < -1)),
                               'truth_band_power':reference.tolist(),
                               'mean_draw_band_power':np.mean(powers,axis=0).tolist(),
                               'truth_high_k_power_fraction':float(reference[-1]/reference.sum()),
                               'draw_high_k_power_fraction':float(np.mean(powers,axis=0)[-1]/np.mean(powers,axis=0).sum()),
                               'draw_power_over_truth':(np.mean(powers,axis=0)/np.maximum(reference,1e-30)).tolist()})
            if row['anchor_id'] in report['registration']['refinement_anchors']:
                sampler_functionals.append({'anchor_id':row['anchor_id'],'method':method,
                    'observed':functional_delta(refined,current,mask),
                    'complete':functional_delta(refined,current,np.ones_like(mask))})
                obs=ds.inference_conditions(index)
                co,fi,seeds=p.generate_pair(c,ds,obs,older[method,'coarse'],older[method,'fine'],method,'eval-0',device)
                earlier=p.reconstruct(co,fi)
                target=args.root/f'{row["anchor_id"]}_{method}_checkpoint{earlier_step}.h5'
                with h5py.File(target,'x') as f:
                    f.attrs['checkpoint_step']=earlier_step
                    f.attrs['paired_sample_id']='eval-0'
                    for name,value in {'coarse_delta':co,'fine_residual':fi,**earlier}.items():
                        f.create_dataset(name,data=value)
                metrics=p.tensor_metrics(earlier['tensor_core'],current,mask)
                spread=np.array(record['ensemble']['observed']['mean_pointwise_draw_std'])
                drift.append({'anchor_id':row['anchor_id'],'method':method,'metrics':metrics,
                    'observed_functionals':functional_delta(earlier['tensor_core'],current,mask),
                    'complete_functionals':functional_delta(earlier['tensor_core'],current,np.ones_like(mask)),
                    'eigen_rms_change_over_draw_std':(np.array(metrics['eigen_rmse'])/np.maximum(spread,1e-30)).tolist(),
                    'sample_sha256':p.sha256(target)})
        print(f'POWER/DRIFT CHECK {index+1}/96',flush=True)
    result['parent_density_power']={'bands_h_mpc':['(0,.08]','(.08,.16]','(.16,.32]','>.32'],
        'window':'same Hann taper and weighted demeaning on full 96-cubed local parents; no mask',
        'rows':power_rows,'median_ratio':{m:np.median([r['draw_power_over_truth'] for r in power_rows if r['method']==m],axis=0).tolist()
                                         for m in ('cfm','diffusion')}}
    result['late_checkpoint_draw_drift']={'earlier':earlier_step,'current':current_step,'paired_seed':True,'rows':drift,
        'median_eigen_rms_change_over_draw_std':{m:np.median([r['eigen_rms_change_over_draw_std'] for r in drift if r['method']==m],axis=0).tolist()
                                                 for m in ('cfm','diffusion')}}
    result['sampler_functional_drift']=sampler_functionals
    result['report_source_sha256']=p.sha256(__file__)
    preflight()
    if continuation_root and checked_binding(continuation_root,p.provenance(c,ds)) != binding:
        raise ValueError('continuation binding changed during report')
    p.write_json(args.root/'INTERPRETATION_SUMMARY.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(name,r) in zip(axes.flat,result['optimization'].items()):
        for phase,values in r['by_phase'].items():
            ax.plot(steps,values,'o-',label=phase,alpha=.65)
        ax.plot(steps,r['means_'+'_'.join(map(str,steps))],'ko-',label='All training phases',lw=2)
        ax.set(title=name,xlabel='Training update',ylabel='Fixed-noise objective loss',xticks=steps)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Optimization probes on training data — not held-out calibration')
    fig.savefig(args.root/'fixed_noise_convergence.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for method in ('cfm','diffusion'):
        ax.plot(np.arange(4),result['parent_density_power']['median_ratio'][method],'o-',label=method)
    ax.axhline(1,color='k',ls='--',label='Matched target power')
    ax.set(yscale='log',xticks=np.arange(4),xticklabels=result['parent_density_power']['bands_h_mpc'],
           xlabel='Wavenumber band [h/Mpc]',ylabel='Median draw power / target power',
           title='Full local-parent density spectra — matched window, training panel')
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(args.root/'density_power_ratio.png',dpi=160)
    plt.close(fig)
    print(json.dumps(result['optimization'],indent=2))
    print(json.dumps(result['sampler_refinement'],indent=2))


if __name__=='__main__':
    main()
