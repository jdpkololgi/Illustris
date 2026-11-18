"""Actual P12 encoder-input example and all-visible-phase redshift coverage."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute,sha256
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
from workflows.abacus_tweb.p8_train_unet_patch import model_inputs

ROOT=Path('/pscratch/sd/d/dkololgi/abacus/p12a_lowz_20261029_v1')
REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'docs/figures/p12a_training_examples_20261030'

def run():
    require_compute();OUT.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    edges=np.linspace(.01,.55,28);fine_edges=np.array([.01,.03,.05,.075,.1,.15,.25,.35,.45,.55])
    rows=[];bindings={};example=None
    with P10PhaseBalancedLoader(ROOT/'training_contract_v1') as loader:
        for phase in loader.training_phases+loader.validation_phases:
            record=loader.phase_records[phase]
            assignment_path=Path(record['inputs']['assignment'])
            with np.load(assignment_path) as a:
                eligible=a['supervised_eligible'].astype(bool)
                parent=a['parent_node_id'][eligible]
                core_ids=a['core_id'][eligible]
            z=np.load(loader.root/'phases'/phase/'parent_redshift.npy',mmap_mode='r')[parent]
            if (not np.isfinite(z).all() or np.any(z<.01) or np.any(z>.55)
                    or len(z)!=record['authoritative_rows']):raise ValueError('Redshift/row coverage mismatch '+phase)
            hist=np.histogram(z,edges)[0];fine=np.histogram(z,fine_edges)[0]
            if hist.sum()!=len(z) or np.any(fine==0):raise ValueError('Missing redshift coverage '+phase)
            rows.append(dict(phase=phase,role='training' if phase in loader.training_phases else 'development',
                             count=len(z),zmin=float(z.min()),zmax=float(z.max()),counts=hist.tolist(),diagnostic_counts=fine.tolist()))
            bindings[str(assignment_path)]=sha256(assignment_path)
            bindings[str(loader.root/'phases'/phase/'parent_redshift.npy')]=sha256(loader.root/'phases'/phase/'parent_redshift.npy')
            # Fixed first training phase; choose most populated core in .15-.25,
            # using positions/counts only, never target values.
            if example is None and phase==loader.training_phases[0]:
                bins=np.bincount(core_ids[(z>=.15)&(z<.25)])
                example=(phase,int(np.argmax(bins)))
            print('COVERAGE',phase,len(z),z.min(),z.max(),flush=True)
        fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
        for role,color,style in [('training','#2166ac','-'),('development','#b35806','--')]:
            group=[r for r in rows if r['role']==role]
            totals=np.sum([r['counts'] for r in group],axis=0)
            axes[0].stairs(totals/1e6,edges,color=color,linestyle=style,lw=2,label=f'{role.capitalize()} ({len(group)} phases)')
            axes[1].stairs(totals/totals.sum()/np.diff(edges),edges,color=color,linestyle=style,lw=2,label=role.capitalize())
        axes[0].set(ylabel='Eligible galaxies per bin [millions]',xlabel='Observed redshift z',xlim=(.01,.55),ylim=(0,None))
        axes[1].set(ylabel='Normalized redshift density',xlabel='Observed redshift z',xlim=(.01,.55),ylim=(0,None))
        axes[0].legend();axes[1].legend()
        for ax in axes:
            ax.set_xticks([.01,.1,.2,.3,.4,.55]);ax.grid(axis='y',alpha=.18)
        fig.suptitle('P12-A encoder data: eligible galaxies span z = 0.01–0.55')
        fig.savefig(OUT/'redshift_coverage.png',dpi=180);fig.savefig(OUT/'redshift_coverage.pdf');plt.close(fig)

        phase,core_id=example;adapter=loader.field_adapter(phase)
        patch=adapter.extract(core_id,context_halo_voxels=48)
        inputs,coords=model_inputs(patch,loader.field_normalization,'cpu')
        values=inputs.numpy()[0]
        if not np.isfinite(values).all():raise ValueError('Nonfinite actual model input')
        parents=patch.authoritative_parent_id
        frac=patch.authoritative_frac_index_local
        # Choose middle occupied z slice geometrically, then show a three-cell slab.
        center=int(np.floor(np.median(frac[:,2])+.5));lo=max(0,center-1);hi=min(values.shape[3],center+2)
        select=(frac[:,2]>=lo-.5)&(frac[:,2]<hi-.5)
        if select.sum()<5:raise ValueError('Example slab has too few galaxies')
        cell=float(adapter.manifest['caps']['SGC' if patch.cap==0 else 'NGC']['cell_mpc'])
        xy=(frac[select,:2]+.5)*cell
        target=np.asarray(loader.targets_by_parent(phase)[parents[select]],float)
        if np.any(np.diff(target,axis=1)<0):raise ValueError('Unordered targets')
        z=np.load(loader.root/'phases'/phase/'parent_redshift.npy',mmap_mode='r')[parents[select]]
        fig,axes=plt.subplots(2,3,figsize=(14,8.4),layout='constrained')
        extent=[0,values.shape[1]*cell,0,values.shape[2]*cell]
        titles=['Standardized log(1 + CIC count)','Selection-corrected density proxy','Survey exposure']
        for i,ax in enumerate(axes[0]):
            im=ax.imshow(values[i,:,:,lo:hi].mean(axis=2).T,origin='lower',extent=extent,cmap='viridis',aspect='equal')
            ax.scatter(xy[:,0],xy[:,1],s=6,facecolors='none',edgecolors='white',linewidths=.35)
            ax.set_title(titles[i]);fig.colorbar(im,ax=ax,shrink=.8)
        lower,upper=np.quantile(target,[.01,.99])
        for i,ax in enumerate(axes[1]):
            im=ax.scatter(xy[:,0],xy[:,1],c=target[:,i],s=10,cmap='coolwarm',vmin=lower,vmax=upper)
            ax.set(xlim=extent[:2],ylim=extent[2:]);ax.set_aspect('equal')
            ax.set_title(r'Galaxy target $\lambda_'+str(i+1)+'$');fig.colorbar(im,ax=ax,shrink=.8,extend='both')
        for ax in axes.flat:
            ax.set_xlabel('Patch x [Mpc]');ax.set_ylabel('Patch y [Mpc]')
            x0,y0=(patch.core_start[:2]-patch.context_start[:2])*cell
            dx,dy=(patch.core_stop[:2]-patch.core_start[:2])*cell
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((x0,y0),dx,dy,fill=False,edgecolor='black',ls='--',lw=1))
        fig.suptitle(f'P12-A actual input / galaxy targets — {phase}, core {core_id}\n'
                     f'{select.sum():,} core galaxies in a {cell*(hi-lo):.1f} Mpc slab; z = {z.min():.3f}–{z.max():.3f}')
        fig.savefig(OUT/'input_target_example.png',dpi=160);fig.savefig(OUT/'input_target_example.pdf');plt.close(fig)
        np.savez_compressed(OUT/'example_arrays.npz',model_input_slab=values[:,:,:,lo:hi].mean(axis=3),
                            xy_mpc=xy,parent_id=parents[select],eigenvalues=target,redshift=z)
        result=dict(pass_=True,scope='eligible supervised galaxies in actual full training contract; no sealed phases',
                    phase_rows=rows,bin_edges=edges.tolist(),diagnostic_edges=fine_edges.tolist(),bindings=bindings,
                    example=dict(phase=phase,core_id=core_id,galaxies=int(select.sum()),input_shape=list(values.shape),
                                 selection='first training phase, most populated core in z .15-.25, central occupied slab; no target selection',
                                 target='ordered matter-tidal eigenvalues at galaxy positions, NOT a continuous target field',
                                 display='actual model channels averaged across slab; core galaxies only; dashed core box',
                                 target_color_limits=[float(lower),float(upper)]))
        result['pass']=result.pop('pass_');(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('FIGURES_COMPLETE',OUT,flush=True)

def zoom_from_saved():
    """Presentation-only view; do not re-read or rebuild scientific products."""
    require_compute()
    record=json.loads((OUT/'verification.json').read_text())
    with np.load(OUT/'example_arrays.npz') as data:
        values=data['model_input_slab'];xy=data['xy_mpc'];target=data['eigenvalues']
    # The exact geometric span comes from the saved shape and fixed P12 grid.
    phase=record['example']['phase'];core=record['example']['core_id']
    with P10PhaseBalancedLoader(ROOT/'training_contract_v1') as loader:
        adapter=loader.field_adapter(phase);cap=int(adapter.core_cap[core])
        grid=adapter.manifest['caps']['SGC' if cap==0 else 'NGC'];cell=float(grid['cell_mpc'])
        start=np.maximum(adapter.core_start[core]-48,0)
        x0,y0=(adapter.core_start[core][:2]-start[:2])*cell
        dx,dy=(adapter.core_stop[core][:2]-adapter.core_start[core][:2])*cell
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,3,figsize=(14,8.8),layout='constrained')
    extent=[0,values.shape[1]*cell,0,values.shape[2]*cell]
    names=['Standardized log(1 + CIC count)','Selection-corrected density proxy','Survey exposure']
    for i,ax in enumerate(axes[0]):
        im=ax.imshow(values[i].T,origin='lower',extent=extent,cmap='viridis',aspect='equal')
        ax.add_patch(Rectangle((x0,y0),dx,dy,fill=False,edgecolor='white',ls='--',lw=1.4))
        ax.set_title(names[i]);ax.set(xlabel='Context x [Mpc]',ylabel='Context y [Mpc]')
        fig.colorbar(im,ax=ax,shrink=.78)
    lower,upper=record['example']['target_color_limits']
    for i,ax in enumerate(axes[1]):
        im=ax.scatter(xy[:,0]-x0,xy[:,1]-y0,c=target[:,i],s=13,cmap='coolwarm',vmin=lower,vmax=upper)
        ax.set(xlim=(0,dx),ylim=(0,dy),xlabel='Core x [Mpc]',ylabel='Core y [Mpc]');ax.set_aspect('equal')
        ax.set_title(r'Core zoom: galaxy $\lambda_'+str(i+1)+'$ label')
        fig.colorbar(im,ax=ax,shrink=.78,extend='both')
    fig.suptitle(f'P12-A input / target example — {phase}, core {core}\n'
                 'Top: actual three-channel input, 15 Mpc slab mean; bottom: same-slab core galaxy labels')
    fig.savefig(OUT/'input_target_example_zoom.png',dpi=160)
    fig.savefig(OUT/'input_target_example_zoom.pdf');plt.close(fig)

if __name__=='__main__':
    import sys
    if '--zoom-only' in sys.argv:zoom_from_saved()
    else:run()
