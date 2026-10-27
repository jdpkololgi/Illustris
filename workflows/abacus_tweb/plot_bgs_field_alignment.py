#!/usr/bin/env python3
"""Reproducible BGS coordinate/field registration figures from the five-phase audit.

This is not a galaxy-bias, clustering-amplitude or model-performance test.
No fitted transform, new native-object search, FFT, or catalogue mutation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import numpy as np
from galaxy_position_mapping import field_indices, periodic_delta
from validate_galaxy_position_mapping import Field, ALLOWED
from tweb_grid_sampling import mapped_member

SAMPLES = Path('/pscratch/sd/d/dkololgi/abacus/galaxy_position_validation_20261019/qualified_job60003795')
DENSITY = Path('/pscratch/sd/d/dkololgi/AbacusSummit_densities/density_fields/AbacusSummit_base_c000_ph000_z0.200_ngrid_2048_10pc_density_field.npy')
BINS = np.array([0., .05, .10, .15, .20, .30, .40, .50, .60])
COLORS = ['#007f8b', '#d17a22', '#7352a2', '#4c8a4d', '#c54e66']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--render-only', action='store_true', help='Reuse saved paired samples; bounded layout correction only')
    args=ap.parse_args()
    if not args.render_only and not(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):
        ap.error('An approved compute step is required')
    args.output.mkdir(parents=True, exist_ok=args.render_only)
    prior=json.loads((args.output/'manifest.json').read_text()) if args.render_only else None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    records=[]; sources=[]; phase_stats={}; ph0=None; grid0=None
    for phase in ([] if args.render_only else ALLOWED):
        path=SAMPLES/f'{phase}_sample.npz'
        with np.load(path) as f: s={k:f[k] for k in f.files}
        in_range=(s['Z_COSMO']>=0)&(s['Z_COSMO']<.6)
        covered=np.isfinite(s['independent_native']).all(axis=1)&in_range
        grid=Field(phase)
        p=s['reconstructed_native'][covered]; q=s['independent_native'][covered]
        a=grid.sample(p); b=grid.sample(q)
        z=s['Z_COSMO'][covered]
        error=np.linalg.norm(periodic_delta(p,q),axis=1)
        same=np.all(field_indices(p)==field_indices(q),axis=1)
        row=dict(z=z,a=a,b=b,error=error,same=same,phase=np.full(len(z),phase),
                 source_row=s['original_row'][covered],reference=s['reference_kind'][covered])
        records.append(row)
        phase_stats[phase]=dict(total_sample_rows=int(in_range.sum()),independently_checked=int(covered.sum()),
            z_min=float(z.min()),z_max=float(z.max()),max_position_error_mpc_h=float(error.max()),
            trace_pearson=corr(a.sum(1),b.sum(1)),same_voxel_fraction=float(same.mean()),
            rms_eigenvalue_error=float(np.sqrt(np.mean((a-b)**2))),
            changed_web_classes=int(np.sum((a>.2).sum(1)!=(b>.2).sum(1))))
        sources.append(dict(path=str(path),sha256=digest(path),field=str(grid.root)))
        if phase=='ph000': ph0=s;grid0=grid
        else: grid.cache.clear()
        print('SAMPLED',phase,phase_stats[phase],flush=True)
    if args.render_only:
        with np.load(args.output/'paired_field_samples.npz') as f: joined={k:f[k] for k in f.files}
        records=[{k:v[joined['phase']==phase] for k,v in joined.items()} for phase in ALLOWED]
        sources=prior['sources'];phase_stats=prior['phases']
        with np.load(SAMPLES/'ph000_sample.npz') as f: ph0={k:f[k] for k in f.files}
        grid0=Field('ph000')
    else:
        joined={k:np.concatenate([r[k] for r in records]) for k in records[0]}
        np.savez_compressed(args.output/'paired_field_samples.npz',**joined)
    z=joined['z']; a=joined['a']; b=joined['b']; error=joined['error']
    metrics=[]
    for lo,hi in zip(BINS[:-1],BINS[1:]):
        m=(z>=lo)&(z<hi)
        metrics.append(dict(z_low=float(lo),z_high=float(hi),n=int(m.sum()),
            trace_pearson=corr(a[m].sum(1),b[m].sum(1)),
            eigenvalue_pearson=[corr(a[m,i],b[m,i]) for i in range(3)],
            error_p99_mpc_h=float(np.percentile(error[m],99)),max_error_mpc_h=float(error[m].max()),
            same_voxel_fraction=float(joined['same'][m].mean())))
    # Figure 1: redshift-wide quantitative agreement, with explicit sample counts.
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    centers=(BINS[:-1]+BINS[1:])/2
    for phase,color,row in zip(ALLOWED,COLORS,records):
        vals=[];errs=[]
        for lo,hi in zip(BINS[:-1],BINS[1:]):
            m=(row['z']>=lo)&(row['z']<hi)
            vals.append(corr(row['a'][m].sum(1),row['b'][m].sum(1)))
            errs.append(np.percentile(row['error'][m],99) if m.any() else np.nan)
        axs[0,0].plot(centers,vals,'o-',ms=4,color=color,label=phase)
        axs[0,1].plot(centers,errs,'o-',ms=4,color=color)
    rmin=min(v['trace_pearson'] for v in metrics)
    axs[0,0].set(ylabel='Pearson r: tidal trace',ylim=(min(.9999,rmin-1e-5),1.000005),title='Independent positions give the same field')
    axs[0,0].ticklabel_format(axis='y',style='plain',useOffset=False)
    axs[0,0].legend(ncol=3,fontsize=9)
    axs[0,1].axhline(.005,ls='--',color='#888888',label='Predeclared 0.005 Mpc/h tolerance')
    axs[0,1].set(yscale='log',ylabel='99th percentile position error [Mpc/h]',title='Residuals remain far below a 0.977 Mpc/h cell')
    axs[0,1].legend(fontsize=8)
    axs[1,0].bar(centers,[100*v['same_voxel_fraction'] for v in metrics],width=np.diff(BINS)*.8,color='#007f8b')
    axs[1,0].set(ylabel='Identical grid cell [%]',ylim=(0,104),title='Nearest-cell agreement across five phases')
    for x,v in zip(centers,metrics): axs[1,0].text(x,96,f"{100*v['same_voxel_fraction']:.2f}",ha='center',va='top',rotation=90,fontsize=8,color='white')
    axs[1,1].bar(centers,[v['n'] for v in metrics],width=np.diff(BINS)*.8,color='#7352a2')
    axs[1,1].set(ylabel='Independently checked galaxies',yscale='log',title='Coverage is explicit; bins have unequal widths')
    for ax in axs.ravel():
        ax.axvline(.15,color='#555555',ls=':',lw=1);ax.set(xlim=(0,.6),xlabel='Cosmological redshift');ax.grid(alpha=.12)
    fig.suptitle('BGS mock → native matter field: agreement across 0 < z < 0.6',fontsize=16)
    fig.get_layout_engine().set(rect=(0,.065,1,.87))
    fig.text(.04,.012,'Five development phases; fixed z = 0.2 simulation field, R = 7 Mpc/h. Dotted line: catalogue branch at z = 0.15.\nLow-z references: sampled native centrals; high-z references: original cubic galaxies, including satellites.',fontsize=10)
    fig.savefig(args.output/'01_redshift_alignment.png',dpi=180);fig.savefig(args.output/'01_redshift_alignment.pdf');plt.close(fig)
    # Figure 2: directly paired tidal trace and eigenvalues, no subsampling of metrics.
    fig,axs=plt.subplots(2,4,figsize=(15,7.6),layout='constrained')
    for rr,(lo,hi) in enumerate([(0,.15),(.15,.6)]):
        m=(z>=lo)&(z<hi)
        for cc in range(4):
            x=b[m].sum(1) if cc==0 else b[m,cc-1]
            y=a[m].sum(1) if cc==0 else a[m,cc-1]
            ax=axs[rr,cc];hb=ax.hexbin(x,y,gridsize=65,mincnt=1,norm=LogNorm(vmin=1,vmax=int(m.sum())),cmap='viridis')
            lim=[min(x.min(),y.min()),max(x.max(),y.max())];ax.plot(lim,lim,color='#d54a3a',ls='--',lw=.8)
            r=corr(x,y);rms=float(np.sqrt(np.mean((x-y)**2)))
            ax.set(xlabel='Field at independent reference',ylabel='Field at RA/Dec/Z_COSMO position',
                   title=('Tidal trace' if cc==0 else f'λ{cc}')+f' | {lo:g} ≤ z < {hi:g}')
            ax.text(.04,.96,f'N = {len(x):,}\nr = {r:.8f}\nRMS Δ = {rms:.2g}',transform=ax.transAxes,va='top',fontsize=9,
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        fig.colorbar(hb,ax=axs[rr,:],label='Paired galaxies per hexagon',shrink=.7)
    fig.suptitle('Tidal-field agreement: independent native positions versus sky reconstruction',fontsize=15)
    fig.get_layout_engine().set(rect=(0,.06,1,.88))
    fig.text(.04,.012,'Trace = λ₁ + λ₂ + λ₃ (the density-related tidal scalar). These correlations test registration, not galaxy bias or predictive skill.\nAll independent pairs in range are shown; exact satellite XYZ checks are available only in the high-z branch.',fontsize=10)
    fig.savefig(args.output/'02_tidal_correlations.png',dpi=180);fig.savefig(args.output/'02_tidal_correlations.pdf');plt.close(fig)
    # Figure 3: deterministic representative low/mid/high-z locations in actual fields.
    # Choose central nearest each requested redshift, not the densest/best-looking location.
    density=np.load(DENSITY,mmap_mode='r');half=56;dx=2000/2048
    patches=[];examples=[]
    valid=np.isfinite(ph0['independent_native']).all(1)&(ph0['CEN']==1)
    for target in [.08,.25,.50]:
        ids=np.flatnonzero(valid);j=ids[np.argmin(abs(ph0['Z_COSMO'][ids]-target))]
        p=ph0['reconstructed_native'][j];q=ph0['independent_native'][j];ix,iy,iz=field_indices(p[None])[0]
        ys=(np.arange(iy-half,iy+half+1)%2048).astype(int);zs=(np.arange(iz-half,iz+half+1)%2048).astype(int)
        raw=sum(np.asarray(density[xx][np.ix_(ys,zs)],dtype=float) for xx in (np.arange(ix-7,ix+8)%2048))/15
        for low,high,path in grid0.spans:
            if low<=ix<high:
                arr=mapped_member(path,'eig_vals');trace=sum(arr[k,ix-low][np.ix_(ys,zs)] for k in range(3));break
        gridcenter=np.array([ix,iy,iz])*dx
        pg=(p%2000-gridcenter+1000)%2000-1000;qg=(q%2000-gridcenter+1000)%2000-1000
        patches.append((raw,trace,pg,qg))
        examples.append(dict(source_row=int(ph0['original_row'][j]),z=float(ph0['Z_COSMO'][j]),reference=str(ph0['reference_kind'][j]),
            position_error_mpc_h=float(np.linalg.norm(periodic_delta(p,q))),grid_indices=[int(ix),int(iy),int(iz)]))
    fig,axs=plt.subplots(2,3,figsize=(13,8.8),layout='constrained')
    extent=np.array([-half-.5,half+.5,-half-.5,half+.5])*dx
    for row in range(2):
        data=[p[row] for p in patches]
        # Raw density is stored in simulation units: show its positive log contrast
        # without claiming an unverified mean-density normalization.
        if row==0:
            positive=np.concatenate([v[v>0] for v in data]);floor=float(np.percentile(positive,1))
            data=[np.log10(np.maximum(v,floor)) for v in data]
        lim=np.percentile(np.concatenate([v.ravel() for v in data]),[1,99]);norm=Normalize(*lim)
        for col,(v,patch,example) in enumerate(zip(data,patches,examples)):
            ax=axs[row,col];im=ax.imshow(v.T,origin='lower',extent=extent,cmap='magma',norm=norm)
            pg,qg=patch[2:];ax.plot(qg[1],qg[2],'o',mfc='none',mec='cyan',ms=13,mew=1.5,label='Independent position')
            ax.plot(pg[1],pg[2],'+',color='white',ms=11,mew=1.5,label='Reconstructed position')
            ax.set(xlabel='Native y offset [Mpc/h]',ylabel='Native z offset [Mpc/h]',aspect='equal')
            if row==0:ax.set_title(f"ph000: z = {example['z']:.4f}\nposition difference {example['position_error_mpc_h']:.2g} Mpc/h",fontsize=11)
        fig.colorbar(im,ax=axs[row,:],shrink=.8,label='log₁₀ stored density; 14.65 Mpc/h depth' if row==0 else 'Tidal trace (R = 7 Mpc/h)')
    axs[1,0].legend(loc='lower left',fontsize=8,facecolor='#555555',labelcolor='white')
    fig.suptitle('The sky position lands on the independently identified galaxy in the native field',fontsize=15)
    fig.get_layout_engine().set(rect=(0,.055,1,.87))
    fig.text(.04,.012,'Three centrals nearest z = 0.08, 0.25, 0.50; no density-based selection. Circle and cross overlap at this scale.\nThese are patches of the same z = 0.2 simulation snapshot, not independently evolved light-cone density slices.',fontsize=10)
    fig.savefig(args.output/'03_native_field_registration.png',dpi=180);fig.savefig(args.output/'03_native_field_registration.pdf');plt.close(fig)
    report=dict(completed=True,job_id=prior['job_id'] if prior else os.environ['SLURM_JOB_ID'],node=prior['node'] if prior else socket.gethostname(),script_sha256=digest(__file__),
        sources=sources,raw_density=dict(path=str(DENSITY),bytes=DENSITY.stat().st_size,mtime_ns=DENSITY.stat().st_mtime_ns),
        phases=phase_stats,bins=metrics,examples=examples,
        total_pairs=len(z),trace_pearson=corr(a.sum(1),b.sum(1)),
        changed_web_classes=int(np.sum((a>.2).sum(1)!=(b>.2).sum(1))),
        limitations=['Coordinate/field registration only; not a galaxy-matter clustering or inference performance test.',
         'Equally spaced source-row audit sample, not a volume-limited or selection-weighted galaxy sample.',
         'Low-z independent references are sampled centrals in native slabs 0/33; low-z satellites lack exact XYZ references.',
         'Five phases only; no blind phases opened. Range 0<z<0.6, not the entire raw-catalogue redshift tail.',
         'Native reference positions are independent of the sky transform; both labels sample the same existing field.',
         'Single z=0.2 matter snapshot; no light-cone evolution validation. Raw density units left unrenormalized.'])
    if prior: report['sampling_script_sha256']=prior.get('sampling_script_sha256',prior['script_sha256'])
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',json.dumps({k:report[k] for k in ['total_pairs','trace_pearson','changed_web_classes']}),flush=True)

if __name__=='__main__':main()
