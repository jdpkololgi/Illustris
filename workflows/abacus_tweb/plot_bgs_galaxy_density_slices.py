#!/usr/bin/env python3
"""Overlay every source galaxy in three matched native-density slabs.

Full ph000 source scan, no host indexing, no fit, no production data mutation.
Galaxy cuts use unwrapped observer coordinates to avoid selecting periodic images
at a different redshift. Native density reads wrap only at the simulation edge.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import fitsio
import numpy as np
from galaxy_position_mapping import comoving_distance, OBSERVER, field_indices

BASE=Path('/pscratch/sd/d/dkololgi/abacus/galaxy_position_validation_20261019/qualified_job60003795')
DENS=Path('/pscratch/sd/d/dkololgi/AbacusSummit_densities/density_fields/AbacusSummit_base_c000_ph000_z0.200_ngrid_2048_10pc_density_field.npy')
DX=2000/2048
HALF=56
DEPTH_CELLS=15

def sky_position(ra,dec,z):
    a,d=np.deg2rad(ra.astype(float)),np.deg2rad(dec.astype(float))
    unit=np.column_stack((np.cos(d)*np.cos(a),np.cos(d)*np.sin(a),np.sin(d)))
    return comoving_distance(z)[:,None]*unit+OBSERVER

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--render-only',action='store_true')
    args=ap.parse_args()
    if not args.render_only and not(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):ap.error('Approved compute step required')
    args.output.mkdir(parents=True,exist_ok=args.render_only)
    prior=json.loads((args.output/'manifest.json').read_text()) if args.render_only else None
    if args.render_only:
        source=Path(prior['source']);n=prior['source_rows'];examples=prior['examples'];galaxies=[];images=[]
        for k in range(3):
            with np.load(args.output/f'slice_{k}.npz') as f:
                images.append(f['density']);galaxies.append({key:f[key] for key in ['offset','source_row','z','r','cen','res']})
    else:
        audit=json.loads((BASE/'report.json').read_text());source=Path(audit['phases']['ph000']['sources'][0]['path'])
        with np.load(BASE/'ph000_sample.npz') as f:s={k:f[k] for k in f.files}
        valid=np.isfinite(s['independent_native']).all(1)&(s['CEN']==1)
        centers=[];indices=[];examples=[]
        for target in [.08,.25,.50]:
            ids=np.flatnonzero(valid);j=ids[np.argmin(abs(s['Z_COSMO'][ids]-target))]
            p=sky_position(s['RA'][j:j+1],s['DEC'][j:j+1],s['Z_COSMO'][j:j+1])[0]
            native=(p+1000)%2000-1000;q=field_indices(native[None])[0]
            # Centre each plotted voxel exactly; selection and image have identical edges.
            center=p+((q+.5)*DX-p%2000)
            assert np.max(np.abs(center-p))<=DX
            centers.append(center);indices.append(q)
            examples.append(dict(target_z=target,center_source_row=int(s['original_row'][j]),center_z=float(s['Z_COSMO'][j])))
        selected=[[] for _ in centers];columns=['RA','DEC','Z_COSMO','R_MAG_APP','CEN','RES']
        with fitsio.FITS(source) as f:
            n=f[1].get_nrows()
            for start in range(0,n,1000000):
                block=f[1].read(rows=np.arange(start,min(start+1000000,n)),columns=columns)
                good=np.isfinite(block['Z_COSMO'])&(block['Z_COSMO']>=0)&(block['Z_COSMO']<.65)
                rows=np.flatnonzero(good);b=block[good];pos=sky_position(b['RA'],b['DEC'],b['Z_COSMO'])
                for k,center in enumerate(centers):
                    delta=pos-center
                    m=(np.abs(delta[:,0])<DEPTH_CELLS*DX/2)&(np.abs(delta[:,1])<(HALF+.5)*DX)&(np.abs(delta[:,2])<(HALF+.5)*DX)
                    if m.any():selected[k].append(dict(offset=delta[m],source_row=start+rows[m],z=b['Z_COSMO'][m],r=b['R_MAG_APP'][m],cen=b['CEN'][m],res=b['RES'][m]))
                if start%10000000==0:print('SCANNED',min(start+1000000,n),'of',n,flush=True)
        galaxies=[];images=[];density=np.load(DENS,mmap_mode='r')
        for k,q in enumerate(indices):
            g={key:np.concatenate([part[key] for part in selected[k]]) for key in selected[k][0]}
            assert len(np.unique(g['source_row']))==len(g['source_row'])
            galaxies.append(g)
            ix,iy,iz=q;ys=np.arange(iy-HALF,iy+HALF+1)%2048;zs=np.arange(iz-HALF,iz+HALF+1)%2048
            img=sum(np.asarray(density[xx][np.ix_(ys,zs)],float) for xx in np.arange(ix-7,ix+8)%2048)/DEPTH_CELLS
            assert img.shape==(113,113) and np.isfinite(img).all()
            images.append(img)
            np.savez_compressed(args.output/f'slice_{k}.npz',density=img,center_unwrapped=centers[k],grid_center=q,**g)
            examples[k]['mean_density_mpc_h_minus3']=len(g['z'])/((113*DX)**2*DEPTH_CELLS*DX)
            examples[k]['expected_count_in_spherical_aperture']={str(r):examples[k]['mean_density_mpc_h_minus3']*4*np.pi*r**3/3 for r in [2,4,6,7]}
            examples[k].update(n_galaxies=len(g['z']),n_r_lt_19p5=int((g['r']<19.5).sum()),z_min=float(g['z'].min()),z_max=float(g['z'].max()))
            print('SLICE',k,examples[k],flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    positive=np.concatenate([a[a>0] for a in images]);floor=float(np.percentile(positive,1))
    logs=[np.log10(np.maximum(a,floor)) for a in images];lim=np.percentile(np.concatenate([a.ravel() for a in logs]),[1,99])
    norm=Normalize(*lim);extent=np.array([-HALF-.5,HALF+.5,-HALF-.5,HALF+.5])*DX
    fig,axs=plt.subplots(2,3,figsize=(14,10))
    fig.subplots_adjust(left=.065,right=.89,bottom=.16,top=.87,wspace=.22,hspace=.24)
    for row in range(2):
        for k,(g,img,e) in enumerate(zip(galaxies,logs,examples)):
            ax=axs[row,k];im=ax.imshow(img.T,extent=extent,origin='lower',cmap='magma',norm=norm)
            if row==1:ax.scatter(g['offset'][:,1],g['offset'][:,2],s=9,facecolors='none',edgecolors='#5afff0',linewidths=.6,alpha=.8,rasterized=True)
            ax.set(xlabel='Native y offset [Mpc/h]',ylabel='Native z offset [Mpc/h]',xlim=extent[:2],ylim=extent[2:])
            ax.set_title((f"z ≈ {e['center_z']:.2f}: matter only" if row==0 else f"Same slice + {e['n_galaxies']:,} mock galaxies"),fontsize=12)
    cax=fig.add_axes([.915,.24,.018,.50]);fig.colorbar(im,cax=cax,label='log₁₀ stored particle density (common scale)')
    fig.suptitle('Do the mock galaxies trace the matter web?\nMatched density projections and galaxy overlays',fontsize=18,y=.975)
    fig.text(.065,.055,f'Every source galaxy inside each {DEPTH_CELLS*DX:.2f} Mpc/h thick, {113*DX:.2f} × {113*DX:.2f} Mpc/h slab is plotted.\nFull ph000 flux-limited parent; no additional magnitude, footprint or fibre-assignment cut. Cyan circles: galaxy positions.',fontsize=11)
    fig.text(.065,.016,'Fixed z = 0.2 simulation snapshot; real-space Z_COSMO. Centres retained from the previous figure, not chosen for visual agreement.',fontsize=10)
    fig.savefig(args.output/'01_galaxies_on_density.png',dpi=180);fig.savefig(args.output/'01_galaxies_on_density.pdf');plt.close(fig)
    # A compact descriptive check: galaxy fraction in equal-area density quintiles.
    # Not a selection-corrected galaxy bias estimate or independent phase test.
    fig,axs=plt.subplots(1,3,figsize=(12,4.7));fig.subplots_adjust(left=.08,right=.98,bottom=.25,top=.77,wspace=.25)
    stats=[]
    for k,(ax,g,raw,e) in enumerate(zip(axs,galaxies,images,examples)):
        edges=np.quantile(raw,np.linspace(0,1,6));assert np.all(np.diff(edges)>0)
        q=np.floor((g['offset'][:,1:]/DX)+HALF+.5).astype(int);assert np.all((q>=0)&(q<113))
        atgal=raw[q[:,0],q[:,1]]
        area=np.histogram(raw,bins=edges)[0];count=np.histogram(atgal,bins=edges)[0]
        ratio=(count/count.sum())/(area/area.sum())
        ax.bar(np.arange(1,6),ratio,color='#187e84');ax.axhline(1,color='black',ls='--',lw=1)
        ax.set(xlabel='Projected matter-density quintile\n(low → high; equal map area)',ylabel='Galaxy fraction / area fraction',title=f"z ≈ {e['center_z']:.2f}; N = {len(atgal):,}",xticks=np.arange(1,6))
        stats.append(dict(density_edges=edges.tolist(),pixel_counts=area.tolist(),galaxy_counts=count.tolist(),relative_occupancy=ratio.tolist()))
    ymax=max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:ax.set_ylim(0,ymax)
    fig.suptitle('Are galaxies preferentially located in denser parts of these slices?',fontsize=14)
    fig.text(.08,.04,'Dashed line: uniform placement over map area. Descriptive comparison of these three flux-limited slabs;\nnot a selection-corrected bias estimate, significance test, or light-cone evolution measurement.',fontsize=10)
    fig.savefig(args.output/'02_density_preference.png',dpi=180);fig.savefig(args.output/'02_density_preference.pdf');plt.close(fig)
    report=dict(completed=True,job_id=prior['job_id'] if prior else os.environ['SLURM_JOB_ID'],node=prior['node'] if prior else socket.gethostname(),source=str(source),source_rows=n,
        source_bytes=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns,density=str(DENS),phase='ph000',snapshot_z=.2,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),slice_depth_mpc_h=DEPTH_CELLS*DX,width_mpc_h=113*DX,
        selection='All source galaxies in exact unwrapped slabs; no additional magnitude or angular footprint cut; Z_COSMO < 0.65 prefilter',
        examples=examples,density_preference=stats,limitations=['Single phase and three fixed local volumes.','Flux-limited source; tracer density varies with redshift.','Density projection and galaxies use identical thickness and bounds.','Matter field fixed at snapshot z=0.2; this is not evolution validation.','Uniform-area reference is descriptive, not a survey-selection model.'])
    if prior:report['sampling_script_sha256']=prior.get('sampling_script_sha256',prior['script_sha256'])
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
