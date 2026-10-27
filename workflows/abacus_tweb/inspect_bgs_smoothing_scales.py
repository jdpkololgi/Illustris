#!/usr/bin/env python3
"""Light local 3D density smoothing and tracer-support diagnostic; no T-Web rebuild."""
import json,os,socket
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from plot_bgs_galaxy_density_slices import DENS,DX,HALF,DEPTH_CELLS

def main():
    if not(os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):raise RuntimeError('Compute step required')
    root=Path('outputs/bgs_galaxy_density_slices_20261019');m=json.loads((root/'manifest.json').read_text())
    density=np.load(DENS,mmap_mode='r');scales=[2,4,7];pad=int(np.ceil(4*max(scales)/DX));maps=[]
    for k in range(3):
        with np.load(root/f'slice_{k}.npz') as f:q=f['grid_center']
        xx=np.arange(q[0]-7-pad,q[0]+8+pad)%2048
        yy=np.arange(q[1]-HALF-pad,q[1]+HALF+1+pad)%2048
        zz=np.arange(q[2]-HALF-pad,q[2]+HALF+1+pad)%2048
        cube=np.stack([np.asarray(density[x][np.ix_(yy,zz)],dtype=np.float64) for x in xx])
        row=[]
        for r in scales:
            smooth=gaussian_filter(cube,sigma=r/DX,truncate=4,mode='nearest')
            # All displayed cells are >=4 sigma from a block boundary; no local boundary artefact.
            row.append(smooth[pad:pad+15,pad:pad+113,pad:pad+113].mean(0))
        maps.append(row);print('SMOOTHED',k,flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    logs=np.log10(np.maximum(np.array(maps),np.finfo(float).tiny));vmin,vmax=np.percentile(logs,[1,99])
    fig,axs=plt.subplots(3,3,figsize=(12,12));fig.subplots_adjust(left=.08,right=.88,bottom=.09,top=.91,wspace=.16,hspace=.22)
    extent=np.array([-HALF-.5,HALF+.5,-HALF-.5,HALF+.5])*DX
    for col in range(3):
        for row,r in enumerate(scales):
            ax=axs[row,col];im=ax.imshow(logs[col,row].T,origin='lower',extent=extent,cmap='magma',norm=Normalize(vmin,vmax))
            ax.set_title(f"z ≈ {m['examples'][col]['center_z']:.2f}; R = {r} Mpc/h",fontsize=11)
            if row==2:ax.set_xlabel('y offset [Mpc/h]')
            if col==0:ax.set_ylabel('z offset [Mpc/h]')
    cax=fig.add_axes([.91,.26,.018,.50]);fig.colorbar(im,cax=cax,label='log₁₀ stored density, common scale')
    fig.suptitle('What smaller smoothing retains in the same matter slices',fontsize=16)
    fig.text(.08,.025,'Gaussian smoothing in 3D before the same 14.65 Mpc/h projection. Local padded reads only; no whole-grid FFT.\nThis is a density illustration, not a new tidal-eigenvalue target or a demonstration of inference accuracy.',fontsize=10)
    fig.savefig(root/'03_smoothing_comparison.png',dpi=160);fig.savefig(root/'03_smoothing_comparison.pdf');plt.close(fig)
    report={'scales_mpc_h':scales,'smoothing':'3D Gaussian sigma=R, truncate=4; padded source block, valid interior projection',
        'job_id':os.environ['SLURM_JOB_ID'],'native_fields_found':{'R4':'fullgrid_v2, metadata 2048/2000/R4/threshold0.2; exact input-density provenance not yet recertified','R6_R7':'existing fullgrid_v3 ph000 native fields'},
        'support':[{'z':e['center_z'],'galaxies':e['n_galaxies'],'mean_separation_mpc_h':e['mean_density_mpc_h_minus3']**(-1/3),
                    'expected_spherical_counts':e['expected_count_in_spherical_aperture']} for e in m['examples']],
        'limitations':['Counts are mean-density top-hat aperture diagnostics, not Gaussian effective sample sizes or hard feasibility gates.',
          'Full flux-limited parent is denser than footprint/fibre-assigned sample.','Local slabs, not representative full-redshift counts.',
          'No model training or label regeneration; prior host-index-based scale ranking needs corrected-label re-evaluation.']}
    (root/'smoothing_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['support']),flush=True)
if __name__=='__main__':main()
