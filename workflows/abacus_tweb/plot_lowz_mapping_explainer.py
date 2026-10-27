#!/usr/bin/env python3
"""Local ph000 illustration from existing audit samples and mapped field cells.
Requires a compute step; never updates production catalogues. No fit or FFT.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import numpy as np
from p12a_coordinate_sample_audit import mapped_member, grid_indices

BASE = Path('/pscratch/sd/d/dkololgi/abacus/lowz_index_audit_20261019')
FIELD = Path('/pscratch/sd/d/dkololgi/AbacusSummit_densities/tweb_rank_outputs_fullgrid_v3/dens_AbacusSummit_base_c000_ph000_z0.200_ngrid2048_box2000_thr0p2/backend_optimized_ngrid_2048_rsmooth_7')
DENS = Path('/pscratch/sd/d/dkololgi/AbacusSummit_densities/density_fields/AbacusSummit_base_c000_ph000_z0.200_ngrid_2048_10pc_density_field.npy')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not (os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):
        raise RuntimeError('Run inside an explicitly approved Slurm compute step')
    args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    s = np.load(BASE/'job59999126/sample_and_reconstruction.npz')
    old = np.load(BASE/'job59999126/cleanFalse_direct_x_L2com.npz')
    rec = np.load(BASE/'recovery_job60000317/candidate_sample_mapping.npz')
    assert np.array_equal(s['original_row'], old['original_row'])
    assert np.array_equal(s['original_row'], rec['original_row'])
    true = s['sky_c000_massive_nu_approx'] - 1000.
    wrong = old['lookup_position']
    displacement = (wrong-true+1000.) % 2000. - 1000.
    err = np.linalg.norm(displacement, axis=1)
    low = s['Z_COSMO'] < .15
    central = (s['RES'] == 1) & (s['CEN'] == 1)
    assert np.count_nonzero(low & central) == 135
    cache = {}
    def slab(ix):
        rank = int(ix)//16
        if rank not in cache:
            p = FIELD/f'abacus_cactus_tweb_rank{rank:04d}.npz'
            with np.load(p) as a:
                assert int(a['x_start']) == rank*16 and int(a['x_end']) == (rank+1)*16
            cache[rank] = mapped_member(p, 'eig_vals')
        return cache[rank], int(ix)%16
    def values(pos):
        out = []
        for ix,iy,iz in grid_indices(pos):
            a,j = slab(ix)
            out.append(a[:,j,iy,iz])
        return np.asarray(out)
    good_eig, bad_eig = values(true), values(wrong)
    np.savez(args.output/'sampled_fields.npz', original_row=s['original_row'], galaxy_position=true,
             old_host_position=wrong, galaxy_eigenvalues=good_eig, old_host_eigenvalues=bad_eig)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.axis('off')
    boxes = [(.02,.62,'Cut-sky galaxy\nRA, Dec, Z_COSMO'),(.37,.62,'Observer-centred position\nχ(z) × sky unit vector'),(.72,.62,'Native simulation box\nsubtract (1000,1000,1000)\nwrap periodically at 2000 Mpc/h'),(.72,.12,'Sample grid at galaxy position\nλ₁, λ₂, λ₃ at R = 7 Mpc/h'),(.02,.12,'Old low-z route: HALO_INDEX\nindex in a filtered host list'),(.37,.12,'Used as a raw halo row\nWRONG LOCATION')]
    for x,y,t in boxes:
        ax.text(x+.12,y+.1,t,ha='center',va='center',transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=.7',fc='#f4e4e2' if y<.2 and x<.6 else '#e5f1f4',ec='#647780'),fontsize=10)
    for a,b in [((.26,.72),(.35,.72)),((.61,.72),(.7,.72)),((.84,.58),(.84,.38)),((.26,.22),(.35,.22))]:
        ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=2))
    fig.suptitle('ph000 cut-sky → periodic z = 0.2 simulation: two different lookup routes',fontsize=15)
    fig.text(.04,.015,'Use cosmological redshift for real-space reconstruction. Satellites have distinct galaxy and host positions.\nObserver convention verified on this sample; broader phases and particle-host identities remain unvalidated.',fontsize=10)
    fig.savefig(args.output/'01_coordinate_routes.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.scatter(s['Z_COSMO'][central],err[central],s=17,alpha=.65,label='Old raw-row lookup (resolved centrals)',color='#bd493b')
    ax.scatter(s['Z_COSMO'][low&central],rec['position_error_mpc_h'][low&central],s=19,alpha=.7,label='Recovered filtered-list lookup (135 low-z centrals)',color='#127d8e')
    ax.axvline(.15,color='black',ls='--',lw=1)
    ax.set(yscale='log',xlabel='Cosmological redshift',ylabel='Periodic position error [Mpc/h]',title='The discontinuity is in the catalogue lookup')
    ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(args.output/'02_position_error.png',dpi=180); plt.close(fig)
    ids = np.flatnonzero(low & central)
    chosen = ids[np.argsort(err[ids])[len(ids)//2]]
    density = np.load(DENS,mmap_mode='r')
    assert density.shape == (2048,2048,2048)
    maps=[]; tidal=[]; coordinates=[]
    half=64; step=2000/2048
    for pos in [true[chosen],wrong[chosen]]:
        ix,iy,iz = grid_indices(pos[None])[0]
        ys=(np.arange(iy-half,iy+half+1)%2048).astype(int)
        zs=(np.arange(iz-half,iz+half+1)%2048).astype(int)
        # A bounded 15-cell projection of the existing particle density grid.
        image=np.zeros((len(ys),len(zs)))
        for xx in (np.arange(ix-7,ix+8)%2048):
            image += density[xx][np.ix_(ys,zs)]/15
        maps.append(image)
        a,j=slab(ix)
        tidal.append(sum(a[k,j][np.ix_(ys,zs)] for k in range(3)))
        coordinates.append([int(ix),int(iy),int(iz)])
    fig, axs=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    extent=np.array([-half-.5,half+.5,-half-.5,half+.5])*step
    for row, data in enumerate([maps,tidal]):
        lim=np.percentile(np.concatenate([v.ravel() for v in data]),[2,98])
        norm=Normalize(*lim)
        for col,v in enumerate(data):
            im=axs[row,col].imshow(v.T,origin='lower',extent=extent,cmap='magma',norm=norm)
            axs[row,col].plot(0,0,'+',color='cyan',ms=16,mew=2)
            axs[row,col].set(xlabel='y offset [Mpc/h]',ylabel='z offset [Mpc/h]')
            axs[row,col].set_title(('Galaxy location' if col==0 else 'Old assigned host location')+f'\ngrid cell {coordinates[col]}',fontsize=10)
        fig.colorbar(im,ax=axs[row,:],label='Stored particle-density field: 14.65 Mpc/h projection' if row==0 else 'Tidal trace λ₁ + λ₂ + λ₃, R = 7 Mpc/h; one x-cell slice',shrink=.85)
    fig.suptitle(f'Actual ph000 field around one low-z central: z = {s["Z_COSMO"][chosen]:.4f}\nOld lookup samples a different region, {err[chosen]:.1f} Mpc/h away',fontsize=14)
    fig.savefig(args.output/'03_actual_field_locations.png',dpi=180); plt.close(fig)
    fig, axs=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    correlations={}
    for ax,mask,name in zip(axs,[low&central,~low&central],['z < 0.15: 135 resolved centrals','z ≥ 0.15: 217 resolved centrals']):
        x=good_eig[mask].sum(axis=1); y=bad_eig[mask].sum(axis=1)
        r=float(np.corrcoef(x,y)[0,1]); correlations[name]=r
        lo,hi=min(x.min(),y.min()),max(x.max(),y.max())
        ax.scatter(x,y,s=17,alpha=.65); ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set(xlabel='Trace at reconstructed galaxy position',ylabel='Trace at old assigned host position',title=f'{name}\nPearson r = {r:.3f}')
    fig.suptitle('Label agreement test — not a galaxy clustering or model-performance test')
    fig.savefig(args.output/'04_label_agreement.png',dpi=180); plt.close(fig)
    report=dict(phase='ph000',snapshot_z=.2,job_id=os.environ['SLURM_JOB_ID'],field=str(FIELD),density=str(DENS),
        example_original_row=int(s['original_row'][chosen]),example_position_error_mpc_h=float(err[chosen]),
        central_lowz_median_old_error_mpc_h=float(np.median(err[low&central])),trace_correlations=correlations,
        density_dtype=str(density.dtype),density_projection_cells=15,grid_spacing_mpc_h=step,
        limitations=['One file and phase; resolved centrals only in statistical panels.',
        'Galaxy-position field differs from host-centre estimand for satellites.',
        'Original field files sampled, not fully checksummed; no full-grid FFT or recomputation.',
        'No downstream learnability test; particle-host mapping not independently validated.'])
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
