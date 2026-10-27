#!/usr/bin/env python3
"""Full ph000 radial census and Gaussian sampling-scale guide; NOT an optimum.
N_eff=8*pi^(3/2)*nbar*R^3 follows from 1/integral(W_R^2) for a normalized
3D Gaussian and a homogeneous unweighted Poisson tracer process.
No inference accuracy or calibration is estimated by this calculation.
"""
import argparse,hashlib,json,os,socket
from pathlib import Path
import numpy as np
import fitsio
from galaxy_position_mapping import comoving_distance
SOURCE=Path('/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if not(os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):ap.error('Approved compute step required')
    args.output.mkdir(parents=True,exist_ok=False)
    edges=np.linspace(0,.6,25);centers=(edges[:-1]+edges[1:])/2
    counts=np.zeros((2,24),dtype=np.int64);angular=np.zeros((12,6),dtype=np.int64);excluded=0
    with fitsio.FITS(SOURCE) as f:
        n=f[1].get_nrows()
        for start in range(0,n,1000000):
            b=f[1].read(rows=np.arange(start,min(start+1000000,n)),columns=['Z_COSMO','R_MAG_APP','RA','DEC'])
            z=b['Z_COSMO'];good=np.isfinite(z)&(z>=0)&(z<.6);excluded+=int((~good).sum())
            counts[0]+=np.histogram(z[good],edges)[0];counts[1]+=np.histogram(z[good&(b['R_MAG_APP']<19.5)],edges)[0]
            angular+=np.histogram2d(b['RA'][good]%360,np.sin(np.deg2rad(b['DEC'][good])),bins=[np.linspace(0,360,13),np.linspace(-1,1,7)])[0].astype(np.int64)
            if start%10000000==0:print('SCANNED',min(start+1000000,n),'of',n,flush=True)
    assert np.all(angular>0) and counts[0].sum()+excluded==n and np.all(counts[1]<=counts[0])
    chi=comoving_distance(edges);volume=4*np.pi/3*np.diff(chi**3);density=counts/volume[None,:]
    constant=8*np.pi**1.5
    with np.errstate(divide='ignore'):
        r10=(10/(constant*density))**(1/3);r30=(30/(constant*density))**(1/3)
    np.savez(args.output/'radial_census.npz',z_edges=edges,counts=counts,fullsky_volume=volume,nbar=density,angular_counts=angular,r_neff10=r10,r_neff30=r30)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(13,6));fig.subplots_adjust(left=.08,right=.98,bottom=.23,top=.79,wspace=.27)
    names=['Raw mock parent','Magnitude-only r < 19.5'];colors=['#137f91','#ba4c45']
    for j,(label,color) in enumerate(zip(names,colors)):
        axs[0].plot(centers,density[j],'o-',color=color,ms=3,label=label)
        axs[1].plot(centers,r10[j],'-',color=color,label=label+'; N_eff = 10')
        axs[1].plot(centers,r30[j],'--',color=color,label=label+'; N_eff = 30')
        axs[1].fill_between(centers,r10[j],r30[j],color=color,alpha=.12)
    axs[0].set(yscale='log',xlabel='Cosmological redshift',ylabel='Full-sky shell number density [(Mpc/h)⁻³]',title='Measured mock sampling density',xlim=(0,.6));axs[0].legend(fontsize=9)
    for scale in [4,7,12,20]:
        axs[1].axhline(scale,color='#777777',alpha=.45,lw=.8)
        axs[1].text(.004,scale,f'{scale}',fontsize=8,va='bottom',color='#555555')
    axs[1].set(yscale='log',xlabel='Cosmological redshift',ylabel='Gaussian width R [Mpc/h]',title='Widths giving illustrative Poisson support',xlim=(0,.6))
    axs[1].legend(fontsize=8,loc='upper left')
    for ax in axs:ax.grid(alpha=.15)
    fig.suptitle('BGS smoothing versus redshift: sampling guide, not a measured optimum',fontsize=16,y=.97)
    fig.text(.08,.075,'N_eff = 8 pi^(3/2) nbar R^3 for an unweighted homogeneous Poisson sample and a normalized 3D Gaussian.\n10 and 30 are illustrative choices, not validated accuracy thresholds; shaded regions are not confidence intervals.',fontsize=10)
    fig.text(.08,.025,'ph000 full-sky parent, Δz = 0.025. No footprint, fibre assignment, weights or redshift failures. Selection can reduce support further.',fontsize=10)
    fig.savefig(args.output/'01_sampling_scale_vs_redshift.png',dpi=180);fig.savefig(args.output/'01_sampling_scale_vs_redshift.pdf');plt.close(fig)
    rows=[]
    for i,z in enumerate(centers):
        rows.append({'z_low':float(edges[i]),'z_high':float(edges[i+1]),'z_center':float(z),
            'parent_count':int(counts[0,i]),'bright_count':int(counts[1,i]),'parent_nbar':float(density[0,i]),'bright_nbar':float(density[1,i]),
            'parent_R_Neff10':float(r10[0,i]),'bright_R_Neff10':float(r10[1,i]),'parent_R_Neff30':float(r30[0,i]),'bright_R_Neff30':float(r30[1,i])})
    import csv
    with (args.output/'radial_census.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'completed':True,'job_id':os.environ['SLURM_JOB_ID'],'source':str(SOURCE),'source_rows':n,'outside_z_range':excluded,
        'source_bytes':SOURCE.stat().st_size,'source_mtime_ns':SOURCE.stat().st_mtime_ns,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'geometry':'Full-sky comoving shell volumes; raw unmasked parent. Coarse equal-solid-angle bins all populated, not a survey-mask measurement.',
        'formula':'N_eff=8*pi**1.5*nbar*R**3; normalized 3D Gaussian, homogeneous unweighted Poisson reference',
        'not_an_optimum':True,'thresholds_illustrative':[10,30],'rows':rows,
        'limitations':['No inference evaluation, optimum or calibrated scale established.','One mock phase; full-sky mean density hides local clustering and angular selection.',
        'Magnitude-only bright subset is not the observed BGS catalogue.','Formula describes scalar density Poisson support; tidal eigenvalue errors require separate validation.',
        'No confidence intervals; threshold band is sensitivity to an arbitrary support choice.','Field truth is the fixed z=0.2 snapshot, not an evolving light cone.']}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('COMPLETE',flush=True)
if __name__=='__main__':main()
