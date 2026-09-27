"""Reproducible static calibration plots; optional compute-only draw reduction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
SOURCES=[BASE/'cfm_mixed_20260926_v1/COMPLETE.json',BASE/'cfm_additional_20260926_v1/COMPLETE.json']
PHASES=[f'ph{i:03d}' for i in range(12,18)]
COLORS=['#2563a6','#d07a26','#8064a2','#597d46','#b34f77','#177e89']

def inputs():
    rows=[];paths={};hashes={}
    for source in SOURCES:
        data=json.loads(source.read_text());hashes[str(source)]=sha(source)
        rows.extend(r for r in data['rows'] if r['arm']=='mixed')
        for filename,h in data['input_hashes'].items():
            p=Path(filename)
            if 'fine26624' not in filename or '/nfe128/' not in filename:continue
            if sha(p)!=h:raise ValueError('case receipt checksum')
            v=json.loads(p.read_text())
            if v['phase'] not in PHASES:raise ValueError('unauthorized phase')
            seed=int(next(s for s in p.parts if s.startswith('seed')).split('_')[0][4:])
            paths[(v['phase'],seed,v['pair_id'])]=p
    if len(rows)!=192 or len(paths)!=192:raise ValueError('incomplete 6-phase panel')
    return rows,paths,hashes

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight',facecolor='white')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)

def ranks(rows,out):
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for ax,g,title in zip(axes.flat,['core_mass','block_mass','fine_mass','density','tidal','eigengap'],
                           ['Core mass','Block mass','Fine-sensitive regional mass','Density','Tidal eigenvalues','Eigengaps']):
        for phase,color in zip(PHASES,COLORS):
            for seed,style in [(17,'-'),(29,'--')]:
                hist=np.sum([r['scores'][g]['rank_histogram'] for r in rows if (r['phase'],r['seed'])==(phase,seed)],axis=0)
                ax.plot((np.arange(8)+.5)/8,hist/hist.sum()*8,color=color,ls=style,lw=1.4,label=phase if seed==17 else None)
        ax.axhline(1,color='#333333',lw=1,ls=':');ax.set_title(title);ax.set_ylim(0,3);ax.grid(alpha=.15)
        ax.set_xlabel('Randomized truth rank');ax.set_ylabel('Frequency / uniform expectation')
    axes[0,0].legend(ncol=2,fontsize=8)
    fig.suptitle('SBC-style held-out mock ranks | mixed coarse13k / fine26k EMA',fontsize=16)
    fig.text(.5,.015,'32 draws per anchor; 16 anchors per phase. Solid: seed17; dashed: seed29.\nOverlapping regions and repeated seeds are not independent simulations; no iid confidence bands.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.95));save(fig,out,'sbc_ranks')

def diagram(out):
    fig,ax=plt.subplots(figsize=(14,10));ax.set(xlim=(0,14),ylim=(0,10));ax.axis('off')
    def box(x,y,w,h,text,color='#e8f0fa'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',facecolor=color,edgecolor='#536171'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#536171',lw=1.6))
    ax.text(7,9.65,'Conditional field posterior: the current coupled CFM model',ha='center',fontsize=17,weight='bold')
    box(.3,8,5.3,1,'Abacus mock galaxies + survey-response products\nCounts, expected counts, support/exposure,\nboundary distance, line of sight, redshift/radius')
    box(7.7,8,5.6,1,'Training only: paired R7 matter density\nSplit into positive coarse density +\nblock-zero-mean log-density residual', '#f8eddc')
    box(.3,6.3,5.3,1,'Global normalization fitted on training only\n12-channel joint view: 64 × 48 × 48\n12-channel wide view: 48 × 48 × 48')
    arrow((2.9,8),(2.9,7.3))
    box(.3,4.4,5.3,1.25,'COARSE conditional flow | EMA update13,312\nGaussian noise → 64 Heun steps (128 NFE)\n3D residual U-Net + time embedding\nSpatial joint/wide context cross-attention')
    arrow((2.9,6.3),(2.9,5.65))
    box(7.7,4.4,5.6,1.25,'FINE conditional flow | EMA update26,624\nIndependent projected noise → 64 Heun steps\nSame U-Net/context design; joint fine domain\nConditioned on this sampled coarse field')
    arrow((5.6,5),(7.7,5));ax.text(6.6,5.2,'shared coarse draw',ha='center',fontsize=9)
    ax.plot([5.6,6.5,6.5,10.5],[6.8,6.8,5.95,5.95],color='#536171',lw=1.6)
    arrow((10.5,5.95),(10.5,5.65));ax.text(8.5,6.02,'same observations',ha='center',fontsize=9)
    box(7.7,6.3,5.6,.9,'Training: CFM velocity loss, spectral weighting\nα = 0.30; EMA + learning-rate decay\nFine training uses TRUE coarse fields', '#f8eddc')
    arrow((10.5,8),(10.5,7.2))
    box(3.3,2.25,7.4,1.15,'Positive, block-mass-conserving physical decode\nρ = lift(ρcoarse) × exp(u) / lift(block mean exp(u))\nδ = ρ − 1; neighboring fine regions generated JOINTLY', '#e8f1e5')
    arrow((2.9,4.4),(5,3.4));arrow((10.5,4.4),(9,3.4))
    box(3.3,.5,7.4,1.05,'Posterior field draws → regional masses and spectra\nConsistent wide + fine tidal operator → ordered eigenvalues\nPilot T-Web classes: number of eigenvalues > 0', '#e8f1e5')
    arrow((7,2.25),(7,1.55))
    ax.text(7,.08,'Mock-domain exploratory model; not yet DESI-calibrated. Pilot class threshold differs from production VAC.',ha='center',fontsize=9)
    save(fig,out,'model_dataflow')

def regional(x):
    # Same physical regions as the authoritative evaluator; leading axis=draw.
    cores=[x[:,s:s+16,16:32,16:32].mean((1,2,3)) for s in (16,32)]
    blocks=[x[:,s:s+24,y:y+24,z:z+24].mean((1,2,3)) for s in (8,32) for y in (0,24) for z in (0,24)]
    fine=[x[:,s:s+12,y:y+12,z:z+12].mean((1,2,3)) for s in (17,33) for y in (17,19) for z in (17,19)]
    return np.stack(cores+blocks+fine,axis=1)

def extract(rows,paths,cache):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('raw draw reduction requires compute allocation')
    samples=[];truth=[];identities=[]
    for row in rows:
        identity=(row['phase'],row['seed'],row['pair']);p=paths[identity];v=json.loads(p.read_text());parts=[]
        for name,h in sorted(v['chunks'].items()):
            chunk=p.parent/name
            if sha(chunk)!=h:raise ValueError('draw checksum')
            with np.load(chunk) as f:parts.append(regional(f['rho']-1))
        x=np.concatenate(parts)
        np.testing.assert_allclose(x.mean(0),row['scores']['region_mean'],rtol=2e-5,atol=1e-7)
        np.testing.assert_allclose(x.std(0,ddof=1),row['scores']['region_std'],rtol=2e-5,atol=1e-7)
        samples.append(x);truth.append(row['scores']['region_truth']);identities.append(identity)
        print('EXTRACT',identity,flush=True)
    np.savez(cache,samples=np.array(samples),truth=np.array(truth),identities=np.array(identities))

def tarp_curves(x,y,seed=71,width=.25):
    rng=np.random.default_rng(seed);refs=rng.normal(0,width,size=(64,len(y),y.shape[-1]))
    ds=((x[None]-refs[:,:,None,:])**2).sum(-1);dt=((y[None]-refs)**2).sum(-1)
    less=(ds<dt[:,:,None]).sum(-1);equal=(ds==dt[:,:,None]).sum(-1)
    # Finite-M randomized ranks preserve a uniform null rather than a staircase bias.
    ranks=(less+rng.random(less.shape)*(equal+1))/(x.shape[1]+1)
    alpha=np.linspace(0,1,21)
    return alpha,(ranks[:,:,None]<=alpha).mean(0)

def tarp(cache,out):
    with np.load(cache) as v:x=v['samples'];y=v['truth'];ids=v['identities']
    fig,axes=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True)
    records={}
    for ax,phase,color in zip(axes.flat,PHASES,COLORS):
        for seed,style in [(17,'-'),(29,'--')]:
            use=(ids[:,0]==phase)&(ids[:,1]==str(seed))
            for sl,label,c in [(slice(0,2),'Core pair','#2563a6'),(slice(2,10),'8 blocks','#d07a26'),(slice(10,18),'8 fine regions','#8064a2')]:
                a,curves=tarp_curves(x[use,:,sl],y[use,sl]);curve=curves.mean(0)
                ax.plot(a,curve,color=c,ls=style,label=label if seed==17 else None)
                records[f'{phase}_{seed}_{label}']=curve.tolist()
        ax.plot([0,1],[0,1],color='#333333',ls=':',lw=1);ax.set_title(phase+' | 16 anchor pairs');ax.grid(alpha=.15)
        ax.set_xlabel('Credibility level');ax.set_ylabel('Empirical coverage');ax.set(xlim=(0,1),ylim=(0,1))
    axes[0,0].legend(fontsize=9)
    fig.suptitle('TARP: joint regional-mass posterior checks',fontsize=16)
    fig.text(.5,.015,'32 draws; 64 random-reference repetitions (not 64 extra simulations). Solid: seed17; dashed: seed29.\nEuclidean distance in physical δ units; references N(0, 0.25²). Finite-M randomized ranks; no iid confidence bands.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.95));save(fig,out,'tarp_regions')
    (out/'tarp_curves.json').write_text(json.dumps(dict(alpha=a.tolist(),curves=records),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--extract',action='store_true');p.add_argument('--cache');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rows,paths,hashes=inputs()
    if args.extract:extract(rows,paths,Path(args.cache))
    ranks(rows,out);diagram(out)
    if args.cache and Path(args.cache).exists():tarp(Path(args.cache),out)
    (out/'sources.json').write_text(json.dumps(hashes,indent=2))
