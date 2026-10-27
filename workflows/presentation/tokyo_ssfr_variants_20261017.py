#!/usr/bin/env python3
"""One-series A07 and cached-wedge continuous design variants; no new inference."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
ROOT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010');OUT=ROOT/'a07_simplified_20261017';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((ROOT/'a07_combined/plotted_statistics.json').read_text());theme=json.loads((ROOT/'theme.json').read_text())
cache=ROOT/'density_checks_20261014/matched_galaxies.npz';m=json.loads((cache.parent/'MANIFEST.json').read_text());assert sha(cache)==m['files'][cache.name]
a=dict(np.load(cache));y=a['log_ssfr'];n=len(y);nb=8
_,cell=np.unique(a['mass_z_cells'],return_inverse=True);_,reg=np.unique(a['region'],return_inverse=True);nc=cell.max()+1;nr=reg.max()+1
variables={'trace':a['trace'],'density':np.log10(a['density'])}
bins={};counts={};edges={}
for key,x in variables.items():
 edges[key]=np.quantile(x,np.linspace(0,1,nb+1));assert np.all(np.diff(edges[key])>0)
 bins[key]=np.searchsorted(edges[key][1:-1],x,side='right')
 counts[key]=np.bincount(bins[key]*nc+cell,minlength=nb*nc).reshape(nb,nc)
keep=np.logical_and.reduce([(v>=20).all(0) for v in counts.values()]);assert keep.sum()>0
mask=keep[cell];ref=np.bincount(cell[mask],minlength=nc)[keep].astype(float);ref/=ref.sum()
rng=np.random.default_rng(20261017);mult=rng.multinomial(nr,np.full(nr,1/nr),size=128)
stats={}
for key,x in variables.items():
 b=bins[key];ind=(b*nc+cell)*nr+reg
 def agg(v):return np.bincount(ind,weights=v,minlength=nb*nc*nr).reshape(nb,nc,nr)[:,keep,:]
 N=agg(np.ones(n));Y=agg(y);X=agg(x);tot=N.sum(2)
 mean=(Y.sum(2)/tot)@ref;mx=(X.sum(2)/tot)@ref
 # Fixed pooled reference and fixed support; refit each bin/cell mean per resample.
 bn=np.einsum('bcr,kr->kbc',N,mult);by=np.einsum('bcr,kr->kbc',Y,mult)
 good=(bn>0).all((1,2));boot=(by[good]/bn[good])@ref;assert good.sum()>=120
 # Independent direct weighted means and exact mass/z cell-balance contract.
 w=np.zeros(n)
 for j in range(nb):
  ix=(b==j)&mask; lookup=np.zeros(nc);lookup[keep]=ref/tot[j];w[ix]=lookup[cell[ix]]
  np.testing.assert_allclose(w[ix].sum(),1,atol=1e-13)
  np.testing.assert_allclose(np.bincount(cell[ix],weights=w[ix],minlength=nc)[keep],ref,atol=1e-13)
  np.testing.assert_allclose(np.sum(w[ix]*y[ix]),mean[j],atol=1e-12)
 # Cell-only outcome cannot produce an environmental effect under this estimator.
 cell_only=np.arange(keep.sum(),dtype=float)
 np.testing.assert_allclose((np.broadcast_to(cell_only,(nb,len(cell_only))))@ref,np.repeat(cell_only@ref,nb),atol=1e-12)
 stats[key]=dict(x=mx.tolist(),mean=mean.tolist(),interval=np.quantile(boot,[.16,.84],axis=0).tolist(),edges=edges[key].tolist(),retained_n=int(mask.sum()),common_cells=int(keep.sum()),valid_bootstrap=int(good.sum()),counts_per_bin=np.bincount(b[mask],minlength=nb).tolist(),high_minus_low=float(mean[-1]-mean[0]))
colors=[theme['class_palette'][k] for k in ['void','sheet','filament','knot']];names=d['class_order']
plt.rcParams.update({'font.family':theme['font'],'text.color':'#F5F5F5','axes.labelcolor':'#F5F5F5','xtick.color':'#C8CDD4','ytick.color':'#C8CDD4','axes.edgecolor':'#77818E','pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'black'})
for variant in ('classes','trace','density'):
 f=plt.figure(figsize=(1920/144,1080/144),dpi=144,facecolor='black');left=f.add_axes([.08,.31,.385,.43],facecolor='black');right=f.add_axes([.605,.31,.35,.43],facecolor='black')
 for ax in (left,right):
  ax.tick_params(axis='x',labelsize=17,pad=11,length=0);ax.tick_params(axis='y',labelsize=16,pad=7)
  for side in ('top','right'):ax.spines[side].set_visible(False)
 left.set_xticks(range(4),names);f.text(.2725,.858,'Environment fractions',ha='center',fontsize=22);f.text(.78,.858,'sSFR and environment',ha='center',fontsize=22)
 series=list(d['class_fractions']);alphas=[.42,.70,1.];width=.23
 for j,name in enumerate(series):
  values=d['class_fractions'][name];xs=np.arange(4)+(j-1)*width;left.bar(xs,values,width,color=colors,alpha=alphas[j],edgecolor='#E2E7ED',linewidth=.6)
  for xx,v in zip(xs,values):left.text(xx,v+.01,f'{v*100:.0f}',ha='center',fontsize=12,color='#E2E7ED')
 left.set(ylim=(0,.53),xlim=(-.6,3.6),yticks=[0,.1,.2,.3,.4,.5]);left.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));left.set_ylabel('Fraction of galaxies',fontsize=18,labelpad=9)
 left.legend(handles=[Patch(facecolor='#E2E7ED',alpha=al,label=s) for s,al in zip(series,alphas)],loc='lower center',bbox_to_anchor=(.5,1.015),ncol=3,fontsize=12,frameon=False,handlelength=1,columnspacing=.8)
 f.text(.78,.785,'Comparable stellar mass and redshift',ha='center',fontsize=12,color='#C8CDD4')
 right.set_ylabel('Mean log₁₀(sSFR / yr⁻¹)',fontsize=18,labelpad=11);right.set(ylim=(-11.10,-10.50),yticks=[-11.1,-10.9,-10.7,-10.5])
 if variant=='classes':
  r=d['ssfr'];mean=np.array(r['mean']);ci=np.array(r['interval'])
  right.set_xticks(range(4),names);right.set_xlim(-.6,3.6)
  for k in range(4):right.errorbar(k,mean[k],yerr=[[max(0,mean[k]-ci[0,k])],[max(0,ci[1,k]-mean[k])]],fmt='o',color=colors[k],ms=9,elinewidth=1.6,capsize=4)
  f.text(.78,.209,'Cluster − Void: −0.19 dex',ha='center',fontsize=17)
  note='16–84% spatial bootstrap; systematics not included'
 else:
  r=stats[variant];x=np.array(r['x']);mean=np.array(r['mean']);ci=np.array(r['interval']);c='#9CF6D5'
  right.plot(x,mean,'o-',color=c,lw=1.7,ms=5);right.fill_between(x,ci[0],ci[1],color=c,alpha=.2,lw=0)
  right.set_xlabel('Inferred tidal trace' if variant=='trace' else 'log₁₀(local galaxy density / n̄)',fontsize=16,labelpad=9)
  if variant=='trace':right.set_xticks([0,.5,1,1.5])
  note=f'Right: illustrative wedge, {int(mask.sum()):,} galaxies · 16–84% spatial bootstrap'
 f.text(.2725,.209,'0.20 ≤ z < 0.30',ha='center',fontsize=18)
 f.text(.52,.148,note,ha='center',fontsize=15,color='#C8CDD4');f.text(.52,.105,'Descriptive associations · real-DESI calibration unverified',ha='center',fontsize=16,color='#C8CDD4')
 for ext in ('png','pdf','svg'):f.savefig(OUT/f'A07_{variant}.{ext}',dpi=288 if ext=='png' else 144)
 plt.close(f)
(OUT/'statistics.json').write_text(json.dumps(dict(continuous=stats,input_n=n,method='Eight quantile bins on actual continuous variable; common .1 dex mass / .01 redshift cells with >=20 galaxies in every bin of both estimators; pooled fixed target mix; 128 paired nside8 sky-block bootstraps, fixed edges/support/reference, refitted within-cell means.',scope='Continuous panels: historical cached wedge RA120-160 Dec14.5-30.6, z.2-.3, positive legacy mass/SFR and valid all-seven-properties selection. Left fractions and categorical sSFR: original full-survey statistics. These panels are not the same population.',uncertainty='Spatial sampling only. No SED, environment posterior, selection or model systematics. Lines guide the eye; no fit or interpolation of class codes.',source_cache_sha256=sha(cache)),indent=2)+'\n')
np.savez_compressed(OUT/'continuous_rows.npz',TARGETID=a['TARGETID'],mass=a['mass'],z=a['z'],log_ssfr=y,trace=variables['trace'],log_density=variables['density'],mass_z_cell=cell,region=reg,retained=mask,trace_bin=bins['trace'],density_bin=bins['density'])
shutil.copy2(ROOT/'a07_combined/plotted_statistics.json',OUT/'original_fullsurvey_statistics.json');shutil.copy2(ROOT/'a07_combined/manifest.json',OUT/'original_fullsurvey_manifest.json');shutil.copy2(cache.parent/'provenance.json',OUT/'wedge_provenance.json')
(OUT/'scripts').mkdir(exist_ok=True);shutil.copy2(__file__,OUT/'scripts'/Path(__file__).name)
print(json.dumps(stats,indent=2))
