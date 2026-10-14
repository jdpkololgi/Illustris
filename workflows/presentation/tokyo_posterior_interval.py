#!/usr/bin/env python3
"""One title-free posterior interval figure from saved, exposed ph006 draws.

Selection uses only posterior class concentration; no truth arrays are read.
No model inference, scheduler action, or posterior correction is performed.
"""
import argparse, datetime, hashlib, json, platform, shutil, socket, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
SOURCE=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_halo48_candidate_20260924_v1')
COLORS=['#9CF6D5','#5B8FF9','#E85D9E','#F2C14E']
NAMES=['Void','Sheet','Filament','Knot']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def physical(s,mean,std):
 # Match the canonical audit: unstandardize first, then float64 softplus gaps.
 theta=np.asarray(np.asarray(s)*std+mean,dtype=np.float64)
 e=np.empty_like(theta);e[...,0]=theta[...,0]
 e[...,1]=e[...,0]+np.logaddexp(0.,theta[...,1]);e[...,2]=e[...,1]+np.logaddexp(0.,theta[...,2])
 return e

def extract(out):
 import torch
 torch.set_num_threads(1)
 auditpath=SOURCE/'posterior/calibration_audit/P12A_CALIBRATION_AUDIT.json'
 a=read(auditpath);p=a['provenance'];inputs=[]
 for name in ['checkpoint','dataset_marker','source_result']:
  path=Path(p[name]);h=sha(path);assert h==p[name+'_sha256'],name
  inputs.append({'path':str(path),'sha256':h})
 ck=torch.load(p['checkpoint'],map_location='cpu',weights_only=False)
 mean=np.asarray(ck['theta_mean']);std=np.asarray(ck['theta_std'])
 samples=np.load(p['samples'],mmap_mode='r');index=np.load(p['evaluation_index'])
 assert samples.shape==(50000,512,3) and index.shape==(50000,)
 maxprob=np.empty(len(index));prob=np.empty((len(index),4))
 for start in range(0,len(index),256):
  e=physical(samples[start:start+256],mean,std)
  assert np.isfinite(e).all() and (np.diff(e,axis=-1)>=0).all()
  cls=(e>.2).sum(-1)
  prob[start:start+len(e)]=np.stack([(cls==i).mean(1) for i in range(4)],axis=-1)
  maxprob[start:start+len(e)]=prob[start:start+len(e)].max(1)
 med=float(np.median(maxprob));chosen=int(np.argmin(np.abs(maxprob-med)))
 draws=physical(samples[chosen],mean,std)
 classes=(draws>.2).sum(-1);counts=np.bincount(classes,minlength=4)
 probs=counts/len(draws)
 assert np.array_equal(probs,prob[chosen]) and counts.sum()==512
 levels=np.array([.025,.16,.50,.84,.975]);q=np.quantile(draws,levels,axis=0,method='linear')
 ready=read(p['dataset_marker']);vp=Path(ready['validation']['path'])
 vh=sha(vp);assert vh==ready['validation']['sha256']
 # Read identity/context only: not truth_eigenvalues or theta_softplus.
 with np.load(vp) as d:
  j=int(index[chosen]);node=int(d['parent_node_id'][j]);context=d['context'][j]
  identity={k:int(d[k][j]) for k in ['shell','cap','fold','superblock_id','random_support','phase_index']}
 assert identity['random_support']==1
 np.savez_compressed(out/'data/posterior_example.npz',joint_eigenvalue_draws=draws,
    quantile_levels=levels,quantiles=q,class_counts=counts,class_probabilities=probs,
    evaluation_index=chosen,validation_index=j,parent_node_id=node,context=context,
    selection_population_max_class_probability=maxprob)
 for key in ['samples','evaluation_index']:
  path=Path(p[key]);inputs.append({'path':str(path),'sha256':sha(path)})
 inputs.extend([{'path':str(auditpath),'sha256':sha(auditpath)},{'path':str(vp),'sha256':vh}])
 m={'asset_id':'A02_single_galaxy_intervals','phase':'ph006','data_kind':'mock galaxy; already-exposed validation sample',
    'model_id':'P12-A halo48 candidate 20260924 v1','checkpoint_sha256':p['checkpoint_sha256'],
    'posterior_temperature':1.,'draw_count':512,'new_sampling':False,
    'sampling_solver_settings':'reused saved calibration-audit draws; no solver run; see source audit for original settings',
    'target':'ordered normalized tidal eigenvalues; R=7 h^-1 Mpc; target epoch z=0.2',
    'class_threshold':.2,'class_rule':'count eigenvalues strictly greater than threshold in each JOINT triplet',
    'selection_rule':'nearest to unweighted median maximum posterior class probability across all 50,000 saved evaluation rows; ties use first evaluation-array row; no truth values inspected',
    'selection_population_size':50000,'population_median_max_class_probability':med,
    'selected_max_class_probability':float(probs.max()),'evaluation_array_row':chosen,
    'validation_dataset_row':j,'parent_node_id':node,'identity':identity,'observed_redshift':float(context[3]),
    'intervals':'marginal equal-tailed central 68% [q16,q84] and 95% [q2.5,q97.5]; median q50; numpy linear quantiles',
    'class_probabilities':dict(zip(NAMES,probs.tolist())),'class_counts':dict(zip(NAMES,counts.tolist())),
    'quantile_levels':levels.tolist(),'quantiles_by_level_then_eigenvalue':q.tolist(),
    'input_paths_and_sha256':inputs,'source_repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    'extraction_script_sha256':sha(__file__),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'allowed_claim':'continuous per-galaxy eigenvalue uncertainty supports derived class probabilities; categorical labels discard information',
    'limitations':['one illustrative exposed mock galaxy, not a calibration demonstration','conditional on registered model and mock/HOD','no eigenvectors or joint spatial field inferred','512-draw empirical class probabilities; zero draws is not proof of zero underlying probability'],
    'geometry_coordinate_definition':'not applicable: parameter-space interval plot','display_thinning':'none',
    'selection_seed':'not applicable: deterministic posterior-only rule','truth_used_for_selection_or_plot':False,
    'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,
      'host':socket.gethostname(),'threads':1,'scheduler_actions':'none'}}
 write(out/'data/receipt.json',m)
 print(json.dumps({'evaluation_row':chosen,'parent_node_id':node,'probabilities':m['class_probabilities'],'median_max_probability':med}),flush=True)

def render(out):
 d=np.load(out/'data/posterior_example.npz');q=d['quantiles'];probs=d['class_probabilities']
 plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#F5F5F5','axes.labelcolor':'#F5F5F5',
   'xtick.color':'#C8CDD4','ytick.color':'#F5F5F5','axes.edgecolor':'#C8CDD4',
   'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'black'})
 fig=plt.figure(figsize=(1920/144,1080/144),dpi=144,facecolor='black')
 fig.text(.15,.87,'Mock galaxy',fontsize=17,color='#C8CDD4')
 # An interval key rather than a title or a paragraph.
 handles=[Line2D([0],[0],color='#7D8795',lw=2.5,label='95%'),
          Line2D([0],[0],color='#F5F5F5',lw=7,label='68%'),
          Line2D([0],[0],color='#F5F5F5',lw=0,marker='o',markersize=7,label='Median')]
 fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.9,.91),ncol=3,frameon=False,fontsize=15,
            handlelength=1.7,columnspacing=1.6)
 ax=fig.add_axes([.15,.39,.75,.39],facecolor='black')
 span=float(q[-1].max()-q[0].min());lo=min(float(q[0].min()),.2)-.1*span;hi=max(float(q[-1].max()),.2)+.1*span
 ax.axvline(.2,color='#C8CDD4',lw=1.5,ls=(0,(5,5)),zorder=0)
 ax.text(.2,2.52,'λₜₕ = 0.2',fontsize=15,ha='center',va='bottom',color='#C8CDD4')
 for i,y in enumerate([2,1,0]):
  ax.plot(q[[0,4],i],[y,y],color='#7D8795',lw=2.5,solid_capstyle='round')
  ax.plot(q[[1,3],i],[y,y],color='#F5F5F5',lw=7,solid_capstyle='round')
  ax.plot(q[2,i],y,'o',ms=8,mec='black',mew=1.1,mfc='#F5F5F5',zorder=4)
 ax.set(xlim=(lo,hi),ylim=(-.5,2.7),yticks=[2,1,0],yticklabels=['λ₁','λ₂','λ₃'])
 ax.tick_params(axis='y',length=0,pad=24,labelsize=25)
 ax.tick_params(axis='x',labelsize=17,pad=10,length=5)
 ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
 ax.set_xlabel('λ (dimensionless)',fontsize=18,labelpad=14)
 for name in ['left','top','right']:ax.spines[name].set_visible(False)
 ax.spines['bottom'].set_color('#616973')
 bar=fig.add_axes([.15,.195,.75,.05],facecolor='black')
 start=0.
 for value,color in zip(probs,COLORS):
  if value>0:bar.barh(0,value,left=start,height=1,color=color,edgecolor='black',linewidth=1)
  start+=value
 bar.set(xlim=(0,1),ylim=(-.5,.5));bar.axis('off')
 for i,(name,value,color) in enumerate(zip(NAMES,probs,COLORS)):
  x=.15+.75*(i+.5)/4
  shown=f'{100*value:.0f}%' if value else '0 / 512'
  fig.text(x,.132,f'{name}  {shown}',color=color,ha='center',fontsize=17)
 fig.text(.525,.065,'Class probabilities from joint draws',ha='center',fontsize=14,color='#C8CDD4')
 fig.canvas.draw();r=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
 for t in fig.texts:
  b=t.get_window_extent(r);assert b.x0>=0 and b.y0>=0 and b.x1<=w and b.y1<=h,t.get_text()
 for ext in ['png','pdf','svg']:fig.savefig(out/f'A02_posterior_intervals.{ext}',dpi=144)
 plt.close(fig)
 im=Image.open(out/'A02_posterior_intervals.png').convert('RGB');assert im.size==(1920,1080)
 im.resize((1280,720),Image.Resampling.LANCZOS).save(out/'qa/preview_720p.png')
 assert b'/FontFile2' in (out/'A02_posterior_intervals.pdf').read_bytes()
 m=read(out/'data/receipt.json');m['axis_limits']=[lo,hi];m['render_script_sha256']=sha(__file__)
 m['output_hashes']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
 write(out/'manifest.json',m)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--render-only',action='store_true');a=p.parse_args()
 for folder in ['data','qa']:(a.output/folder).mkdir(parents=True,exist_ok=True)
 if not a.render_only:extract(a.output)
 render(a.output)
if __name__=='__main__':main()
