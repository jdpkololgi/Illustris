#!/usr/bin/env python3
"""Tokyo A02--A04 from immutable saved draws and source-bound schematics.
No model inference, phase opening, scheduler action, or product mutation.
"""
import argparse, csv, datetime, hashlib, json, platform, shutil, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image, ImageOps, ImageDraw
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010')
MODEL=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_halo48_candidate_20260924_v1')
DESI=Path('/global/homes/d/dkololgi/GraphWeb_DESI')
WHITE,GREY,DIM='#F5F5F5','#C8CDD4','#77818E'
BLUE,GOLD='#5B8FF9','#F2C14E'
NAMES=['Void','Wall','Filament','Cluster']
COLORS=['#9CF6D5',BLUE,'#E85D9E',GOLD]
PHASES=['ph000','ph002','ph003','ph004','ph005']
FEATURES=['base_lambda1','base_lambda2','base_lambda3','redshift','log_ntilde_mpc3','cap_ngc','log1p_random_support_boundary_distance_mpc']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def physical(s,mean,std):
 t=np.asarray(s*std+mean,dtype=np.float64);e=np.empty_like(t);e[...,0]=t[...,0]
 e[...,1]=e[...,0]+np.logaddexp(0,t[...,1]);e[...,2]=e[...,1]+np.logaddexp(0,t[...,2]);return e

def prepare(out):
 import torch
 torch.set_num_threads(1);inputs=[]
 def bind(path,expected=None,copy=False):
  path=Path(path);digest=sha(path)
  if expected:assert digest==expected,str(path)
  item={'path':str(path),'sha256':digest}
  if copy:
   target=out/'data/sources'/(digest[:12]+'_'+path.name);shutil.copy2(path,target);item['local_copy']=str(target.relative_to(out))
  inputs.append(item)
  return read(path) if path.suffix=='.json' else None
 old=read(ROOT/'posterior_single/manifest.json');oldpath=ROOT/'posterior_single/data/posterior_example.npz'
 bind(oldpath,old['output_hashes']['data/posterior_example.npz']);bind(ROOT/'posterior_single/manifest.json',copy=True)
 with np.load(oldpath) as d:concentration=d['selection_population_max_class_probability'].copy()
 assert concentration.shape==(50000,)
 # Fixed before selected draws are read: Q25/Q75 concentration, first tie.
 targets=np.quantile(concentration,[.25,.75],method='linear')
 selected=np.array([np.argmin(abs(concentration-x)) for x in targets]);assert len(set(selected))==2
 audit=bind(MODEL/'posterior/calibration_audit/P12A_CALIBRATION_AUDIT.json',copy=True);p=audit['provenance']
 ready=bind(p['dataset_marker'],p['dataset_marker_sha256'],True);result=bind(p['source_result'],p['source_result_sha256'],True)
 bind(p['checkpoint'],p['checkpoint_sha256']);ck=torch.load(p['checkpoint'],map_location='cpu',weights_only=False)
 assert ready['feature_names']==result['conditioning_features']==FEATURES
 assert ready['training_phases']==PHASES and ready['validation_phase']=='ph006'
 assert ready['validation']['calibration_folds']==[0,1] and ready['validation']['evaluation_folds']==[2,3,4]
 assert not ready['phase_is_feature'] and not ready['fold_is_feature']
 previous={x['path']:x['sha256'] for x in old['input_paths_and_sha256']}
 for key in ['samples','evaluation_index']:bind(p[key],previous[p[key]])
 samples=np.load(p['samples'],mmap_mode='r');index=np.load(p['evaluation_index']);assert samples.shape==(50000,512,3)
 draws=physical(np.asarray(samples[selected]),ck['theta_mean'],ck['theta_std'])
 assert np.isfinite(draws).all() and (np.diff(draws,axis=-1)>=0).all()
 classes=(draws>.2).sum(-1);counts=np.array([np.bincount(row,minlength=4) for row in classes]);probs=counts/512
 assert np.array_equal(probs.max(1),concentration[selected])
 levels=np.array([.025,.16,.5,.84,.975]);q=np.quantile(draws,levels,axis=1,method='linear').transpose(1,0,2)
 vp=ready['validation']['path'];bind(vp,ready['validation']['sha256']);ids=[]
 with np.load(vp) as d:
  # Identity/context only: no truth or target arrays loaded.
  for pos in selected:
   j=int(index[pos]);identity={k:int(d[k][j]) for k in ['parent_node_id','shell','cap','fold','superblock_id','random_support','phase_index']}
   identity.update(evaluation_array_row=int(pos),validation_dataset_row=j,observed_redshift=float(d['context'][j,3]))
   assert identity['random_support']==1 and identity['fold'] in [2,3,4];ids.append(identity)
 np.savez_compressed(out/'data/A02_posteriors.npz',joint_eigenvalue_draws=draws,quantile_levels=levels,quantiles=q,class_counts=counts,class_probabilities=probs,selected_evaluation_rows=selected,selection_population_concentration=concentration)
 with (out/'data/A02_values.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['example','parent_node_id','eigenvalue','q025','q16','q50','q84','q975'])
  for n in range(2):
   for a in range(3):w.writerow([n+1,ids[n]['parent_node_id'],a+1,*q[n,:,a]])
 summaries=[]
 for phase in PHASES+['ph006']:
  marker=ready['phase_audit'][phase];s=bind(marker['summary_marker'],marker['summary_marker_sha256'],True)
  assert s['pass'] and s['out_of_fold'] and s['context_halo_voxels']==48 and not s['sealed_phase_opened']
  assert s['training_phases']==(PHASES if phase=='ph006' else [x for x in PHASES if x!=phase]);summaries.append(s)
 full=summaries[-1];bind(full['checkpoint'],full['checkpoint_sha256']);enc=torch.load(full['checkpoint'],map_location='cpu',weights_only=False)
 assert enc['epoch']==20 and enc['training_phases']==PHASES
 golden=bind(REPO/'docs/evidence/p12/GOLDEN_OBSERVER_REPLAY_20260925_v3.json',copy=True)
 assert golden['checkpoint_sha256']==p['checkpoint_sha256'] and golden['observer_fields_pass'] and golden['posterior_parity']
 for run in golden['runs']:
  assert set(run['field_max_abs'])=={'counts','exposure_apodized','log_count_ratio'} and all(x==0 for x in run['field_max_abs'].values())
 release=bind(DESI/'docs/evidence/p12a_loa_full_20260925/SOURCE_MANIFEST.json',copy=True)
 for path,digest in [(p['checkpoint'],p['checkpoint_sha256']),(full['checkpoint'],full['checkpoint_sha256']),(p['dataset_marker'],p['dataset_marker_sha256'])]:assert release['artifacts'][path]==digest
 spatial=bind(REPO/'docs/evidence/p4/spatial_manifest.json',copy=True);rotations=bind(REPO/'docs/evidence/p4/rotations.json',copy=True)
 assert spatial['pass'] and spatial['unit_contract']['core_mpc_h']==64
 assert rotations['0']=={'train_folds':[2,3,4],'validation_fold':1,'development_test_fold':0}
 for rel in ['SCIENCE_LOG.md','docs/plan_generalisable_graphweb_vac.md','docs/p12a_halo48_candidate_plan_20260924.md','docs/p12a_golden_conditional_results_20260925.md','workflows/sbi/p12_calibration_diagnostics.py','workflows/sbi/p12_train_base_response_fmpe.py']:bind(REPO/rel)
 bind(MODEL/'source/workflows/abacus_tweb/p8_train_unet_patch.py');bind(DESI/'docs/p12a_provisional_release_20260928.md')
 receipt={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_id':'P12-A halo48 candidate 20260924 v1; frozen Loa deployment model','posterior_checkpoint_sha256':p['checkpoint_sha256'],'encoder_checkpoint_sha256':full['checkpoint_sha256'],'encoder_epoch':20,'input_paths_and_sha256':inputs,'conditioning_features':FEATURES,'verified_channels':['counts','exposure_apodized','log_count_ratio'],
 'selection_rule':'Nearest to unweighted Q25 and Q75 of maximum class probability among saved 50000 ph006 evaluation rows; first array row breaks ties; no truth read.','concentration_quantile_targets':targets.tolist(),'selected_concentrations':probs.max(1).tolist(),'identities':ids,'class_labels':NAMES,'class_probabilities':probs.tolist(),'class_counts':counts.tolist(),'class_rule':'count(lambda_a > 0.2) within each joint triplet','draw_count_each':512,'posterior_temperature':1.,'interval_definition':'Marginal equal-tailed central 68% and 95%; NumPy linear quantiles; median point',
 'sampling':'Saved audit draws only; no new sampling. Original source calls sbi sample_batched (sample fallback) without explicit solver overrides. Exact library-default tolerances were not serialized in audit receipt.','new_sampling':False,'truth_used':False,'kde':'not applicable: interval display','training_phases':PHASES,'selection_phase':'ph006; already exposed','ph006_folds':{'reserved_calibration':[0,1],'evaluation':[2,3,4]},'spatial_rotation_shown':rotations['0'],'unit_contract':spatial['unit_contract'],
 'confirmation':'Original ph001 result belongs to original P12-A and is already exposed. Fresh halo48 confirmation remains required; no fresh phase is assigned or opened here; no E2E reservation reassigned.','schematic_geometry':'Not to scale; no actual sky positions. Context may overlap; supervised core ownership is distinct.','target':'ordered normalized tidal eigenvalues; Gaussian R=7 h^-1 Mpc; target epoch z=0.2','halo_voxels':48,'encoder_grid_mpc':5,'alignment_voxels':8,
 'deviations_from_brief':['Use Void/Wall/Filament/Cluster as requested.','Use U-PATCH, not Residual U-PATCH: frozen source has concatenated U-Net skips and sequential convolution blocks.','Fresh confirmation is required for halo48, not a completed or transferable ph001 gate.'],
 'allowed_claims':{'A02':'Continuous per-galaxy uncertainty yields derived class probabilities; these examples are not a calibration test.','A03':'Spatial core ownership and phase cross-fitting prevent the illustrated supervised leakage; many cores are not independent universes.','A04':'Information flow for frozen provisional per-galaxy halo48; no joint spatial field or eigenvectors.'},'limits':['Concentration is not accuracy or a claim about density/support.','Zero of 512 draws is not proof of zero underlying probability.','Real-DESI coverage unverified; no new qualification.'],'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'threads':1,'scheduler_actions':'none'}}
 write(out/'data/receipt.json',receipt);print(json.dumps({'selected':ids,'probabilities':probs.tolist()}),flush=True)

def setup():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'text.color':WHITE,'axes.labelcolor':WHITE,'axes.edgecolor':DIM,'xtick.color':GREY,'ytick.color':WHITE,'savefig.facecolor':'black','pdf.fonttype':42,'svg.fonttype':'none'})
def canvas(title,review):
 f=plt.figure(figsize=(1920/144,1080/144),dpi=144,facecolor='black')
 if review:f.text(.05,.94,title,fontsize=27,va='center')
 return f
def text(f,x,y,s,size=17,color=WHITE,ha='left'):return f.text(x,y,s,fontsize=size,color=color,ha=ha,va='center')
def box(f,x,y,w,h,edge=DIM,fill='#11161D',style='solid'):f.add_artist(Rectangle((x,y),w,h,transform=f.transFigure,facecolor=fill,edgecolor=edge,linewidth=1.6,linestyle=style,zorder=0))
def arrow(f,start,end,dashed=False):f.add_artist(FancyArrowPatch(start,end,transform=f.transFigure,arrowstyle='-|>',mutation_scale=17,linewidth=1.5,color=GREY,linestyle='--' if dashed else '-',zorder=1))
def export(f,out,name):
 f.canvas.draw();renderer=f.canvas.get_renderer();width,height=f.canvas.get_width_height()
 from matplotlib.text import Text
 for t in f.findobj(Text):
  if t.get_visible() and t.get_text():
   b=t.get_window_extent(renderer);assert b.x0>=0 and b.y0>=0 and b.x1<=width and b.y1<=height,(name,t.get_text(),b.bounds)
 for ext in ['png','pdf','svg']:f.savefig(out/'static'/(name+'.'+ext),dpi=288 if ext=='png' else 144)
 plt.close(f);im=Image.open(out/'static'/(name+'.png')).convert('RGB');assert im.size==(3840,2160)
 preview=im.resize((1280,720),Image.Resampling.LANCZOS);preview.save(out/'qa'/(name+'_720p.png'));ImageOps.grayscale(preview).save(out/'qa'/(name+'_grayscale.png'))
 assert b'/FontFile2' in (out/'static'/(name+'.pdf')).read_bytes();assert '<text' in (out/'static'/(name+'.svg')).read_text()

def a02(out,review):
 d=np.load(out/'data/A02_posteriors.npz');q,probs=d['quantiles'],d['class_probabilities'];lo,hi=q[:,0].min(),q[:,-1].max();span=hi-lo;limits=[float(lo-.09*span),float(hi+.09*span)]
 f=canvas('A02  |  Eigenvalue uncertainty and derived classes',review)
 for n,left in enumerate([.10,.57]):
  text(f,left,.83,['Lower class concentration','Higher class concentration'][n],20)
  text(f,left,.785,['Mock example · 25th percentile','Mock example · 75th percentile'][n],14,GREY)
  ax=f.add_axes([left,.425,.365,.29],facecolor='black');ax.axvline(.2,color=GREY,lw=1.4,ls=(0,(5,5)));ax.text(.2,2.65,'λₜₕ = 0.2',fontsize=14,ha='center',color=GREY)
  for i,y in enumerate([2,1,0]):
   ax.plot(q[n,[0,4],i],[y,y],color=DIM,lw=2.6);ax.plot(q[n,[1,3],i],[y,y],color=WHITE,lw=7,solid_capstyle='round');ax.plot(q[n,2,i],y,marker=['o','s','D'][i],ms=7,mfc=WHITE,mec='black',mew=1)
  ax.set(xlim=limits,ylim=(-.5,2.7),yticks=[2,1,0],yticklabels=['λ₁','λ₂','λ₃']);ax.tick_params(axis='y',length=0,labelsize=21,pad=12);ax.tick_params(axis='x',labelsize=15,pad=7)
  from matplotlib.ticker import MaxNLocator
  ticks=MaxNLocator(nbins=4).tick_values(*limits);ax.set_xticks(ticks[(ticks>=limits[0]) & (ticks<=limits[1])]);ax.set_xlabel('λ (dimensionless)',fontsize=17,labelpad=9)
  for side in ['left','top','right']:ax.spines[side].set_visible(False)
  bar=f.add_axes([left,.265,.365,.043],facecolor='black');start=0
  for value,color in zip(probs[n],COLORS):bar.barh(0,value,left=start,height=1,color=color,edgecolor='black',linewidth=1);start+=value
  bar.set(xlim=(0,1),ylim=(-.5,.5),yticks=[],xticks=[0,.5,1]);bar.tick_params(labelsize=15,length=3,pad=3)
  for s in bar.spines.values():s.set_visible(False)
  for i,(name,value,color) in enumerate(zip(NAMES,probs[n],COLORS)):
   x=left+.365*(i+.5)/4;text(f,x,.192,name,15,color,'center');text(f,x,.155,f'{100*value:.1f}%',16,WHITE,'center')
 handles=[Line2D([0],[0],color=DIM,lw=2.6,label='95%'),Line2D([0],[0],color=WHITE,lw=7,label='68%'),Line2D([0],[0],color=WHITE,lw=0,marker='o',label='Median')]
 f.legend(handles=handles,loc='center',bbox_to_anchor=(.52,.875),ncol=3,frameon=False,fontsize=14,handlelength=1.3,columnspacing=1.7)
 text(f,.5,.093,'Marginal intervals; class probabilities computed from joint triplet draws',14,GREY,'center')
 text(f,.5,.05,'P12-A halo48 · exposed ph006 mocks · 512 saved draws each · R = 7 h⁻¹ Mpc · target z = 0.2',13,GREY,'center')
 export(f,out,'A02_posteriors_'+('review' if review else 'plot_only'));return limits

def a03(out,review):
 f=canvas('A03  |  Hold out space and phases',review);text(f,.06,.84,'Spatial ownership',21);text(f,.94,.84,'Schematic · not to scale',14,GREY,'right')
 box(f,.065,.475,.28,.30,BLUE,'#0B1628','dashed');box(f,.135,.525,.14,.15,WHITE,'#273548')
 text(f,.205,.729,'Observed input context',15,GREY,'center');text(f,.205,.622,'Scored core',19,WHITE,'center');text(f,.205,.575,'64 h⁻¹ Mpc',16,WHITE,'center');text(f,.205,.443,'Labels used only in owned cores',14,GREY,'center')
 box(f,.40,.50,.21,.235,GREY,'#11161D','dashed');text(f,.505,.655,'Held-out region',18,WHITE,'center');text(f,.505,.605,'Disjoint scored cores',14,GREY,'center');text(f,.505,.555,'Development fold 0',15,WHITE,'center')
 for y,s,size,col in [(.735,'Blocked split · rotation 0',18,WHITE),(.68,'Training folds 2, 3, 4',17,WHITE),(.628,'Validation fold 1',17,WHITE),(.576,'Development fold 0',17,WHITE),(.513,'Context may overlap;',14,GREY),(.475,'supervised ownership does not.',14,GREY)]:text(f,.67,y,s,size,col)
 stages=[(.06,.195),(.30,.195),(.54,.195),(.78,.16)]
 for i,(x,w) in enumerate(stages):
  box(f,x,.213,w,.165,GOLD if i==3 else DIM,style='dashed' if i==3 else 'solid')
  if i<3:arrow(f,(x+w+.006,.295),(stages[i+1][0]-.006,.295),i==2)
 for x,head,body in [(.1575,'Five training phases','ph000, ph002–005'),(.3975,'Leave one phase out','Out-of-fold summaries'),(.6375,'Fit → freeze posterior','ph006 selects / audits'),(.86,'Fresh phase','Still required')]:text(f,x,.327,head,16,WHITE,'center');text(f,x,.264,body,14,GOLD if x==.86 else GREY,'center')
 text(f,.06,.16,'ph006: folds 0–1 reserved for calibration; folds 2–4 for evaluation. Already exposed.',14,GREY)
 text(f,.06,.112,'Many spatial cores ≠ independent simulation phases',18)
 text(f,.06,.062,'Original ph001 confirmation belongs to original P12-A; it does not confirm the halo48 refit.',14,GREY)
 export(f,out,'A03_protocol_'+('review' if review else 'plot_only'))

def a04(out,review):
 f=canvas('A04  |  Frozen per-galaxy inference pipeline',review);text(f,.06,.845,'DEPLOYMENT  ·  frozen U-PATCH + P12-A halo48',18)
 for x,w in [(.06,.24),(.35,.23),(.63,.31)]:box(f,x,.58,w,.205)
 for x,y,s,size,col in [(.18,.744,'Three input channels',17,WHITE),(.18,.692,'Galaxy counts',16,GREY),(.18,.651,'Apodized exposure',16,GREY),(.18,.610,'Log count / expectation',16,GREY),(.465,.738,'U-PATCH encoder',20,WHITE),(.465,.678,'3-D U-Net → galaxy head',15,GREY),(.465,.619,'Full five-phase fit · halo48',14,GREY),(.785,.738,'Three eigenvalue summaries',17,WHITE),(.785,.677,'+ redshift, expected density, cap',15,GREY),(.785,.631,'+ random-support boundary distance',14,GREY)]:text(f,x,y,s,size,col,'center')
 arrow(f,(.306,.685),(.344,.685));arrow(f,(.586,.685),(.624,.685))
 box(f,.67,.265,.27,.21,BLUE);text(f,.805,.427,'Conditional FMPE',21,WHITE,'center');text(f,.805,.365,'7 conditioning features',17,GREY,'center');text(f,.805,.306,'Frozen posterior · 20260924 v1',14,GREY,'center');arrow(f,(.805,.574),(.805,.481));text(f,.827,.528,'Condition',14,GREY)
 box(f,.355,.265,.265,.21);text(f,.4875,.427,'Joint eigenvalue draws',18,WHITE,'center');text(f,.4875,.363,'λ₁ ≤ λ₂ ≤ λ₃',25,WHITE,'center');text(f,.4875,.302,'Softplus increments → eigenvalues',14,GREY,'center');arrow(f,(.664,.37),(.626,.37))
 box(f,.06,.265,.245,.21);text(f,.1825,.427,'Class probabilities',20,WHITE,'center');text(f,.1825,.371,'Count λₐ > 0.2 in each draw',15,GREY,'center');text(f,.1825,.316,'Void / Wall / Filament / Cluster',14,WHITE,'center');arrow(f,(.349,.37),(.311,.37))
 box(f,.06,.112,.88,.087,DIM,'#080B10','dashed');text(f,.075,.155,'TRAINING ONLY',14,GREY);text(f,.275,.155,'5 leave-one-phase-out encoders → OOF summaries + targets → fit FMPE',15);arrow(f,(.91,.20),(.91,.259),True)
 text(f,.5,.055,'Provisional per-galaxy inference · R = 7 h⁻¹ Mpc · target z = 0.2 · real-DESI coverage unverified',13,GREY,'center')
 export(f,out,'A04_pipeline_'+('review' if review else 'plot_only'))

def finish(out,limits):
 script=Path(__file__).resolve();shutil.copy2(script,out/'scripts'/script.name)
 diff=subprocess.run(['git','diff','--no-index','--','/dev/null',str(script)],capture_output=True,text=True);assert diff.returncode==1;(out/'scripts/source.patch').write_text(diff.stdout)
 receipt=read(out/'data/receipt.json');theme=read(ROOT/'theme.json');theme['class_palette']=dict(zip(NAMES,COLORS));write(out/'theme.json',theme)
 contact=Image.new('RGB',(1280,3*756),'black');draw=ImageDraw.Draw(contact)
 for i,name in enumerate(['A02_posteriors','A03_protocol','A04_pipeline']):
  draw.text((30,i*756+9),name,fill='white',font_size=22);contact.paste(Image.open(out/'qa'/(name+'_plot_only_720p.png')),(0,i*756+36))
 contact.save(out/'qa/contact_sheet.png')
 m=dict(receipt);m.update(asset_ids=['A02','A03','A04'],intended_slides=['S3','S4','S5'],source_repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),source_dirty_diff='scripts/source.patch: this new renderer only',renderer_sha256=sha(script),common_eigenvalue_limits=limits,display_truncation='Central 95% interval endpoints plus padding; no density or sample cloud truncated',display_thinning='none',selection_seed='not applicable: deterministic posterior-only rule',uncertainty='A02 marginal posterior intervals; A03/A04 not applicable: schematic',masks='A02 chosen rows have random_support=1; A03/A04 no numerical masks',command='python workflows/presentation/tokyo_a02_a04.py --output '+str(out),qa_status='Numerical/export checks pass; previews require visual sign-off',output_hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'})
 write(out/'manifest.json',m)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'a02_a04');ap.add_argument('--render-only',action='store_true');args=ap.parse_args()
 for d in ['static','data/sources','scripts','qa']:(args.output/d).mkdir(parents=True,exist_ok=True)
 if not args.render_only:prepare(args.output)
 setup()
 for review in [False,True]:limits=a02(args.output,review);a03(args.output,review);a04(args.output,review)
 finish(args.output,limits);print('A02–A04 exports complete.',flush=True)
if __name__=='__main__':main()
