#!/usr/bin/env python3
"""Saved P12-A local posterior and A01 survey-process slide; no new inference."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,shutil,zipfile
from pathlib import Path
import numpy as np
import fitsio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle
from PIL import Image
ROOT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010')
OUT=ROOT/'local_environments_selection_20261017'
WHITE,GREY,GOLD='#F5F5F5','#B9C2D0','#F2C14E'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def arrow(f,a,b):f.add_artist(FancyArrowPatch(a,b,transform=f.transFigure,arrowstyle='-|>',mutation_scale=22,color=GREY,lw=2))
def canvas():return plt.figure(figsize=(16,9),dpi=120,facecolor='black')
def export(f,name):
 f.canvas.draw();r=f.canvas.get_renderer()
 for t in f.texts:
  b=t.get_window_extent(r);assert b.x0>=0 and b.y0>=0 and b.x1<=1920 and b.y1<=1080,t.get_text()
 for ext in ['png','pdf','svg']:f.savefig(OUT/(name+'.'+ext),dpi=240 if ext=='png' else 120,facecolor='black')
 plt.close(f)
 assert b'/FontFile2' in (OUT/(name+'.pdf')).read_bytes()
 assert '<text' in (OUT/(name+'.svg')).read_text()
def environment_data():
 p=ROOT/'a02_a04/data/A02_posteriors.npz';m=read(ROOT/'a02_a04/manifest.json');assert sha(p)==m['output_hashes']['data/A02_posteriors.npz']
 receipt=read(ROOT/'a02_a04/data/receipt.json');identity=receipt['identities'][0];node=identity['parent_node_id']
 with np.load(p) as f:draws=f['joint_eigenvalue_draws'][0].copy();prob=f['class_probabilities'][0]
 assert draws.shape==(512,3) and np.isfinite(draws).all() and (np.diff(draws,axis=-1)>=0).all()
 classes=(draws>.2).sum(-1);np.testing.assert_array_equal(np.bincount(classes,minlength=4)/512,prob)
 chosen=[]
 for k in (0,1):
  candidates=np.flatnonzero(classes==k);median=np.median(draws[candidates],axis=0)
  chosen.append(int(candidates[np.argmin(np.sum(((draws[candidates]-median)/draws.std(0))**2,axis=-1))]))
 cat=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/p1_canonical')
 cm=read(cat/'CATALOGUE_COMPLETE.json');assert cm['pass'] and cm['geometry']['units']=='Mpc'
 points_path=Path(cm['artifacts']['points']);assert sha(points_path)==cm['artifacts']['points_sha256']
 points=np.load(points_path,mmap_mode='r');center=points[node,:3].copy();selected=[];positions=[]
 # Fixed visual crop, before inspecting point counts/appearance. Not the full encoder context.
 for start in range(0,len(points),250000):
  block=np.asarray(points[start:start+250000]);offset=block[:,:3]-center
  keep=(abs(offset[:,0])<40)&(abs(offset[:,1])<40)&(abs(offset[:,2])<7.5)&(block[:,3]==identity['cap'])
  selected.extend(start+np.flatnonzero(keep));positions.append(offset[keep])
 indices=np.asarray(selected);xyz=np.concatenate(positions)
 assert (indices==node).sum()==1
 parent=cm['canonical_parent'];assert Path(parent['path']).stat().st_size==parent['bytes']
 with fitsio.FITS(parent['path']) as f:
  rows=f[1].read(rows=indices,columns=['TARGETID','RA','DEC','Z','ZWARN'])
  target=f[1].read(rows=[node],columns=['TARGETID','RA','DEC','Z','ZWARN'])[0]
 assert abs(float(target['Z'])-identity['observed_redshift'])<1e-7
 assert len(np.unique(rows['TARGETID']))==len(rows)
 assert np.all(rows['ZWARN']==0) and np.all((rows['Z']>=.1)&(rows['Z']<.6))
 # Verify the target row's sky direction independently against saved Cartesian position.
 ra,dec=np.deg2rad([float(target['RA']),float(target['DEC'])]);direction=np.array([np.cos(ra)*np.cos(dec),np.sin(ra)*np.cos(dec),np.sin(dec)])
 np.testing.assert_allclose(center/np.linalg.norm(center),direction,rtol=0,atol=1e-10)
 np.savez_compressed(OUT/'data/local_posterior.npz',draws=draws,class_probabilities=prob,chosen_draw_indices=chosen,
   observed_xyz_relative_mpc=xyz,parent_rows=indices,targetid=rows['TARGETID'],target_parent_row=node,center_mpc=center)
 meta=dict(identity=identity,targetid=int(target['TARGETID']),observed_redshift=float(target['Z']),
  checkpoint=receipt['posterior_checkpoint_sha256'],model=receipt['model_id'],posterior_source=str(p),posterior_sha256=sha(p),
  points_source=str(points_path),points_sha256=sha(points_path),catalogue_receipt_sha256=sha(cat/'CATALOGUE_COMPLETE.json'),
  parent_catalogue_binding=parent,parent_hash_status='Inherited receipt plus current size; only displayed rows read',
  probabilities=prob.tolist(),draw_indices=chosen,triplets=draws[chosen].tolist(),displayed_galaxies=len(xyz),
  selection='Reuse A02 lower-concentration example, nearest Q25 among 50000 saved evaluation rows; no truth selection',
  draw_selection='Actual saved draw nearest each class-conditional marginal median in full-posterior-SD-scaled Euclidean distance; illustrative stratified selection, not two unselected draws',
  probability_definition='Counts across all 512 joint draws; threshold strictly >0.2',
  displayed_context='Actual same-ph006 observed redshift-space points, relative Planck18 Mpc, x/y +/-40 and z +/-7.5; not full 3D encoder context',
  scope='Local eigenvalue posterior at highlighted galaxy only. No density field, spatial joint posterior, eigenvectors or orientations.',
  sampling='Reused 512 saved draws; no inference or new sampling')
 write(OUT/'data/local_provenance.json',meta)
 return draws,prob,chosen,xyz,meta

def local_figure(draws,prob,chosen,xyz,meta):
 f=canvas();f.text(.055,.93,'Possible local environments',fontsize=25)
 f.text(.65,.83,'Same observed galaxy context',fontsize=22,ha='center')
 colors=['#9CF6D5','#5B8FF9']
 for k,y in enumerate([.54,.19]):
  f.add_artist(Rectangle((.055,y),.36,.30,transform=f.transFigure,facecolor='#0B0E14',edgecolor='#4B5360',lw=1,zorder=0))
  f.text(.075,y+.245,['Void','Wall'][k],fontsize=24,color=colors[k]);f.text(.395,y+.25,f'{100*prob[k]:.1f}%',ha='right',fontsize=23,color=colors[k])
  ax=f.add_axes([.115,y+.055,.265,.145],facecolor='#0B0E14',zorder=2)
  ax.axvline(.2,color=GOLD,ls=(0,(4,3)),lw=1.5)
  vals=draws[chosen[k]]
  for j in range(3):
   ax.plot(vals[j],2-j,marker=['o','s','D'][j],color=colors[k],ms=8)
  ax.set(xlim=(-.45,.35),ylim=(-.6,2.7),yticks=[2,1,0],yticklabels=['λ₁','λ₂','λ₃'],xticks=[-.4,0,.2])
  ax.tick_params(colors=GREY,labelsize=12);ax.tick_params(axis='y',length=0,labelsize=16)
  for s in ['top','left','right']:ax.spines[s].set_visible(False)
  ax.spines['bottom'].set_color('#596270')
  ax.text(.2,2.6,'λₜₕ',color=GOLD,ha='center',fontsize=13)
  f.text(.075,y+.012,f'{k} eigenvalue'+('s' if k!=1 else '')+' above threshold',fontsize=12,color=GREY)
  # Separate arrow endpoints at the observed panel edge.
  arrow(f,(.435,y+.15),(.565,.60 if k==0 else .43))
 ax=f.add_axes([.585,.275,.345,.49],facecolor='#0B0E14')
 ax.scatter(xyz[:,0],xyz[:,1],s=7,c=WHITE,alpha=.8,linewidths=0)
 ax.scatter([0],[0],s=170,facecolors='none',edgecolors=GOLD,linewidths=2)
 ax.scatter([0],[0],s=28,c=GOLD,zorder=4)
 ax.set(xlim=(-40,40),ylim=(-40,40),aspect='equal',xticks=[],yticks=[])
 for s in ax.spines.values():s.set_color('#596270')
 ax.plot([15,35],[-34,-34],color=WHITE,lw=2);ax.text(25,-31,'20 Mpc',ha='center',color=WHITE,fontsize=12)
 f.text(.755,.225,'Posterior for the highlighted galaxy',ha='center',fontsize=17,color=GOLD)
 f.text(.755,.17,'Observed neighbours shown in projection',ha='center',fontsize=14,color=GREY)
 f.text(.055,.095,'One saved eigenvalue triplet per possibility · probabilities from all 512 joint draws',fontsize=15,color=GREY)
 f.text(.055,.045,'P12-A · Abacus ph006 mock · local eigenvalues only · λₜₕ = 0.2',fontsize=14,color=GREY)
 export(f,'local_environment_posterior')

def survey_figure():
 p=ROOT/'a01_ph002/data/A01_density_tracers.npz';m=read(ROOT/'a01_ph002/manifest.json');assert sha(p)==m['output_hashes']['data/A01_density_tracers.npz']
 with np.load(p) as d:rho=d['rho_over_mean'].copy();xy=d['xy'].copy();ext=d['extent'].copy();ids=d['targetid'].copy()
 assert rho.shape==(32,16) and len(xy)==718 and len(np.unique(ids))==718
 f=canvas();f.text(.5,.9,'Train on what the survey can observe',ha='center',fontsize=28)
 for x,title in [(.055,'Simulation'),(.38,'Survey selection'),(.665,'Observed mock')]:f.text(x,.75,title,fontsize=23)
 a=f.add_axes([.055,.395,.275,.245]);im=a.imshow(rho.T,origin='lower',extent=ext,interpolation='bilinear',cmap='viridis',vmin=0,vmax=4);a.set(xticks=[],yticks=[])
 b=f.add_axes([.665,.395,.275,.245],facecolor='#0B0E14');b.scatter(xy[:,0],xy[:,1],s=5,c=WHITE,linewidths=0);b.set(xlim=ext[:2],ylim=ext[2:],aspect='equal',xticks=[],yticks=[])
 for ax in [a,b]:
  for sp in ax.spines.values():sp.set_color('#596270')
 f.add_artist(Rectangle((.38,.365),.235,.30,transform=f.transFigure,facecolor='#0B0E14',edgecolor='#596270',lw=1))
 for y,text in zip([.605,.54,.475,.41],['Galaxy population','Brightness limit','Sky coverage','Redshift success']):f.text(.4975,y,text,fontsize=18,ha='center',color=GREY)
 arrow(f,(.34,.515),(.37,.515));arrow(f,(.625,.515),(.655,.515))
 f.text(.1925,.31,'Underlying matter density',fontsize=16,ha='center')
 f.text(.8025,.31,'718 selected mock galaxies',fontsize=16,ha='center')
 cbax=f.add_axes([.075,.245,.235,.015]);cb=f.colorbar(im,cax=cbax,orientation='horizontal',ticks=[0,2,4]);cb.ax.tick_params(colors=GREY,labelsize=12);cb.set_label('ρ / ρ̄',color=GREY,fontsize=14)
 f.text(.5,.13,'Apply the observing process to the training simulations',ha='center',fontsize=23)
 f.text(.5,.055,'Abacus ph002 · matched real-space slice · R = 7 h⁻¹ Mpc · actual observed-success selection',ha='center',fontsize=13,color=GREY)
 export(f,'survey_selection_training')
 shutil.copy2(p,OUT/'data/A01_density_tracers.npz')
 write(OUT/'data/survey_provenance.json',dict(source=str(p),sha256=sha(p),source_manifest=str(ROOT/'a01_ph002/manifest.json'),source_manifest_sha256=sha(ROOT/'a01_ph002/manifest.json'),
    galaxies=718,interpretation='Source-bound endpoints; central box describes the observing process, not newly simulated cuts. HOD tracers connect matter to galaxies.',
    coordinates='True mock real-space positions reused from A01; observational selection by exact TARGETID, no RSD displacement rendered',
    support='All original selected points retained, including 11 outside raw random support; no hatching drawn',new_selection=False,new_inference=False))

def revised_fields():
 # Render saved extracted slices only; stop all work on the abandoned illustrative diagram.
 import importlib.util
 p=Path(__file__).with_name('tokyo_many_fields_20261017.py')
 spec=importlib.util.spec_from_file_location('saved_fields',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 mod.OUT=OUT
 with np.load(ROOT/'many_fields_20261017/data/simulation.npz') as d:mod.render(d['rho_over_mean'],d['xy'],'exploratory_field_reference',True)
 shutil.copy2(ROOT/'many_fields_20261017/data/simulation.npz',OUT/'data/field_reference.npz')
 shutil.copy2(ROOT/'many_fields_20261017/data/provenance.json',OUT/'data/field_reference_provenance.json')
 shutil.copy2(p,OUT/'scripts'/p.name)

def main():
 for d in ['data','scripts']:(OUT/d).mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','text.color':WHITE,'pdf.fonttype':42,'svg.fonttype':'none'})
 d=environment_data();local_figure(*d);survey_figure();revised_fields()
 shutil.copy2(__file__,OUT/'scripts'/Path(__file__).name)
 print(json.dumps({'output':str(OUT),'context_galaxies':d[-1]['displayed_galaxies'],'probabilities':d[1].tolist()}))
if __name__=='__main__':main()
