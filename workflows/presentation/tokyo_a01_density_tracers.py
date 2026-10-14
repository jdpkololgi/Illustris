#!/usr/bin/env python3
"""A01: matched saved Abacus ph002 R7 matter density and selected mock tracers.
Reuses the geometry chosen before inspection for the preceding tidal illustration.
No inference, field generation, catalogue re-selection, or scheduler action.
"""
import argparse,datetime,hashlib,json,platform,shutil,subprocess
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image,ImageOps
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010')
WHITE='#F5F5F5';GREY='#C8CDD4'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def prepare(out):
 src=ROOT/'tidal_ph002';m=read(src/'manifest.json');path=src/'data/tidal_overlay.npz'
 assert sha(path)==m['output_hashes']['data/tidal_overlay.npz']
 d=np.load(path);row=m['geometry'];assert row['phase']=='ph002' and row['role']=='train'
 target=next(x for x in m['inputs'] if x['path'].endswith('.h5'))
 assert sha(target['path'])==target['sha256']
 with h5py.File(target['path']) as f:
  assert f.attrs['smoothing_count']==1 and f.attrs['smoothing_mpc_h']==7
  assert f.attrs['pair_id']==m['pair_id']
  rho=np.asarray(f['rho_joint'][16:48,16:32,24])
 assert rho.shape==(32,16) and np.isfinite(rho).all() and (rho>0).all()
 trace_error=float(np.max(abs(rho-1-np.trace(d['tensor'],axis1=-2,axis2=-1))))
 assert trace_error<2e-6
 # A tiny 64x32x2 hyperslab of the original response field, bound by its receipt.
 srec=m['response_source'];sp=Path(srec['path'])
 assert sp.stat().st_size==srec['bytes'] and sp.stat().st_mtime_ns==srec['mtime_ns']
 start=np.asarray(row['center'])-16
 slab=(slice(start[0],start[0]+64),slice(start[1],start[1]+32),slice(start[2]+16,start[2]+18))
 with h5py.File(sp) as f:support_raw=np.asarray(f['support_random'][slab])
 assert np.array_equal(support_raw.reshape(32,2,16,2,2).mean((1,3,4)),d['support'])
 projected_support=support_raw.mean(axis=2)
 # Match the earlier per-galaxy mask arithmetic without removing any row.
 rawcell=float(row['grid']['cell_mpc_h']);xyz=d['xyz'];lower=d['lower']
 ij=np.floor((xyz-np.array([lower[0],lower[1],float(d['zlo'])]))/rawcell).astype(int)
 assert np.array_equal(support_raw[ij[:,0],ij[:,1],ij[:,2]].astype(bool),d['supported'])
 assert len(xyz)==718 and len(np.unique(d['targetid']))==718
 np.savez_compressed(out/'data/A01_density_tracers.npz',rho_over_mean=rho,xy=d['xy'],xyz=xyz,targetid=d['targetid'],supported=d['supported'],support_raw=support_raw,projected_support=projected_support,extent=m['xy_bounds_mpc_h'],z_bounds=m['slice_z_bounds_mpc_h'])
 shutil.copy2(src/'manifest.json',out/'data/tidal_source_manifest.json')
 meta={'asset_id':'A01','intended_slide':'S2','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'simulation':m['simulation'],'phase':'ph002','epoch_z':.2,'scientific_status':'saved simulation truth and actual selected mock galaxies; not schematic or inference',
 'geometry':row,'geometry_selection':m['geometry_selection'],'coordinates':m['coordinate_definition'],'xy_bounds_mpc_h':m['xy_bounds_mpc_h'],'slice_z_bounds_mpc_h':m['slice_z_bounds_mpc_h'],
 'density_statistic':'saved rho_joint / mean density after one Gaussian R7 smoothing and 2x2x2 cell averaging; exact match to 1+trace(saved normalized tidal tensor)',
 'smoothing_mpc_h':7.,'density_cell_mpc_h':m['cell_mpc_h'],'slice_thickness_mpc_h':m['slice_thickness_mpc_h'],'support_cell_mpc_h':rawcell,'density_trace_max_abs_error':trace_error,
 'galaxy_selection':m['selection'],'galaxies_displayed':718,'parent_bright_galaxies_in_same_slab':m['parent_slab_count'],'galaxies_outside_raw_random_support_retained':int((~d['supported']).sum()),
 'bias_interpretation':'These are HOD-selected BGS Bright observed-success tracers, not a random subsample of matter; this slice does not measure a bias coefficient or establish completeness.',
 'support_definition':'Existing observer-coordinate random-support field sampled on the displayed chart; projected fraction over the two native depth cells. Dense hatch: neither cell supported. Dotted hatch: one of two supported. No hatch: both supported.',
 'support_counts':{'native_xy_columns':int(projected_support.size),'absent':int((projected_support==0).sum()),'partial':int(((projected_support>0)&(projected_support<1)).sum()),'full':int((projected_support==1).sum())},
 'support_caveat':'This is a finite-random-support diagnostic, not a density mask or an exact real-space completeness map. Galaxies are placed at true mock real-space positions; the observed-coordinate response is not de-redshifted. All selected galaxies remain plotted, including 11 outside raw support.',
 'source_inputs':[{'path':str(path),'sha256':sha(path)},{'path':str(src/'manifest.json'),'sha256':sha(src/'manifest.json')},target],
 'response_source':srec,'source_catalogues':m['large_catalogues'],'catalogue_access_this_render':'none: reuse exact previously hash-verified TARGETID join',
 'allowed_claim':'The hidden smoothed matter field is sampled sparsely and selectively by the observed-success mock galaxy population. Hatching marks random-support limitations, not absent matter.',
 'limits':['One exposed training-region illustration, not posterior validation.','The 2D slice is not the full 3D model conditioning volume.','No survey-bias measurement, missing-galaxy reconstruction, or eigenvalue posterior is shown.'],
 'new_field_generation':False,'new_inference':False,'display_thinning':'none','selection_seed':'not applicable: reuse prior deterministic geometry','checkpoint_model_id':'not applicable: saved simulation truth','draw_count_solver':'not applicable','uncertainty':'not applicable',
 'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'threads':1,'scheduler_actions':'none'}}
 write(out/'data/receipt.json',meta)
 print(json.dumps({'rho_range':[float(rho.min()),float(rho.max())],'galaxies':718,'support':meta['support_counts'],'trace_error':trace_error}),flush=True)
def render(out):
 d=np.load(out/'data/A01_density_tracers.npz');rho=d['rho_over_mean'];xy=d['xy'];support=d['projected_support'];extent=d['extent'];m=read(out/'data/receipt.json')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'text.color':WHITE,'axes.labelcolor':WHITE,'xtick.color':GREY,'ytick.color':GREY,'axes.edgecolor':'#77818E','savefig.facecolor':'black','pdf.fonttype':42,'svg.fonttype':'none','hatch.linewidth':.65})
 vmax=float(np.ceil(rho.max()));assert vmax==4
 for review in [False,True]:
  f=plt.figure(figsize=(1920/144,1080/144),dpi=144,facecolor='black')
  if review:f.text(.055,.94,'A01  |  The hidden field and its selected tracers',fontsize=28,va='center')
  axes=[]
  for n,left in enumerate([.075,.565]):
   ax=f.add_axes([left,.40,.39,.40],facecolor='#101319');axes.append(ax)
   ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal')
   ax.set_title(['Smoothed matter density','Sparse, selected mock galaxies'][n],fontsize=21,loc='left',pad=20)
   ax.set_xlabel('x [h⁻¹ Mpc]',fontsize=18,labelpad=10)
   if n==0:ax.set_ylabel('y [h⁻¹ Mpc]',fontsize=18,labelpad=9)
   else:ax.tick_params(labelleft=False)
   for axis,bounds in [(ax.xaxis,extent[:2]),(ax.yaxis,extent[2:])]:
    ticks=MaxNLocator(nbins=4,prune='both').tick_values(*bounds);axis.set_ticks(ticks[(ticks>bounds[0])&(ticks<bounds[1])])
   ax.tick_params(labelsize=16,pad=7)
   if n==0:
    im=ax.imshow(rho.T,origin='lower',extent=extent,interpolation='bilinear',cmap='viridis',norm=Normalize(0,vmax))
   else:
    cell=m['support_cell_mpc_h']
    for i,j in np.argwhere(support<1):
     missing=support[i,j]==0
     ax.add_patch(Rectangle((extent[0]+i*cell,extent[2]+j*cell),cell,cell,facecolor='#30343A' if missing else '#20242B',edgecolor='#949BA5' if missing else '#676E79',linewidth=0,hatch='////' if missing else '..',zorder=1))
    ax.scatter(xy[:,0],xy[:,1],s=9,c=WHITE,edgecolors='black',linewidths=.3,zorder=3)
  cax=f.add_axes([.075,.265,.39,.025]);cb=f.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,1,2,3,4]);cb.ax.tick_params(labelsize=15,pad=5);cb.set_label(r'$\rho_R / \bar{\rho} = 1 + \delta_R$  (dimensionless)',fontsize=17,labelpad=10)
  f.legend(handles=[Line2D([0],[0],marker='o',lw=0,color=WHITE,markersize=7,label='718 selected mock galaxies')],loc='center',bbox_to_anchor=(.76,.284),frameon=False,fontsize=17)
  f.legend(handles=[Patch(facecolor='#30343A',edgecolor='#949BA5',hatch='////',label='No random support'),Patch(facecolor='#20242B',edgecolor='#949BA5',hatch='..',label='Partial support')],loc='center',bbox_to_anchor=(.76,.222),ncol=2,frameon=False,fontsize=14,handlelength=1.8,columnspacing=1.4)
  f.text(.76,.172,'Hatching is a support diagnostic, not absent matter',fontsize=13,color=GREY,ha='center')
  f.text(.5,.116,'Same real-space slab · true mock positions · observed-success BGS selection',fontsize=15,color=GREY,ha='center')
  f.text(.5,.062,'Abacus ph002 · z = 0.2 · R = 7 h⁻¹ Mpc · density cell / slab = 6.766 h⁻¹ Mpc',fontsize=14,color=GREY,ha='center')
  f.canvas.draw();r=f.canvas.get_renderer();w,h=f.canvas.get_width_height()
  from matplotlib.text import Text
  for t in f.findobj(Text):
   if t.get_visible() and t.get_text():
    b=t.get_window_extent(r);assert b.x0>=0 and b.y0>=0 and b.x1<=w and b.y1<=h,(t.get_text(),b.bounds)
  stem='A01_density_tracers_'+('review' if review else 'plot_only')
  for ext in ['png','pdf','svg']:f.savefig(out/'static'/(stem+'.'+ext),dpi=288 if ext=='png' else 144)
  plt.close(f);image=Image.open(out/'static'/(stem+'.png')).convert('RGB');assert image.size==(3840,2160)
  preview=image.resize((1280,720),Image.Resampling.LANCZOS);preview.save(out/'qa'/(stem+'_720p.png'));ImageOps.grayscale(preview).save(out/'qa'/(stem+'_grayscale.png'))
  image.resize((1920,1080),Image.Resampling.LANCZOS).save(out/'static'/(stem+'_1080p.png'))
  assert b'/FontFile2' in (out/'static'/(stem+'.pdf')).read_bytes()
 m.update(colormap='viridis',density_color_limits=[0,vmax],display_interpolation='bilinear interpolation of saved 32x16 density cells for display only; no new physical smoothing; support and points use native positions',
  command='python workflows/presentation/tokyo_a01_density_tracers.py --output '+str(out),source_repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),renderer_sha256=sha(__file__),qa_status='Numerical and export checks pass; visual inspection pending')
 script=Path(__file__).resolve();shutil.copy2(script,out/'scripts'/script.name)
 diff=subprocess.run(['git','diff','--no-index','--','/dev/null',str(script)],capture_output=True,text=True);assert diff.returncode==1;(out/'scripts/source.patch').write_text(diff.stdout)
 m['source_dirty_diff']='scripts/source.patch: new A01 renderer only';m['output_hashes']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'};write(out/'manifest.json',m)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'a01_ph002');p.add_argument('--render-only',action='store_true');a=p.parse_args()
 for name in ['data','scripts','static','qa']:(a.output/name).mkdir(parents=True,exist_ok=True)
 if not a.render_only:prepare(a.output)
 render(a.output)
if __name__=='__main__':main()
