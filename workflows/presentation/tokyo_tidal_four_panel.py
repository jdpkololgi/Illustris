#!/usr/bin/env python3
"""One title-free, matched lambda1/lambda2/lambda3/T-web truth figure."""
import argparse,datetime,hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
DEFAULT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010/tidal_ph002')
COLORS=['#9CF6D5','#5B8FF9','#E85D9E','#F2C14E']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=DEFAULT)
 ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output
 for folder in ['data','qa','scripts']:(out/folder).mkdir(parents=True,exist_ok=True)
 src=args.source/'data/tidal_overlay.npz';provenance=args.source/'manifest.json'
 previous=json.loads(provenance.read_text());assert sha(src)==previous['output_hashes']['data/tidal_overlay.npz']
 d=np.load(src);e=d['eigenvalues'];xy=d['xy'];classes=(e>.2).sum(-1)
 assert e.shape==(32,16,3) and np.isfinite(e).all() and (np.diff(e,axis=-1)>=0).all()
 assert np.array_equal(classes,d['classes']) and len(xy)==718
 assert np.allclose(np.linalg.eigvalsh(d['tensor']),e,rtol=0,atol=1e-12)
 bounds=previous['xy_bounds_mpc_h'];lim=max(.25,float(np.ceil(np.abs(e).max()*4)/4))
 norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
 plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#F5F5F5','axes.labelcolor':'#F5F5F5',
  'xtick.color':'#C8CDD4','ytick.color':'#C8CDD4','axes.edgecolor':'#89919C','font.size':15,
  'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'black'})
 fig=plt.figure(figsize=(1920/144,1080/144),dpi=144,facecolor='black')
 axes=[]
 positions=[(.075,.555),(.53,.555),(.075,.165),(.53,.165)]
 names=['λ₁','λ₂','λ₃','T-web · λₜₕ = 0.2']
 for k,((left,bottom),name) in enumerate(zip(positions,names)):
  ax=fig.add_axes([left,bottom,.36,.32],facecolor='black');axes.append(ax)
  if k<3:im=ax.imshow(e[...,k].T,origin='lower',extent=bounds,interpolation='nearest',cmap='RdBu_r',norm=norm)
  else:ax.imshow(classes.T,origin='lower',extent=bounds,interpolation='nearest',cmap=ListedColormap(COLORS),vmin=-.5,vmax=3.5,alpha=.78)
  ax.scatter(xy[:,0],xy[:,1],s=5.2,c='white',edgecolors='black',linewidths=.3,zorder=3)
  ax.set(xlim=bounds[:2],ylim=bounds[2:],aspect='equal')
  ax.set_title(name,loc='left',fontsize=21,pad=12)
  ax.xaxis.set_major_locator(MaxNLocator(nbins=4,prune="both"));ax.yaxis.set_major_locator(MaxNLocator(nbins=3,prune="both"))
  ax.tick_params(labelsize=15,length=4,pad=5)
  if k<2:ax.tick_params(labelbottom=False)
  elif k==2:ax.set_xlabel('x [h⁻¹ Mpc]',fontsize=17,labelpad=8)
  if k%2==0:ax.set_ylabel('y [h⁻¹ Mpc]',fontsize=17,labelpad=8)
  else:ax.tick_params(labelleft=False)
 cax=fig.add_axes([.931,.28,.014,.49]);cb=fig.colorbar(im,cax=cax)
 cb.ax.tick_params(labelsize=15,length=4,pad=5);cb.set_ticks([-1.5,0.,1.5])
 cb.outline.set_edgecolor('#89919C');cax.set_title('λ',fontsize=19,pad=10)
 fig.text(.916,.525,'Dimensionless',rotation=90,fontsize=14,ha='center',va='center',color='#C8CDD4')
 fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(COLORS,['Void','Wall','Filament','Cluster'])],
  loc='center',bbox_to_anchor=(.71,.082),ncol=4,frameon=False,fontsize=14,
  handlelength=.8,handletextpad=.4,columnspacing=.8)
 fig.text(.075,.026,'Simulation truth · ph002 · 718 mock galaxies · real space · R = 7 h⁻¹ Mpc · cell/slab = 6.766 h⁻¹ Mpc',
  fontsize=13,color='#C8CDD4')
 fig.canvas.draw();r=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
 for t in fig.texts:
  b=t.get_window_extent(r);assert b.x0>=0 and b.y0>=0 and b.x1<=w and b.y1<=h,t.get_text()
 for ax in axes:
  for t in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
   if t.get_visible() and t.get_text():
    b=t.get_window_extent(r);assert b.x0>=0 and b.y0>=0 and b.x1<=w and b.y1<=h,(t.get_text(),b.bounds)
 fig.savefig(out/'tidal_eigenvalues_and_web.png',dpi=288)
 fig.savefig(out/'tidal_eigenvalues_and_web.pdf',dpi=288)
 plt.close(fig)
 im=Image.open(out/'tidal_eigenvalues_and_web.png').convert('RGB');assert im.size==(3840,2160)
 im.resize((1920,1080),Image.Resampling.LANCZOS).save(out/'tidal_eigenvalues_and_web_1080p.png')
 im.resize((1280,720),Image.Resampling.LANCZOS).save(out/'qa/preview_720p.png')
 assert b'/FontFile2' in (out/'tidal_eigenvalues_and_web.pdf').read_bytes()
 shutil.copy2(src,out/'data/tidal_overlay.npz');shutil.copy2(provenance,out/'data/source_manifest.json')
 shutil.copy2(__file__,out/'scripts'/Path(__file__).name)
 m={'asset_id':'tidal_eigenvalues_and_web','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'scientific_status':'saved simulation truth; not model predictions or posterior draws',
  'phase':'ph002','simulation_epoch_z':.2,'source_geometry':previous['geometry'],
  'source_selection_rule':previous['geometry_selection'],'source_inputs':previous['inputs'],
  'source_slice':{'path':str(src),'sha256':sha(src)},'source_manifest':{'path':str(provenance),'sha256':sha(provenance)},
  'panels':['lambda1','lambda2','lambda3','count(lambda_a > 0.2)'],
  'shared_eigenvalue_colour_limits':[-lim,lim],'colormap':'RdBu_r; linear, zero-centred, no clipping',
  'class_labels':['void','wall','filament','cluster'],'class_colours':COLORS,'class_threshold':.2,
  'class_cell_counts':np.bincount(classes.ravel(),minlength=4).tolist(),
  'galaxy_count_each_panel':len(xy),'thinning':'none','support_overlay':'omitted; all galaxies shown identically, including 11 previously marked crosses',
  'coordinates':previous['coordinate_definition'],'galaxy_selection':previous['selection'],
  'xy_bounds_mpc_h':bounds,'z_bounds_mpc_h':previous['slice_z_bounds_mpc_h'],
  'smoothing_mpc_h':7.,'cell_mpc_h':float(d['cell']),'slab_thickness_mpc_h':float(d['cell']),
  'field_statistic':previous['field_statistic'],'uncertainty':'not applicable: simulation truth',
  'draw_count_solver':'not applicable: reuses saved 32x16 tensor slice; no new field generation',
  'display_interpolation':'nearest; native 32x16 field cells; 3840-pixel PNG is a direct high-DPI render',
  'source_script_sha256':sha(__file__),'source_repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
  'command':'python workflows/presentation/tokyo_tidal_four_panel.py --output '+str(out),
  'scheduler_actions':'none','threads':1,
  'checks':{'eigenvalues_match_saved_tensor':True,'classification_matches_saved_classes':True,'all_galaxies_retained':True,
            'shared_geometry':True,'embedded_pdf_fonts':True,'png_pixels':[3840,2160]},
  'output_hashes':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(json.dumps({'shared_limits':[-lim,lim],'cell_counts':m['class_cell_counts'],'galaxies':len(xy)}),flush=True)
if __name__=='__main__':main()
