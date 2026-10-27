#!/usr/bin/env python3
"""Talk exports: conceptual degeneracy and saved conditional field draws.
Small saved-array extraction/rendering only; no sampling, FFT, or scheduler work.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import json, hashlib, shutil, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
REPO=Path(__file__).resolve().parents[2]
OUT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010/many_fields_20261017')
BASE=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
ROOT=BASE/'wholefield_pilot_20261013_v1'
PAIR='ph012_NGC_s0_interior_00'
DRAW_IDS=[0,1,2]  # Fixed before inspecting any field.
CELL=6.766
WHITE,GREY='#F5F5F5','#B9C2D0'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def bind(rec):
 assert sha(rec['path'])==rec['sha256'],rec['path']
 return rec

def prepare():
 dest=ROOT/'evaluation/unet_x_17/step9984/ph012'/PAIR/'sampled/fullcap/nfe128'
 binding=read(dest/'BINDING.json');complete=read(dest/'COMPLETE.json')
 assert complete['passed'] and complete['binding']==sha(dest/'BINDING.json')
 assert binding['coarse']=='sampled' and not binding['truth_is_input'] and binding['pair']==PAIR
 geometry=read(ROOT/'data/ph012/COMPLETE.json')
 bind(geometry['geometry'])
 row=next(r for r in geometry['pairs'] if r['pair_id']==PAIR)
 assert row['role']=='development' and row['phase']=='ph012'
 origin=np.array(row['grid']['origin_mpc_h'])+3.383*(np.array(row['center'])-16)
 selection=BASE/'calibration_investigation_20261006_v1/b1b2/prepare/ph012.json'
 s=read(selection);case=next(r for r in s['cases'] if r['pair_id']==PAIR)
 bind(case['output']);bind(case['condition'])
 np.testing.assert_allclose(origin,case['origin_mpc_h'],rtol=0,atol=1e-10)
 # Bound observation receipt links expanded and original conditioning.
 obsreceipt=ROOT/'data/ph012'/PAIR/'OBSERVATIONS.json'
 obs=read(obsreceipt);bind(obs['output']);bind(obs['old_condition']);bind(obs['raw_receipt'])
 assert obs['passed'] and not obs['contains_matter']
 assert obs['old_condition']['sha256']==case['condition']['sha256']
 with np.load(case['output']['path']) as f:
  xyz=f['xyz'];ids=f['targetid'];flat=f['flat_index'];support=f['support']
  assert np.unique(ids).size==len(ids)
  ij=np.floor((xyz-origin)/CELL).astype(int)
  expected=(ij[:,0]//16)*4096+np.ravel_multi_index((ij[:,0]%16,ij[:,1],ij[:,2]),(16,16,16))
  np.testing.assert_array_equal(flat,expected)
  np.testing.assert_array_equal(np.bincount(flat,minlength=8192),f['cell_counts'].reshape(-1))
 # Single central saved z cell of the two owned cores, same slab for points.
 mask=(ij[:,2]==8)
 xy=(xyz[mask,:2]-origin[:2]);selected_ids=ids[mask]
 assert ((xy>=0)&(xy<np.array([32,16])*CELL)).all()
 support_slice=support.reshape(32,16,16)[:,:,8]
 assert (support_slice>0).all(), 'Do not omit a missing-support mask in this chosen slab'
 draws=[];sources=[]
 for i in DRAW_IDS:
  p=dest/f'draw{i:03d}.npz'
  with np.load(p) as f:
   assert f['rho'].shape==(96,64,64)
   rho=f['rho'][32:64,24:40,32].copy()
  assert rho.shape==(32,16) and np.isfinite(rho).all() and (rho>0).all()
  draws.append(rho);sources.append(dict(path=str(p),sha256=sha(p)))
 fields=np.array(draws)
 assert all(np.any(fields[i]!=fields[j]) for i in range(3) for j in range(i))
 np.savez_compressed(OUT/'data/simulation.npz',rho_over_mean=fields,xy=xy,targetid=selected_ids,
                     draw_indices=DRAW_IDS,origin_mpc_h=origin,extent=[0,32*CELL,0,16*CELL],
                     z_bounds_mpc_h=origin[2]+np.array([8,9])*CELL,support=support_slice)
 meta=dict(pair=PAIR,phase='ph012',role='exposed development',architecture='U-Net clean endpoint',
   seed=17,step=9984,nfe=128,draw_ids=DRAW_IDS,draw_selection='First three saved draws; chosen before field inspection',
   geometry_selection='ph012, NGC, lowest shell, interior pair 00; central owned-core z cell 8; fixed before field inspection',
   binding=binding,source_binding_sha256=sha(dest/'BINDING.json'),completion_sha256=sha(dest/'COMPLETE.json'),
   sources=sources,selection=case,observations=obs,geometry=row,
   galaxy_count=int(mask.sum()),all_selected_galaxies=int(len(ids)),zero_support_excluded_in_parent=case['zero_support_excluded'],
   point_definition='Actual observed-success mock positions in observed redshift-space Cartesian frame, not jittered cell centers',
   field_definition='Saved R7-smoothed rho/mean on 6.766 Mpc/h cells; single cell-thick xy slice',
   point_field_alignment='Same observer chart and spatial slab. Redshift-space tracers need not sit on real-space matter peaks.',
   target_epoch=0.2,smoothing_mpc_h=7,cell_mpc_h=CELL,origin_mpc_h=origin.tolist(),
   conditioning='Same complete 3-D joint and wide observed channels for all draws; displayed dots show only this slab',
   posterior_qualified=False,new_inference=False,new_draws=False,
   limitations=['Exploratory learned conditional field samples; not demonstrated calibrated joint posterior',
    'Finite multiwindow fields, not globally coherent whole-survey realizations',
    'Density only: no reconstructed tidal tensor or eigenvectors in this figure',
    'No truth field used to choose the region, draw indices or displayed density range',
    'Original draw files newly hashed; their historical completion receipt did not serialize per-draw hashes',
    'Checkpoint hash inherited from saved binding; checkpoint not re-read; observation and galaxy payload hashes freshly verified'])
 write(OUT/'data/provenance.json',meta)
 return fields,xy,meta

def schematic():
 # Positive hand-authored intensity sketches, not simulation or posterior samples.
 x,y=np.meshgrid(np.linspace(0,2,320),np.linspace(0,1,160),indexing='ij')
 fields=[]
 for i,(shift,width,amp) in enumerate([(-.055,.045,.75),(0,.075,1.),(.065,.055,1.3)]):
  curve=.47+.16*np.sin(3.2*x)+shift*np.sin(2*np.pi*x)
  branch=.18+.28*x+shift*np.cos(4*x)
  rho=.13+amp*np.exp(-.5*((y-curve)/width)**2)
  rho+=(1.0-.15*i)*np.exp(-.5*((y-branch)/(.035+.018*i))**2)
  for xx,yy,aa in [(.38,.61,1.1),(1.12,.43,1.4),(1.7,.66,.7)]:
   rho+=aa*np.exp(-((x-xx)**2+(y-yy)**2)/(.018+.007*i))
  fields.append(rho)
 rng=np.random.default_rng(20261017);xg=rng.uniform(.06,1.94,80)
 yg=np.where(rng.random(80)<.65,.47+.16*np.sin(3.2*xg),.18+.28*xg)+rng.normal(0,.045,80)
 xy=np.column_stack((xg,np.clip(yg,.04,.96)))
 np.savez_compressed(OUT/'data/schematic.npz',illustrative_intensity=fields,xy=xy,seed=20261017)
 return np.array(fields),xy

def render(fields,xy,stem,simulation=False,meta=None):
 extent=[0,32*CELL,0,16*CELL] if simulation else [0,2,0,1]
 vmax=float(np.ceil(fields.max())) if simulation else float(fields.max())
 fig=plt.figure(figsize=(16,9),dpi=120,facecolor='black')
 fig.text(.055,.968,'Exploratory density-field draws' if simulation else 'Possible underlying fields',fontsize=23,va='top')
 fig.text(.605,.755,'One observed galaxy set',fontsize=23)
 for i,y in enumerate([.665,.375,.085]):
  ax=fig.add_axes([.055,y,.29,.258],facecolor='black')
  im=ax.imshow(fields[i].T,origin='lower',extent=extent,cmap='viridis',vmin=0,vmax=vmax,interpolation='bilinear',aspect='equal')
  ax.text(.025,.94,f'Draw {i+1}' if simulation else f'Possibility {i+1}',transform=ax.transAxes,
          va='top',fontsize=15,color=WHITE,bbox=dict(facecolor='black',alpha=.65,edgecolor='none',pad=4))
  ax.set_xticks([]);ax.set_yticks([])
  for sp in ax.spines.values():sp.set_color('#4B5360')
  fig.add_artist(FancyArrowPatch((.365,y+.129),(.585,[.60,.505,.41][i]),transform=fig.transFigure,
                arrowstyle='-|>',mutation_scale=23,color=GREY,lw=2))
 ax=fig.add_axes([.605,.35,.355,.316],facecolor='#0B0E14')
 ax.scatter(xy[:,0],xy[:,1],s=4 if simulation else 14,c=WHITE,linewidths=0,alpha=.9)
 ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',xticks=[],yticks=[])
 for sp in ax.spines.values():sp.set_color('#596270')
 fig.text(.78,.28,f"{len(xy):,} selected mock galaxies" if simulation else 'Sparse and selectively observed',ha='center',fontsize=18)
 fig.text(.78,.22,'Same spatial slice in every panel',ha='center',fontsize=16,color=GREY)
 fig.text(.47,.855,'One set of tracers\ndoes not fix one field',ha='center',va='center',fontsize=17,linespacing=1.5)
 if simulation:
  ax.plot([.74*extent[1],.74*extent[1]+50],[.09*extent[3]]*2,color=WHITE,lw=2)
  ax.text(.74*extent[1]+25,.135*extent[3],'50 h⁻¹ Mpc',color=WHITE,ha='center',fontsize=12)
  cax=fig.add_axes([.055,.052,.29,.012]);cb=fig.colorbar(im,cax=cax,orientation='horizontal')
  cb.set_ticks([0,vmax/2,vmax]);cb.ax.tick_params(labelsize=11,colors=GREY,pad=2)
  fig.text(.37,.049,'ρ / ρ̄',fontsize=15)
  fig.text(.605,.155,'Separate exploratory field model · Abacus ph012',fontsize=15,color=GREY)
  fig.text(.605,.105,'R = 7 h⁻¹ Mpc · slice depth = 6.766 h⁻¹ Mpc',fontsize=14,color=GREY)
  fig.text(.605,.055,'Current P12-A work infers local eigenvalues, not density fields.',fontsize=13,color=GREY)
 else:
  fig.text(.605,.13,'Conceptual illustration',fontsize=15,color=GREY)
  fig.text(.605,.075,'Not simulation data or posterior draws',fontsize=14,color=GREY)
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for t in fig.texts:
  b=t.get_window_extent(renderer);assert b.x0>=0 and b.y0>=0 and b.x1<=1920 and b.y1<=1080,t.get_text()
 for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{stem}.{ext}',dpi=240 if ext=='png' else 120,facecolor='black')
 plt.close(fig)
 im=Image.open(OUT/f'{stem}.png').convert('RGB');assert im.size==(3840,2160)
 im.resize((1280,720),Image.Resampling.LANCZOS).save(OUT/f'{stem}_preview.png')
 assert b'/FontFile2' in (OUT/f'{stem}.pdf').read_bytes()
 assert '<text' in (OUT/f'{stem}.svg').read_text()
 return dict(color_limits=[0,vmax],bilinear_display_interpolation=True,field_values_clipped=0)

def main():
 for d in ['data','scripts']:(OUT/d).mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','text.color':WHITE,'axes.labelcolor':WHITE,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
 fields,xy,meta=prepare()
 display=render(fields,xy,'simulation_posterior',True,meta)
 # Retired illustrative diagram: no further rendering.
 shutil.copy2(__file__,OUT/'scripts'/Path(__file__).name)
 write(OUT/'data/display.json',display)
 print(json.dumps(dict(output=str(OUT),galaxies=meta['galaxy_count'],display=display)))
if __name__=='__main__':main()
