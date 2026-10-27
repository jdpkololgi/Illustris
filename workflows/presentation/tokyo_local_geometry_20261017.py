#!/usr/bin/env python3
"""Illustrative local environment geometry; actual saved posterior probabilities."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,shutil,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon,FancyArrowPatch
ROOT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010')
OUT=ROOT/'filament_and_local_geometry_20261017'
WHITE,GREY,GOLD='#F5F5F5','#B9C2D0','#F2C14E'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def project(p):
 angle=np.deg2rad(30);rot=np.array([[np.cos(angle),-np.sin(angle),0],[.44*np.sin(angle),.44*np.cos(angle),.898]])
 return np.asarray(p)@rot.T

def geometry(ax,kind):
 rng=np.random.default_rng(202610170+kind);color=['#9CF6D5','#5B8FF9'][kind]
 # Same arbitrary viewing angle; neither orientation nor scale is inferred.
 corners=np.array(list(itertools.product([-1,1],repeat=3)))
 for i,a in enumerate(corners):
  for b in corners[i+1:]:
   if np.sum(a!=b)==1:
    q=project([a,b]);ax.plot(q[:,0],q[:,1],color='#46505E',lw=.8,alpha=.6,zorder=0)
 if kind==0:
  points=rng.uniform(-1,1,(75,3));points=points[(np.linalg.norm(points,axis=1)>.9)&(np.linalg.norm(project(points),axis=1)>.5)]
  # Leave a few tracers in the underdense interior; void is not absolutely empty.
  points=np.vstack((points,rng.uniform(-.35,.35,(4,3))))
  q=project(points);ax.scatter(q[:,0],q[:,1],s=4,c=color,alpha=.55,lw=0)
  t=np.linspace(0,2*np.pi,200)
  # Equatorial guide marks the illustrative underdense interior, not a measured boundary.
  q=project(np.column_stack((.9*np.cos(t),.9*np.sin(t),np.zeros(len(t)))))
  ax.plot(q[:,0],q[:,1],ls=(0,(3,4)),lw=1,color=color,alpha=.45)
 else:
  q=project([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])
  ax.add_patch(Polygon(q,facecolor=color,edgecolor=color,lw=1,alpha=.18))
  points=np.column_stack((rng.uniform(-1,1,300),rng.uniform(-1,1,300),rng.normal(0,.035,300)))
  q=project(points);ax.scatter(q[:,0],q[:,1],s=5,c=color,alpha=.8,lw=0)
 ax.scatter([0],[0],s=65,facecolors='none',edgecolors=GOLD,lw=1.4,zorder=5)
 ax.scatter([0],[0],s=10,c=GOLD,zorder=6)
 ax.set(xlim=(-1.6,1.6),ylim=(-1.6,1.6),aspect='equal');ax.axis('off')
 return points

def main():
 for d in ('data','scripts'):(OUT/d).mkdir(parents=True,exist_ok=True)
 previous=ROOT/'local_environments_selection_20261017';m=json.loads((previous/'MANIFEST.json').read_text())
 p=previous/'data/local_posterior.npz';assert sha(p)==m['files']['data/local_posterior.npz']
 with np.load(p) as d:xy=d['observed_xyz_relative_mpc'][:,:2];prob=d['class_probabilities'];draws=d['draws']
 np.testing.assert_array_equal(np.bincount((draws>.2).sum(-1),minlength=4)/512,prob)
 plt.rcParams.update({'font.family':'DejaVu Sans','text.color':WHITE,'pdf.fonttype':42,'svg.fonttype':'none'})
 f=plt.figure(figsize=(16,9),dpi=120,facecolor='black');f.text(.055,.94,'Possible local environments',fontsize=26)
 f.text(.72,.83,'Same observed galaxy context',fontsize=22,ha='center')
 shapes=[]
 for k,y in enumerate([.54,.18]):
  f.add_artist(Rectangle((.055,y),.36,.31,transform=f.transFigure,facecolor='#0B0E14',edgecolor='#46505E',zorder=0))
  color=['#9CF6D5','#5B8FF9'][k]
  f.text(.075,y+.252,['Void','Wall'][k],fontsize=24,color=color);f.text(.395,y+.252,f'{100*prob[k]:.1f}%',ha='right',fontsize=24,color=color)
  ax=f.add_axes([.095,y+.035,.275,.20],facecolor='#0B0E14');shapes.append(geometry(ax,k))
  f.text(.235,y+.012,['Underdense region','Sheet-like structure'][k],ha='center',fontsize=14,color=GREY)
  f.add_artist(FancyArrowPatch((.435,y+.15),(.565,.60 if k==0 else .43),transform=f.transFigure,arrowstyle='-|>',mutation_scale=22,color=GREY,lw=2))
 ax=f.add_axes([.585,.275,.345,.49],facecolor='#0B0E14');ax.scatter(xy[:,0],xy[:,1],s=7,c=WHITE,alpha=.8,lw=0)
 ax.scatter([0],[0],s=170,facecolors='none',edgecolors=GOLD,lw=2);ax.scatter([0],[0],s=28,c=GOLD)
 ax.set(xlim=(-40,40),ylim=(-40,40),aspect='equal',xticks=[],yticks=[])
 for s in ax.spines.values():s.set_color('#596270')
 ax.plot([15,35],[-34,-34],color=WHITE,lw=2);ax.text(25,-31,'20 Mpc',ha='center',color=WHITE,fontsize=12)
 f.text(.755,.225,'Posterior for the highlighted galaxy',ha='center',fontsize=17,color=GOLD)
 f.text(.755,.17,'Actual observed neighbours · ph006 mock',ha='center',fontsize=14,color=GREY)
 f.text(.5,.095,'Local geometry illustrations — not reconstructed fields or inferred orientations',ha='center',fontsize=17,color=GREY)
 f.text(.5,.045,'Probabilities from 512 saved P12-A eigenvalue draws · λₜₕ = 0.2',ha='center',fontsize=14,color=GREY)
 for ext in ('png','pdf','svg'):f.savefig(OUT/('local_environment_illustrations.'+ext),dpi=240 if ext=='png' else 120,facecolor='black')
 plt.close(f)
 np.savez_compressed(OUT/'data/illustrative_geometry.npz',void_points=shapes[0],wall_points=shapes[1],arbitrary_view_angle_degrees=30)
 shutil.copy2(p,OUT/'data/local_posterior.npz');shutil.copy2(previous/'data/local_provenance.json',OUT/'data/local_provenance.json')
 shutil.copy2(__file__,OUT/'scripts'/Path(__file__).name)
 print('Local geometry figure complete: conceptual geometry, actual probabilities and observed context')
if __name__=='__main__':main()
