import importlib.util
from pathlib import Path
import numpy as np
p=Path('/global/homes/d/dkololgi/TNG/Illustris/workflows/abacus_tweb/p4_build_spatial_manifest.py')
s=importlib.util.spec_from_file_location('p4',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=np.load('/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/ph012/p4_recovery_diagnostic_20261026/fold_inputs.npz')
g=d['component_counts'];c=d['component_context'];k=d['component_cores']
def metrics(f):
 a=np.array([g[f==i].sum(0) for i in range(5)])
 print('deviation',np.max(abs(a-a.mean(0))/np.maximum(a.mean(0),1)),'active ratio',a.sum((1,2)).max()/a.sum((1,2)).min(),'super ratio',np.bincount(f[d['super_component']]).max()/np.bincount(f[d['super_component']]).min(),flush=True)
for cw,kw in [(0.05,0.05),(0,0.05),(0,0)]:
 f=m.greedy_balanced_folds(g,c,k,context_weight=cw,core_weight=kw)
 print('weights',cw,kw,flush=True);metrics(f)
 # Single whole-component relocation, fixed order, strict objective decrease.
 target=np.maximum(g.sum(0)/5,1)
 a=np.array([g[f==i].sum(0) for i in range(5)],dtype=float)
 v=g/target;u=a/target-1
 for sweep in range(20):
  changes=0
  for j in range(len(g)):
   src=int(f[j]);vec=v[j]
   delta=2*((u-u[src])*vec).sum((1,2))+2*(vec*vec).sum()
   delta[src]=0
   dest=int(np.argmin(delta))
   if delta[dest]<-1e-12:
    f[j]=dest;u[src]-=vec;u[dest]+=vec;changes+=1
  if not changes:break
 print('refined sweeps',sweep+1,flush=True);metrics(f)
