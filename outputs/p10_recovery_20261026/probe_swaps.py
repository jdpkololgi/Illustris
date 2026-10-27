exec(open('/global/homes/d/dkololgi/TNG/Illustris/outputs/p10_recovery_20261026/probe_refinement.py').read().split('for cw,kw in ')[0])
f=m.greedy_balanced_folds(g,c,k,context_weight=0,core_weight=.05)
target=np.maximum(g.sum(0)/5,1);v=g/target
u=np.array([v[f==i].sum(0)-1 for i in range(5)])
for step in range(50):
 best=(-1e-12,-1,-1)
 for i in range(len(g)):
  delta=v-v[i]
  scores=2*np.sum((u[f[i]]-u[f])*delta+delta*delta,axis=(1,2))
  scores[f==f[i]]=0
  j=int(np.argmin(scores))
  if scores[j]<best[0]:best=(float(scores[j]),i,j)
 if best[1]<0:break
 _,i,j=best;a,b=int(f[i]),int(f[j]);delta=v[j]-v[i]
 u[a]+=delta;u[b]-=delta;f[i],f[j]=b,a
 if step<5 or step%10==0:print('swap',step,best,flush=True);metrics(f)
print('FINAL',step,flush=True);metrics(f)
np.save('/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/ph012/p4_recovery_diagnostic_20261026/swap_assignment.npy',f)
