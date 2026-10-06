"""Read-only audit of galaxy inference coverage inside atlas display seam slabs."""
import argparse,hashlib,json,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def seam_mask(xyz,cap,tile_index,plan):
 """Union of half-display-cell bands beside existing face-neighbour tiles.

 This is a geometric seam-slab membership, not a sky projection or assertion
 that a threshold surface ought to exist at every member's position.
 """
 stride=np.asarray(plan['ownership_raw_stride']);mask=np.zeros(len(xyz),bool)
 look={(t['cap'],*t['key']):i for i,t in enumerate(plan['tiles'])}
 for name,capid in [('NGC',1),('SGC',0)]:
  rows=np.flatnonzero(cap==capid);g=plan['grids'][name];raw=(xyz[rows]-g['origin_mpc_h'])/g['cell_mpc_h']
  keys=np.floor(raw/stride).astype(int);local=(raw-keys*stride)*g['cell_mpc_h'];extent=stride*g['cell_mpc_h']
  expected=np.array([look[(name,*k)] for k in keys],dtype=np.int64);assert np.array_equal(expected,tile_index[rows])
  neighbours=np.zeros((len(plan['tiles']),3,2),bool)
  for tid,t in enumerate(plan['tiles']):
   if t['cap']!=name:continue
   for j in range(3):
    for side,shift in enumerate([-1,1]):
     k=list(t['key']);k[j]+=shift;neighbours[tid,j,side]=(name,*k) in look
  near=((local<6.766)&neighbours[expected,:,0])|((local>extent-6.766)&neighbours[expected,:,1])
  mask[rows]=near.any(1)
 return mask

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert os.environ.get('SLURM_JOB_ID'),'compute allocation required';a.out.mkdir(parents=True,exist_ok=False)
 plan=json.loads((a.root/'PLAN.json').read_text());receipt=json.loads((a.root/'ATLAS_COMPLETE.json').read_text())
 assert receipt['passed'] and receipt['tiles']==len(plan['tiles'])
 assert sha(a.root/'vac_properties.npz')==plan['properties_sha256']
 assert sha(receipt['output']['path'])==receipt['output']['sha256']
 with np.load(a.root/'vac_properties.npz') as f:xyz=f['xyz'];cap=f['CAP'];ids=f['TARGETID']
 with np.load(receipt['output']['path']) as f:fields={k:f[k] for k in ['p_web','rho_mean','rho_sd','support','field_eligible','tile_index','classes02']}
 n=len(ids);assert n==receipt['rows'];assert len(np.unique(ids))==n
 seam=seam_mask(xyz,cap,fields['tile_index'],plan);seen=np.zeros(n,'u1');finite_draws=np.zeros(n,bool)
 def read(item):
  tid,t=item;path=a.root/'tiles'/t['id']/'COMPLETE.json';r=json.loads(path.read_text());source=r['outputs'][0]
  assert sha(path)==receipt['receipts'][tid]['sha256'];assert sha(source['path'])==source['sha256']
  with np.load(source['path']) as f:rows=f['rows'];density=f['galaxy_density'];classes=f['classes02']
  assert np.array_equal(classes,fields['classes02'][:,rows]);assert np.allclose(density.mean(0),fields['rho_mean'][rows],rtol=2e-6,atol=1e-7)
  return rows,np.isfinite(density).all(0)&(density>0).all(0)
 with ThreadPoolExecutor(max_workers=16) as pool:
  for i,(rows,finite) in enumerate(pool.map(read,enumerate(plan['tiles']))):
   seen[rows]+=1;finite_draws[rows]=finite
   if i%300==0:print('verified tiles',i,flush=True)
 assert np.all(seen==1)
 valid_probability=np.isfinite(fields['p_web']).all(1)&np.isclose(fields['p_web'].sum(1),1)&(fields['p_web']>=0).all(1)&(fields['p_web']<=1).all(1)
 assert finite_draws.all() and valid_probability.all(),'incomplete or invalid galaxy inference'
 result={'schema':'loa-seam-coverage-audit-v1','passed':True,'job':os.environ['SLURM_JOB_ID'],'script_sha256':sha(__file__),'plan_sha256':sha(a.root/'PLAN.json'),'marginals_sha256':receipt['output']['sha256'],'draws':receipt['draws'],'seam_definition':'Within 6.766 Mpc/h of an ownership face with an existing face-neighbour tile; union across axes. Potential 13.532 Mpc/h rendering gaps, independent of density threshold.','caps':{},'unique_targetids':True,'exact_once_tile_ownership':True}
 for name,capid in [('NGC',1),('SGC',0),('ALL',None)]:
  base=np.ones(n,bool) if capid is None else cap==capid;stats={}
  for label,select in [('all',base),('seam_slabs',base&seam),('outside_seam_slabs',base&~seam)]:
   stats[label]={'rows':int(select.sum()),'finite_positive_all_draw_densities':int((select&finite_draws).sum()),'valid_class_probabilities':int((select&valid_probability).sum()),'field_eligible':int((select&fields['field_eligible']).sum()),'not_field_eligible':int((select&~fields['field_eligible']).sum()),'finite_mean':int((select&np.isfinite(fields['rho_mean'])).sum())}
  result['caps'][name]=stats
 (a.out/'COVERAGE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
