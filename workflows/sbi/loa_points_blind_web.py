"""Read-only replay on the registered blind cells and real observed-position export."""
import gzip,json,os,sys
from pathlib import Path
import numpy as np,h5py,fitsio
sys.path.insert(0,os.getcwd())
from workflows.abacus_tweb.p3br_build_random_response import galactic_cap
from workflows.sbi import e2e_coupled_coordinates as coord
root=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
out=root/('loa_points_blind_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=True)
rank=int(os.environ['SLURM_PROCID'])
if rank==0:
 g=fitsio.read(root/'lowz_loa_preparation_61911547/OBSERVED.fits',columns=['RA','DEC','Z'])
 g=g[(g['Z']>=.01)&(g['Z']<.55)];assert len(g)==7813715
 caps=galactic_cap(g['RA'],g['DEC']);bins=np.floor(g['DEC']+90).astype(int)
 for k,c in enumerate(['SGC','NGC']):
  dest=out/c;dest.mkdir();total=0
  for j in np.unique(bins[caps==k]):
   v=g[(caps==k)&(bins==j)]
   # Observed angular position and observer distance, not a synthetic point.
   xy=np.column_stack([v['RA'],coord.radius_mpc_h(v['Z'])]).astype('<f4')
   blob=gzip.compress(xy.tobytes(),compresslevel=3,mtime=0);assert gzip.decompress(blob)==xy.tobytes()
   (dest/f'{j}.gz').write_bytes(blob);total+=len(xy)
  assert total==[2230002,5583713][k];(dest/'count.json').write_text(json.dumps(total))
else:
 ph=['ph018','ph019'][rank-1];d=root/'lowz_cfm_blind_confirmation_v1'
 rows=json.loads((d/'GEOMETRY_FROZEN.json').read_text())['phases'][ph]['pairs']
 truth=np.load(d/'blind_score_61911489'/f'{ph}_truth.npy');results=[]
 for ii,row in enumerate(rows):
  c=row['cap'];xs=[]
  for i in range(32):
   q=d/ph/c/'predictions'/f'draw_{i:03d}'
   with h5py.File(q/'density/FIELD.h5') as f,h5py.File(q/'tides/COMPLETED_FIELD.h5') as t:
    loc=(np.array(row['grid']['origin_mpc_h'])+(np.array(row['center'])-16)*3.383-f.attrs['origin_mpc_h'])/6.766
    a=np.rint(loc).astype(int);assert np.allclose(loc,a)
    s=tuple(slice(v,v+w) for v,w in zip(a,(32,16,16)))
    xs.append(np.concatenate([f['rho'][(0,*s)][...,None],t['eigenvalues'][(0,*s)]],-1).reshape(-1,4))
  with h5py.File(d/'matched_reference_v1'/f'{ph}_{c}.h5') as ref:
   for shell,g in ref[row['pair_id']].items():
    ix=g['flat_cell'][:];x=np.stack(xs)[:,ix];y=truth[ii].reshape(-1,4)[ix];cl=(y[:,1:]>.2).sum(1);pred=(x[:,:,1:]>.2).sum(2)
    for cls in range(4):
     m=cl==cls
     if not m.any():continue
     err=x[:,m,0]-y[m,0];rec=dict(cap=c,pair=row['pair_id'],shell=int(shell),cls=cls,n=int(m.sum()),votes=np.bincount(pred[:,m].ravel(),minlength=4).tolist(),density_bias=float(err.mean()))
     if cls>=2:
      j=4-cls;rec['true_margin']=float((y[m,j]-.2).mean());rec['lambda_bias']=float((x[:,m,j]-y[m,j]).mean())
      down=pred[:,m]==cls-1;rec['down_density_error_sum']=float(err[down].sum())
     results.append(rec)
 assert sum(r['n'] for r in results)=={'ph018':109144,'ph019':121818}[ph]
 (out/f'{ph}.json').write_text(json.dumps(results,allow_nan=False))
