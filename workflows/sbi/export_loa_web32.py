"""Lossless class votes, log-encoded densities and observed counts on 1-degree slices."""
import os, sys, json, gzip, hashlib
from pathlib import Path
import numpy as np,h5py,fitsio
sys.path.insert(0,os.getcwd())
from workflows.sbi import e2e_coupled_coordinates as co
from workflows.abacus_tweb.p3br_build_random_response import galactic_cap
root=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
cap=('NGC','SGC')[int(os.environ['SLURM_PROCID'])]
out=root/('loa_web32_'+os.environ['SLURM_JOB_ID'])/cap
out.mkdir(parents=True,exist_ok=False)
d=root/'lowz_loa_response_61911856'/cap/'map_62099144'
p=d/'CAP_MAP.h5'; receipt=json.loads((d/'MAP_COMPLETE.json').read_text())
assert receipt['passed'] and len(receipt['receipts'])==32 and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==receipt['output']['sha256']
edges=np.linspace(*co.radius_mpc_h(np.array([.01,.55])),209); rad=(edges[:-1]+edges[1:])/2
g=fitsio.read(root/'lowz_loa_preparation_61911547/OBSERVED.fits',columns=['RA','DEC','Z'])
g=g[(g['Z']>=.01)&(g['Z']<.55)&(galactic_cap(g['RA'],g['DEC'])==(cap=='NGC'))]
pix=np.clip((g['DEC']+90).astype(int),0,179)*208*360+(np.searchsorted(edges,co.radius_mpc_h(g['Z']),side='right')-1)*360+(g['RA']%360).astype(int)
counts=np.bincount(pix,minlength=180*208*360).reshape(180,208,360);assert counts.sum()==len(g) and counts.max()<65535
with h5py.File(p) as f:
 names=('rho_draw0','rho_quantiles','web_probability','shell')
 rho,q,prob,shell=[f[n][:] for n in names];o=f.attrs['origin_mpc_h'];dx=f.attrs['cell_mpc_h']
 ras=np.deg2rad(np.arange(360)+.5); kept=[]; checked=0
 for j in range(180):
  dec=np.deg2rad(j-89.5); direction=np.stack((np.cos(dec)*np.cos(ras),np.cos(dec)*np.sin(ras),np.full(360,np.sin(dec))),-1)
  ix=np.floor((rad[:,None,None]*direction-o)/dx).astype(int);valid=((ix>=0)&(ix<shell.shape)).all(-1);idx=tuple(np.clip(ix,0,np.array(shell.shape)-1).T[::-1]) if False else tuple(np.moveaxis(np.clip(ix,0,np.array(shell.shape)-1),-1,0))
  valid &= shell[idx]>=0
  if not valid.any() and not counts[j].any():continue
  a=np.zeros((208,360,9),dtype='<u2');a[...,8]=counts[j]
  if valid.any():
   k=tuple(x[valid] for x in idx);raw=np.column_stack([rho[k],q[k]]);v=prob[k]*32
   assert (np.diff(raw[:,1:],axis=1)>=0).all() and (v==np.rint(v)).all() and (v.sum(1)==32).all()
   code=np.rint(np.log(raw)*2048+32768);assert ((code>0)&(code<65536)).all()
   a[valid,:4]=code;a[valid,4:8]=v
  data=gzip.compress(a.tobytes(),compresslevel=4,mtime=0);(out/f'{j}.gz').write_bytes(data)
  check=np.frombuffer(gzip.decompress(data),'<u2').reshape(a.shape);assert np.array_equal(check,a)
  if valid.any():
   assert np.max(abs(np.log(raw)-(check[valid,:4].astype(float)-32768)/2048))<=.50001/2048
   y,x=np.argwhere(valid)[0];k=tuple(ix[y,x]);assert np.array_equal(check[y,x,4:8],32*f['web_probability'][k])
   checked+=1
  kept.append(j)
 (out/'index.json').write_text(json.dumps(dict(cap=cap,bins=kept,edges=edges.tolist(),z=co.redshift(rad).tolist(),sampled_slabs_checked=checked,galaxies=len(g),source=receipt['output']['sha256'])))
