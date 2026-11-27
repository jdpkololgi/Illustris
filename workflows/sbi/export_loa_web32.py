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
pix=np.clip((g['DEC']+90).astype(int),0,179)*208*720+(np.searchsorted(edges,co.radius_mpc_h(g['Z']),side='right')-1)*720+((g['RA']%360)*2).astype(int)
counts=np.bincount(pix,minlength=180*208*720).reshape(180,208,720);assert counts.sum()==len(g) and counts.max()<65535
with h5py.File(p) as f:
 names=('rho_draw0','rho_quantiles','web_probability','shell')
 rho,q,prob,shell=[f[n][:] for n in names];o=f.attrs['origin_mpc_h'];dx=f.attrs['cell_mpc_h']
 ras=np.deg2rad((np.arange(720)+.5)/2); kept=[]; checked=0;selectors={}
 for j in range(180):
  dec=np.deg2rad(j-89.5); direction=np.stack((np.cos(dec)*np.cos(ras),np.cos(dec)*np.sin(ras),np.full(720,np.sin(dec))),-1)
  ix=np.floor((rad[:,None,None]*direction-o)/dx).astype(int);valid=((ix>=0)&(ix<shell.shape)).all(-1);idx=tuple(np.moveaxis(np.clip(ix,0,np.array(shell.shape)-1),-1,0))
  valid &= shell[idx]>=0
  if not valid.any() and not counts[j].any():continue
  a=np.zeros((208,720,9),dtype='<u2');a[...,8]=counts[j]
  if valid.any():
   k=tuple(x[valid] for x in idx);raw=np.column_stack([rho[k],q[k]]).astype(np.float64);v=prob[k]*32
   assert (np.diff(raw[:,1:],axis=1)>=0).all() and (v==np.rint(v)).all() and (v.sum(1)==32).all()
   code=np.rint(np.log(raw)*2048+32768);assert ((code>0)&(code<65536)).all()
   a[valid,:4]=code;a[valid,4:8]=v
  data=gzip.compress(a.tobytes(),compresslevel=4,mtime=0);(out/f'{j}.gz').write_bytes(data)
  check=np.frombuffer(gzip.decompress(data),'<u2').reshape(a.shape);assert np.array_equal(check,a)
  if valid.any():
   assert np.max(abs(np.log(raw)-(check[valid,:4].astype(float)-32768)/2048))<=.50001/2048
   y,x=np.argwhere(valid)[0];k=tuple(ix[y,x]);assert np.array_equal(check[y,x,4:8],32*f['web_probability'][k])
   checked+=1
   selectors[j]=(np.flatnonzero(valid).astype('i4'),np.ravel_multi_index(tuple(ix[valid].T),shell.shape).astype('i4'))
  kept.append(j)
 for draw in range(1,32):
  base=d.parent/'predictions'/f'draw_{draw:03d}'/'density'; marker=base/'COMPLETE.json'
  assert hashlib.sha256(marker.read_bytes()).hexdigest()==receipt['receipts'][draw][0]
  rec=json.loads(marker.read_text())['field'];path=Path(rec['path'])
  assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==rec['sha256']
  with h5py.File(path) as src:
   assert np.array_equal(src.attrs['origin_mpc_h'],o) and src['rho'].shape==(1,*shell.shape)
   arr=src['rho'][0].ravel()
  folder=out/f'd{draw}';folder.mkdir()
  for j,(pos,ids) in selectors.items():
   raw=np.log(arr[ids].astype(float));codes=np.rint(raw*2048+32768);assert ((codes>0)&(codes<65536)).all()
   a=np.zeros(208*720,'<u2');a[pos]=codes
   blob=gzip.compress(a.tobytes(),compresslevel=1,mtime=0)
   assert np.max(abs((np.frombuffer(gzip.decompress(blob),'<u2')[pos].astype(float)-32768)/2048-raw))<=.50001/2048
   (folder/f'{j}.gz').write_bytes(blob)
 (out/'index.json').write_text(json.dumps(dict(cap=cap,bins=kept,drawBins=sorted(selectors),nra=720,edges=edges.tolist(),z=co.redshift(rad).tolist(),sampled_slabs_checked=checked,galaxies=len(g),source=receipt['output']['sha256'])))
