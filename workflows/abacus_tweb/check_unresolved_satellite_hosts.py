#!/usr/bin/env python3
"""Supplemental unresolved-satellite host closure; NOT exact satellite-position truth.

Recover each sampled satellite's central by RES/FILE_NUM/HALO_INDEX plus mass,
then independently locate that central's field particle. Test the satellite's
own reconstructed offset against the generator's R200mean support. This cannot
recover the original random satellite displacement or certify it independently.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import fitsio
import numpy as np
from abacusnbody.data.read_abacus import read_asdf
from galaxy_position_mapping import reconstruct_native,periodic_delta,cosmology
from validate_galaxy_position_mapping import SIMROOT, COLS, TOL


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--qualification',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 if not(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):ap.error('Compute step required')
 d=json.loads(args.qualification.read_text());args.output.mkdir(parents=True,exist_ok=False)
 report=dict(kind='unresolved-satellite-native-host-closure',independent_satellite_positions=False,phases={},completed=False)
 def save():(args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 for ph,info in d['phases'].items():
  s=np.load(args.qualification.parent/f'{ph}_sample.npz')
  mask=(s['RES']==0)&(s['CEN']==0);sat={k:s[k][mask] for k in COLS};ids=s['original_row'][mask]
  keys=(sat['FILE_NUM'].astype('i8')<<32)+sat['HALO_INDEX'].astype('i8')
  print('SCAN',ph,'satellites',len(keys),flush=True)
  matches=[];match_rows=[]
  with fitsio.FITS(info['sources'][0]['path']) as f:
   h=f[1]
   for lo in range(0,h.get_nrows(),1_000_000):
    hi=min(lo+1_000_000,h.get_nrows())
    a=h.read(rows=np.arange(lo,hi),columns=['FILE_NUM','HALO_INDEX','RES','CEN'])
    key=(a['FILE_NUM'].astype('i8')<<32)+a['HALO_INDEX'].astype('i8')
    m=(a['RES']==0)&(a['CEN']==1)&np.isin(key,keys)
    row=np.flatnonzero(m)+lo
    if len(row):matches.append(h.read(rows=row,columns=COLS));match_rows.extend(row.tolist())
  if not matches:raise ValueError('No matching unresolved central rows')
  cen=np.concatenate(matches);cpos=reconstruct_native(cen['RA'],cen['DEC'],cen['Z_COSMO'])
  entries=[]
  for fn in np.unique(cen['FILE_NUM']):
   p=SIMROOT/f'AbacusSummit_base_c000_{ph}'/'halos/z0.200/field_rv_A'/f'field_rv_A_{int(fn):03d}.asdf'
   print('PARTICLES',ph,int(fn),flush=True)
   tab=read_asdf(p,load=['pos'],verbose=False);points=np.asarray(tab['pos'])
   for ci in np.flatnonzero(cen['FILE_NUM']==fn):
    best=np.inf;best_row=-1;within=0
    for lo in range(0,len(points),1_000_000):
     delta=periodic_delta(points[lo:lo+1_000_000],cpos[ci]);distance=np.einsum('ij,ij->i',delta,delta)
     j=int(np.argmin(distance));within+=int(np.sum(distance<TOL*TOL))
     if distance[j]<best:best=float(distance[j]);best_row=lo+j
    sm=(sat['FILE_NUM']==fn)&(sat['HALO_INDEX']==cen['HALO_INDEX'][ci])
    spos=reconstruct_native(sat['RA'][sm],sat['DEC'][sm],sat['Z_COSMO'][sm])
    offset=np.linalg.norm(periodic_delta(spos,cpos[ci]),axis=1)
    # Public generator uses a comoving R200mean at the snapshot epoch. The
    # mean density at z=0 cancels the explicit (1+z)^3/comoving factors.
    rho=2.77536627e11*(cosmology().Om0+cosmology().Onu0)
    r200=(3*(sat['HALO_MASS'][sm]*1e10)/(800*np.pi*rho))**(1/3)
    entry=dict(file_num=int(fn),halo_index=int(cen['HALO_INDEX'][ci]),central_source_row=int(match_rows[ci]),
      satellite_source_rows=ids[sm].tolist(),central_particle_distance_mpc_h=float(np.sqrt(best)),
      particles_within_tolerance=within,particle_row=best_row,mass_equal=bool(np.all(sat['HALO_MASS'][sm]==cen['HALO_MASS'][ci])),
      satellite_offset_mpc_h=offset.tolist(),r200mean_mpc_h=r200.tolist(),
      offset_within_r200_plus_tolerance=bool(np.all(offset<=r200+TOL)),
      central_particle_match=bool(best<TOL*TOL and within==1))
    entries.append(entry)
   del tab,points;gc.collect()
  matched=sum(len(v['satellite_source_rows']) for v in entries)
  report['phases'][ph]=dict(sampled_satellites=len(ids),matched_to_centrals=matched,entries=entries,
    host_closure_pass=matched==len(ids) and all(v['mass_equal'] and v['central_particle_match'] and v['offset_within_r200_plus_tolerance'] for v in entries))
  save();print('DONE',ph,report['phases'][ph]['host_closure_pass'],flush=True)
 report['completed']=True;report['host_closure_pass']=all(v['host_closure_pass'] for v in report['phases'].values());save()
if __name__=='__main__':main()
