#!/usr/bin/env python3
"""Run the real annotator on source-bound stratified fixtures from qualified phases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import fitsio
import numpy as np
from p12a_coordinate_sample_audit import mapped_member
from tweb_grid_sampling import LABEL_COLUMNS
from galaxy_position_validation_status import load_independent_index, independent_status, LABEL_POLICY_VERSION, load_ambiguity_rows, ambiguity_status


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--certificate',type=Path,required=True);ap.add_argument('--samples',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 if not(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):ap.error('Compute step required')
 q=json.loads(args.certificate.read_text());args.output.mkdir(parents=True,exist_ok=False)
 report=dict(certificate_sha256=digest(args.certificate),phases={},completed=False)
 for ph,info in q['phases'].items():
  s=np.load(args.samples/f'{ph}_sample.npz');selection=[]
  for low in [False,True]:
   for res in [0,1]:
    for cen in [0,1]:
     m=((s['BOX_INDEX']<0)==low)&(s['RES']==res)&(s['CEN']==cen)
     selection.extend(np.flatnonzero(m)[:24].tolist())
  selection.extend([int(np.argmin(s['Z_COSMO'])),int(np.argmax(s['Z_COSMO']))])
  if 'label_equivalent_ambiguity' in s:selection.extend(np.flatnonzero(s['label_equivalent_ambiguity']).tolist())
  selected=np.unique(selection);ids=s['original_row'][selected]
  with fitsio.FITS(info['sources'][0]['path']) as f:data=f[1].read(rows=ids)
  fixture=np.empty(len(data),dtype=[(k,data.dtype[k]) for k in data.dtype.names]+[('SOURCE_ROW','i8')])
  for k in data.dtype.names:fixture[k]=data[k]
  fixture['SOURCE_ROW']=ids
  input_path=args.output/f'{ph}_fixture.fits';fitsio.write(input_path,fixture)
  output=args.output/f'{ph}_annotated.fits'
  cmd=[sys.executable,str(Path(__file__).with_name('annotate_cutsky_with_tweb_eigs.py')),
       '--phase',ph,'--qualification',str(args.certificate),'--cutsky',str(input_path),
       '--tweb-dir',info['field_path'],'--output-dir',str(args.output),'--output-name',output.name,'--chunk-size','37']
  log=args.output/f'{ph}_annotation.log'
  with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  result=fitsio.read(output);manifest=json.loads(Path(str(output)+'.manifest.json').read_text())
  assert manifest['output']['sha256']==digest(output)
  for key in ['SOURCE_ROW','RA','DEC','Z_COSMO','RES','CEN','HALO_MASS']:
   assert np.array_equal(result[key],fixture[key]),key
  satellites=(result['RES']==0)&(result['CEN']==0);valid=np.ones(len(result),dtype=bool)
  assert np.all(result['TWEB_STATUS']==1) and np.all(result['HAS_LABEL']==1)
  assert np.all(result['CWEB']<=3)
  assert manifest['label_policy_version']==LABEL_POLICY_VERSION
  expected_codes=independent_status(ids,load_independent_index(info['independent_position_index']))
  assert np.array_equal(result['INDEP_POS'],expected_codes)
  expected_ambiguity=ambiguity_status(ids,load_ambiguity_rows(info['independent_position_index']))
  assert np.array_equal(result['ID_AMBIG'],expected_ambiguity)
  assert np.all(result['INDEP_POS'][result['ID_AMBIG']==1]==0)
  assert np.all(result['INDEP_POS'][satellites]==0)
  for key in LABEL_COLUMNS:assert np.isfinite(result[key]).all()
  # Independent direct point reads from the original slab members. Does not
  # call the production sampler or reuse the production output's voxel columns.
  pos=s['reconstructed_native'][selected].astype('f4')
  ijk=np.clip(np.floor(np.mod(pos,2000.)/(2000./2048)).astype('i8'),0,2047)
  eig=np.full((len(result),3),np.nan)
  for rec in info['field_sources']:
   p=Path(rec['path'])
   with np.load(p) as a:lo,hi=int(a['x_start']),int(a['x_end'])
   m=(ijk[:,0]>=lo)&(ijk[:,0]<hi)&valid
   if m.any():
    arr=mapped_member(p,'eig_vals');x,y,z=ijk[m].T;eig[m]=arr[:,x-lo,y,z].T
  measured=np.column_stack([result[k] for k in LABEL_COLUMNS[:3]])
  assert np.array_equal(measured[valid],eig[valid])
  # Real CLI should reject a second write without touching the output.
  before=digest(output)
  repeat=subprocess.run(cmd,capture_output=True,text=True)
  assert repeat.returncode!=0 and 'Refusing to overwrite' in repeat.stderr and digest(output)==before
  report['phases'][ph]=dict(rows=len(result),labelled=int(valid.sum()),unresolved_satellites_labelled=int(satellites.sum()),
    identity_ambiguous=int(np.sum(result['ID_AMBIG'])),
    independently_checked=int(np.sum(result['INDEP_POS']>0)),independent_status_exact=True,
    metadata_exact=True,eigenvalues_exact_at_reconstructed_voxels=True,overwrite_rejected=True,
    output=str(output),manifest=str(output)+'.manifest.json',pass_gate=True)
  print(ph,report['phases'][ph],flush=True)
  (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 report['completed']=True;report['pass_gate']=all(x['pass_gate'] for x in report['phases'].values())
 (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
