#!/usr/bin/env python3
"""Exact streaming population census for exposed fiducial BGS CutSky catalogues.

Only metadata columns are read. No sealed phases, HOD variants or density payloads.
Partial per-phase receipts permit an explicit resume without recounting completed
phases; input size/mtime must still match. Never extrapolate sample frequencies.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import fitsio
import numpy as np

ROOT=Path('/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200')
EXPOSED=('ph000',)+tuple(f'ph{i:03d}' for i in range(2,18))

def source_record(p):
    s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def population_counts(a,mask):
    return {f'RES{r}_CEN{c}':int(np.sum(mask&(a['RES']==r)&(a['CEN']==c))) for r in [0,1] for c in [0,1]}

def census(phase,output,chunk_size):
    source=ROOT/f'cutsky_BGS_z0.200_AbacusSummit_base_c000_{phase}.fits'
    before=source_record(source);dest=Path(output)/f'{phase}.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        if old.get('completed') and old['source']==before:return old
        raise ValueError(f'Incomplete/stale phase receipt: {dest}')
    groups={};start=time.monotonic();rows=0
    with fitsio.FITS(source) as f:
        n=f[1].get_nrows()
        for lo in range(0,n,chunk_size):
            a=f[1].read(rows=np.arange(lo,min(lo+chunk_size,n)),columns=['RES','CEN','BOX_INDEX','R_MAG_APP'])
            if not np.isin(a['RES'],[0,1]).all() or not np.isin(a['CEN'],[0,1]).all():raise ValueError('Unknown RES/CEN encoding')
            masks={'full':np.ones(len(a),bool),'bright_r_lt_19p5':np.isfinite(a['R_MAG_APP'])&(a['R_MAG_APP']<19.5),
                   'low_branch':a['BOX_INDEX']<0,'high_branch':a['BOX_INDEX']>=0}
            for key,mask in masks.items():
                counts=population_counts(a,mask)
                target=groups.setdefault(key,dict.fromkeys(counts,0))
                for k,v in counts.items():target[k]+=v
            rows+=len(a)
    if rows!=n or sum(groups['full'].values())!=n or source_record(source)!=before:raise ValueError('Count/source consistency failure')
    for k in groups['full']:
        if groups['full'][k]!=groups['low_branch'][k]+groups['high_branch'][k]:raise ValueError('Branch sum mismatch')
    result=dict(phase=phase,source=before,rows=rows,groups=groups,
                unresolved_satellites=groups['full']['RES0_CEN0'],fraction=groups['full']['RES0_CEN0']/rows,
                elapsed_seconds=time.monotonic()-start,completed=True)
    tmp=dest.with_suffix('.partial');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(dest)
    print(phase,rows,result['unresolved_satellites'],flush=True)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--phases',nargs='+',default=list(EXPOSED));ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--chunk-size',type=int,default=1_000_000);args=ap.parse_args()
    if not(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):ap.error('Approved compute step required')
    if any(p not in EXPOSED for p in args.phases) or len(set(args.phases))!=len(args.phases):ap.error('Only unique exposed fiducial phases allowed')
    if not 1<=args.workers<=2 or not 1<=args.chunk_size<=2_000_000:ap.error('Bounded worker/chunk budget exceeded')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'COMPLETE.json').exists():raise FileExistsError('Census already complete')
    records={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(census,p,str(args.output),args.chunk_size) for p in args.phases]
        for f in as_completed(futures):
            result=f.result();records[result['phase']]=result
    total=sum(r['rows'] for r in records.values());count=sum(r['unresolved_satellites'] for r in records.values())
    result=dict(schema_version='unresolved-satellite-census-v1',phases=dict(sorted(records.items())),rows=total,
      unresolved_satellites=count,fraction=count/total,job_id=os.environ['SLURM_JOB_ID'],completed=True,
      source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      scope='Every row in the named exposed fiducial CutSky catalogues, not merely sampled rows; no selected-observed-sample claim.',
      excluded='ph001 and ph018 onward are not opened; HOD variants and ph006_wdens are excluded.')
    (args.output/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
    print('TOTAL',total,count,flush=True)
if __name__=='__main__':main()
