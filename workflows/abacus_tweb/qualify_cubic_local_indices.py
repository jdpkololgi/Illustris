#!/usr/bin/env python3
"""Repair the audit's cubic reference lookup without rerunning native-particle reads.

Retains the initial failed audit and links its hash; independent native matches
are reused verbatim. Changes the tested reference indexing, not the coordinate
mapping, cosmology, tolerances, or native-particle result. Development phases only.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import fitsio
import numpy as np
from validate_galaxy_position_mapping import Field, read_cubic_correspondence, stats, TOL, ALLOWED
from galaxy_position_mapping import periodic_delta


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if not(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):ap.error('Compute step required')
    previous=args.previous/'report.json';old=json.loads(previous.read_text())
    if old.get('completed') is not True:raise ValueError('Original native audit is incomplete')
    if any(ph not in ALLOWED for ph in old['phases']):raise ValueError('Development phases only')
    args.output.mkdir(parents=True,exist_ok=False)
    report=copy.deepcopy(old);report.update(completed=False,pass_gate=False,
        corrected_reference_indexing=True,previous_audit=dict(path=str(previous),sha256=digest(previous)),
        correction_script_sha256=digest(__file__),cubic_reference_helper_sha256=digest(Path(__file__).with_name('validate_galaxy_position_mapping.py')),
        correction_job_id=os.environ['SLURM_JOB_ID'])
    def save():(args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for ph,info in report['phases'].items():
        print('CUBIC',ph,flush=True)
        for rec in info['sources']:
            st=Path(rec['path']).stat()
            if (st.st_size,st.st_mtime_ns)!=(rec['bytes'],rec['mtime_ns']):raise ValueError('Source changed')
        sample=args.previous/f'{ph}_sample.npz';s=np.load(sample)
        high=s['BOX_INDEX']>=0
        with fitsio.FITS(info['sources'][1]['path']) as f:
            c,starts=read_cubic_correspondence(f[1],{k:s[k][high] for k in ['FILE_NUM','BOX_INDEX']})
        ref=np.column_stack([c[k] for k in ['x','y','z']])-1000.
        pos=s['reconstructed_native']
        residual=np.linalg.norm(periodic_delta(pos[high],ref),axis=1)
        identity=(c['FILE_NUM']==s['FILE_NUM'][high])&(c['HALO_MASS']==s['HALO_MASS'][high])&(c['cen']==s['CEN'][high])
        info['initial_incorrect_cubic_reference']=info.pop('cubic')
        info['cubic']={};info['cubic_file_starts']=starts
        info['cubic_index_semantics']='FILE_NUM-local BOX_INDEX into concatenated cubic galaxy blocks'
        info['sample_source']=dict(path=str(sample),sha256=digest(sample))
        field=Field(ph)
        info['field_sources']=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for _,_,p in field.spans]
        for cen in [0,1]:
            m=s['CEN'][high]==cen
            info['cubic'][f'cen{cen}']=dict(rows=int(m.sum()),identity_matches=int(identity[m].sum()),
                position_error_mpc_h=stats(residual[m]),within_tolerance=int(np.sum(residual[m]<TOL)),
                field=field.compare(pos[high][m],ref[m]))
        groups=list(info['cubic'].values())+info['native_checks']
        info['field_gate']=all(v['field']['same_voxel_fraction']>=.995 and v['field']['rms_eigenvalue_difference']<.01 for v in groups)
        info['pass_gate']=bool(all(v['identity_matches']==v['rows'] and v['within_tolerance']==v['rows'] for v in info['cubic'].values()) and all(v['pass_gate'] for v in info['native_checks']) and info['unresolved_satellites_in_sample']==0 and info['field_gate'])
        arrays={k:s[k] for k in s.files};arrays['independent_native'][high]=ref
        np.savez(args.output/f'{ph}_sample.npz',**arrays)
        save();print(ph,'pass',info['pass_gate'],'cubic max',residual.max(),flush=True)
    report['completed']=True;report['pass_gate']=all(v['pass_gate'] for v in report['phases'].values());save()
    print('COMPLETE',report['pass_gate'],flush=True)
if __name__=='__main__':main()
