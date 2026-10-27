#!/usr/bin/env python3
"""Inspect, logically archive and replace products in a versioned registry.

Archiving here preserves files at their original paths and marks them historical,
so old evidence stays reproducible. It does not reclaim disk space. Physical moves
need a separately reviewed manifest and active-job/dependency checks. Existing
registry snapshots and products are never overwritten.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket

QUARANTINE={'quarantine_legacy_target','archived_in_place'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def lookup(registry,path):
    path=Path(path)
    matches=[u for u in registry['units'] if path==Path(u['path']) or Path(u['path']) in path.parents]
    if not matches:raise ValueError('Path is not covered by this registry')
    return max(matches,key=lambda u:len(u['path']))

def reject_historical_input(path,registry_path=None):
    """Narrow default guard: unknown inputs still need their normal provenance checks."""
    registry_path=Path(registry_path) if registry_path is not None else Path(__file__).resolve().parents[2]/'configs/product_lifecycle_20261019.json'
    if not registry_path.exists():return
    registry=json.loads(registry_path.read_text())
    try:unit=lookup(registry,Path(path).expanduser().resolve())
    except ValueError:return
    if unit['status'] in QUARANTINE:
        raise ValueError(f"Archived historical target cannot seed a new build: {unit['path']}. Use a validated versioned replacement; preserve the original for historical reproduction.")

def require_current(registry,path):
    u=lookup(registry,path)
    if u['status'] in QUARANTINE:raise ValueError(f"Historical/legacy target is blocked for new science use: {u['path']}")
    if u['status']!='corrected_replacement':raise ValueError('Product needs a use-specific lineage/reuse review; registry presence is not certification')
    return u

def archive_in_place(registry,paths):
    result=copy.deepcopy(registry)
    for path in paths:
        u=lookup(result,path)
        if str(Path(path))!=u['path']:raise ValueError('Archive an exact registered product, not an unreviewed subpath')
        if u['status']!='quarantine_legacy_target':raise ValueError('Automatic in-place archive is limited to confirmed legacy target products')
        u['prior_status']=u['status'];u['status']='archived_in_place';u['physical_path_unchanged']=True
        u['archive_reason']='Legacy host-centre target: retain original evidence, block new galaxy-position use.'
    result['physical_actions_performed']=False
    return result

def verify_catalogue_identity(old_path,new_path):
    """A replacement preserves catalogue membership/order, not just a filename."""
    import fitsio
    import numpy as np
    with fitsio.FITS(old_path) as old, fitsio.FITS(new_path) as new:
        n=old[1].get_nrows()
        if n!=new[1].get_nrows():raise ValueError('Replacement row count differs; requires a separate selection migration')
        common=set(old[1].get_colnames())&set(new[1].get_colnames())
        if not {'RA','DEC','Z'}.issubset(common):raise ValueError('Missing shared identity/geometry columns')
        cols=[k for k in ['SOURCE_ROW','TARGETID','RA','DEC','Z','Z_COSMO'] if k in common]
        for start in range(0,n,100000):
            rows=np.arange(start,min(n,start+100000));a=old[1].read(rows=rows,columns=cols);b=new[1].read(rows=rows,columns=cols)
            for k in cols:
                if not np.array_equal(a[k],b[k]):raise ValueError(f'Replacement identity/order/geometry differs in {k}')
    return dict(rows=n,columns=cols,exact=True)

def register_replacement(registry,old_path,receipt_path,reason):
    if not(os.environ.get('SLURM_JOB_ID') and socket.gethostname().startswith('nid')):raise RuntimeError('Replacement payload hashing requires compute')
    result=copy.deepcopy(registry);old=lookup(result,old_path)
    if old['path']!=str(Path(old_path)) or old['status'] not in QUARANTINE:raise ValueError('Exact quarantined catalogue required')
    if not Path(old_path).is_file() or Path(old_path).suffix!='.fits':raise ValueError('This adapter registers annotation FITS replacements only; model/cache replacements need their own validation contract')
    rec=json.loads(Path(receipt_path).read_text())
    from galaxy_position_validation_status import LABEL_POLICY_VERSION
    if (rec.get('schema_version')!='galaxy-position-tweb-annotation-v2' or rec.get('completed') is not True
        or rec.get('label_policy_version')!=LABEL_POLICY_VERSION or rec.get('input_rows_fully_labelled') is not True
        or rec.get('source_rows_verified') is not True):raise ValueError('Replacement annotation has not passed the v2 contract')
    path=Path(rec['output']['path'])
    if path==Path(old_path) or not path.is_file():raise ValueError('A distinct existing corrected output is required')
    if digest(path)!=rec['output']['sha256']:raise ValueError('Replacement output hash mismatch')
    if any(u['path']==str(path) for u in result['units']):raise ValueError('Replacement already registered')
    identity=verify_catalogue_identity(old_path,path)
    proof=dict(path=str(Path(receipt_path).resolve()),sha256=digest(receipt_path),identity=identity)
    old['status']='archived_in_place';old['replacement']=str(path);old['replacement_receipt']=proof
    old['replacement_reason']=reason;old['physical_path_unchanged']=True
    result['units'].append(dict(id=hashlib.sha256(str(path).encode()).hexdigest()[:16],path=str(path),
      family='galaxy_position_annotation_v2',status='corrected_replacement',replaces=str(old_path),
      receipt=proof,phase=rec['phase'],label_policy_version=LABEL_POLICY_VERSION,known_consumers=[],
      archive_ready=False,tags=['replacement_hash_verified']))
    return result

def write_revision(path,registry,source,action):
    registry['parent_registry']=dict(path=str(Path(source).resolve()),sha256=digest(source))
    registry['updated_utc']=datetime.now(timezone.utc).isoformat();registry['last_action']=action
    from collections import Counter
    registry['counts']=dict(Counter(u['status'] for u in registry['units']))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(registry,f,indent=2);f.write('\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--registry',type=Path,required=True)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('check');p.add_argument('path')
    p=sub.add_parser('archive-in-place');p.add_argument('--paths',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('register-replacement');p.add_argument('--old',required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--reason',required=True);p.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();r=json.loads(a.registry.read_text())
    if a.command=='check':print(json.dumps(require_current(r,a.path),indent=2));return
    if a.command=='archive-in-place':result=archive_in_place(r,a.paths)
    else:result=register_replacement(r,a.old,a.receipt,a.reason)
    write_revision(a.output,result,a.registry,a.command)
    print(a.output)
if __name__=='__main__':main()
