#!/usr/bin/env python3
"""Build a non-destructive product quarantine, dependency and retirement register.

Only named owned Scratch roots are scanned, without following directory symlinks.
No payloads or checkpoints are loaded. Legacy label products are quarantined in
this register; physical files remain immutable. Dependencies are review evidence,
not proof that a consumer read the faulty columns. Version names are not proof
of redundancy. This tool never deletes or moves products or changes running jobs.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket

BASE=Path('/pscratch/sd/d/dkololgi/abacus')
FAMILIES=['e2e_field_v1','e2e_field_v2','e2e_field_v3','sbi_caches','SBI_cache files','SBI_cache_micro',
          'phaseB_pooled_ntilde','phaseB_tiled_ntilde','jraph_runs','sbi_runs','regression_runs']
SEALED={'ph001','ph018','ph019'}
SKIP={'source_snapshots','.git','__pycache__','logs','log','slurm_logs'}
INPUT_WORDS=('input','source','parent','target','checkpoint','cache','normaliz','binding','raw','old_','condition','truth','data','train','reference')

def path_id(path):return hashlib.sha256(str(path).encode()).hexdigest()[:16]
def fingerprint(p):
    s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def under(p,root):return p==root or root in p.parents

def refs(value,keys=()):
    if isinstance(value,dict):
        for k,v in value.items():yield from refs(v,keys+(str(k),))
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from refs(v,keys+(str(i),))
    elif isinstance(value,str) and value.startswith(('/pscratch/','/global/')) and '\n' not in value:
        key='.'.join(keys);lower=key.lower()
        direction='input' if any(w in lower for w in INPUT_WORDS) else 'untyped'
        if any(k.lower() in ('output','outputs','destination','dest') for k in keys):direction='output'
        if value.endswith(('.py','.sh','.slurm')) or 'source_code' in lower:direction='code'
        yield dict(key=key,path=value,direction=direction)

def scan_unit(path,max_depth=7):
    receipts=[];files=[];skipped=[];pending=[(path,0)];visited=0
    if path.is_file():return dict(files=[fingerprint(path)],receipts=[],skipped=[])
    while pending:
        current,depth=pending.pop()
        for p in sorted(current.iterdir()):
            visited+=1
            if visited>100000:raise ValueError(f'Unit exceeds bounded 100000-entry scan: {path}')
            if p.is_symlink():skipped.append(dict(path=str(p),reason='symlink'));continue
            if p.is_dir():
                if p.name in SKIP or set(re.findall(r'ph\d{3}',p.name))&SEALED:skipped.append(dict(path=str(p),reason='excluded directory'));continue
                if depth>=max_depth:skipped.append(dict(path=str(p),reason='depth limit'));continue
                pending.append((p,depth+1));continue
            if not p.is_file():continue
            rec=fingerprint(p);files.append(rec)
            if p.suffix!='.json' or rec['bytes']>2*1024**2:continue
            try:raw=p.read_bytes();data=json.loads(raw)
            except (ValueError,OSError):continue
            if not isinstance(data,dict):continue
            # Absolute path references only; no model weights or scientific arrays.
            entries=list(refs(data))
            receipts.append(dict(**rec,sha256=hashlib.sha256(raw).hexdigest(),
              schema=data.get('schema_version',data.get('schema')),
              completed=data.get('completed',data.get('passed',data.get('pass'))),refs=entries))
    return dict(files=files,receipts=receipts,skipped=skipped)

def audit(inventory,output):
    if not(os.environ.get('SLURM_JOB_ID') and socket.gethostname().startswith('nid')):raise RuntimeError('Use approved compute allocation')
    initial=json.loads(inventory.read_text());units={}
    for r in initial['records']:
        if r['category'] in ['lineage_receipt','audit_lineage_before_retraining']:continue
        p=Path(r['path'])
        if not p.exists():continue
        category=r['category']
        quarantine=category in ['rebuild_galaxy_labels','rebuild_or_retire_legacy_annotations','rebuild_target_arrays_rebind_geometry']
        retain=category in ['retain_native_density_and_tweb','not_affected_by_host_lookup']
        units[str(p)]=dict(id=path_id(p),path=str(p),family='P10_or_legacy_annotation',
          status='quarantine_legacy_target' if quarantine else 'retain_native_or_direct_field' if retain else 'review_reuse_or_metadata',
          reason=r['reason'],tags=[category],replacement=None,archive_ready=False,
          receipts=[],files=[],skipped=[],scan_kind='prior_bounded_inventory')
    for family in FAMILIES:
        root=BASE/family
        if not root.exists():continue
        for p in sorted(root.iterdir()):
            if p.is_symlink() or not(p.is_dir() or p.suffix in ['.pkl','.npz','.h5','.hdf5','.pt','.ckpt']):continue
            unit=dict(id=path_id(p),path=str(p),family=family,status='lineage_review_required',
              reason='No assumption that every model used legacy galaxy labels.',tags=[],replacement=None,archive_ready=False,
              scan_kind='bounded_metadata')
            unit.update(scan_unit(p));units[str(p)]=unit
            print('SCANNED',family,p.name,len(unit['files']),len(unit['receipts']),flush=True)
    paths=sorted(units,key=len,reverse=True)
    def owner(value):
        p=Path(value)
        for root in paths:
            if under(p,Path(root)):return root
        return None
    edges=[];all_references=set();incoming=defaultdict(set)
    for path,u in units.items():
        for receipt in u['receipts']:
            for ref in receipt['refs']:
                all_references.add(ref['path']);parent=owner(ref['path'])
                if parent and parent!=path and ref['direction'] in ['input','untyped']:
                    edges.append(dict(consumer=u['id'],dependency=units[parent]['id'],receipt=receipt['path'],key=ref['key'],direction=ref['direction'],referenced_path=ref['path']))
                    incoming[parent].add(path)
                    if units[parent]['status']=='quarantine_legacy_target':
                        u['tags'].append('references_legacy_target_review_column_usage')
    byid={u['id']:u for u in units.values()}
    suspects={u['id'] for u in units.values() if u['status']=='quarantine_legacy_target'}
    changed=True
    while changed:
        changed=False
        for e in edges:
            if e['dependency'] in suspects and e['consumer'] not in suspects:
                suspects.add(e['consumer']);changed=True
    candidates=[]
    for path,u in units.items():
        u['known_consumers']=sorted(incoming[path])
        if u['id'] in suspects and u['status']!='quarantine_legacy_target':u['tags'].append('transitive_legacy_dependency_requires_usage_review')
        u['tags']=sorted(set(u['tags']))
        u['scanned_file_bytes']=sum(f['bytes'] for f in u['files'])
        # Supersession is a review cue only: scientific negative results remain valuable.
        m=re.match(r'(.+)_v(\d+)$',Path(path).name)
        if m:
            newer=[p for p in units if Path(p).parent==Path(path).parent and re.fullmatch(re.escape(m[1])+r'_v\d+',Path(p).name) and int(Path(p).name.rsplit('_v',1)[1])>int(m[2])]
            if newer:
                u['tags'].append('newer_named_version_exists_not_proof_of_redundancy')
                candidates.append(dict(path=path,reason='newer_named_version',possible_successors=sorted(newer),status='review_only',archive_ready=False))
        for f in u['files']:
            p=Path(f['path'])
            if re.search(r'_generation_\d+\.(h5|npy)$',p.name) and str(p) not in all_references:
                candidates.append(dict(path=str(p),reason='generation_payload_not_referenced_in_scanned_receipts',status='review_only',bytes=f['bytes'],archive_ready=False))
        u.pop('files')
    # These source-reviewed families derive targets directly from matter. Keep
    # mixed roots intact; this is not a declaration that every historical version passed.
    evidence={
      str(BASE/'e2e_field_v3/coupled_20260918_v1'):('workflows/sbi/e2e_coupled_target_products.py','Native A+B particle density -> R7 density/tensor targets. Mixed historical root: inspect separate coordinate/target-recovery receipts before retirement.'),
      str(BASE/'e2e_field_v3/wholefield_pilot_20261013_v1'):('workflows/sbi/e2e_wholefield_pilot_data.py','targets() reads native matter DENSITY_COMPLETE and rho payload, not annotated galaxy eigenvalues.')}
    for path,(code,reason) in evidence.items():
        if path in units:
            units[path]['tags'].append('direct_matter_target_builder_reviewed_no_blanket_bug_quarantine')
            units[path]['source_review']=dict(path=code,sha256=hashlib.sha256(Path(code).read_bytes()).hexdigest(),finding=reason,
              limitation='Current builder review plus recorded dependencies; not retrospective recertification of every checkpoint or old coordinate version.')
    payload=dict(schema_version='scientific-product-lifecycle-v1',created_utc=datetime.now(timezone.utc).isoformat(),
      job_id=os.environ['SLURM_JOB_ID'],inventory_source=fingerprint(inventory),
      scope=dict(owned_root=str(BASE),families=FAMILIES,max_depth=7,metadata_only=True,sealed_phase_directories_skipped=sorted(SEALED)),
      physical_actions_performed=False,registry_is_logical_quarantine=True,
      lifecycle_rules=['Quarantine legacy targets from new science use; keep original evidence intact.',
       'Build corrected products at new versioned paths and validate them before registering replacements.',
       'Retire a consumer only after its input-column/target lineage is established; geometry-only references may be reusable.',
       'Archive physically only from a separately reviewed exact-path manifest after active jobs and consumer migrations are checked.',
       'A newer version, failed scientific gate or missing reference in this bounded scan does not prove redundancy.',
       'Keep native fields, controls, negative-result receipts and source snapshots. Scratch archiving is not backup.'],
      units=list(units.values()),dependencies=edges,retirement_candidates=candidates,
      counts=dict(Counter(u['status'] for u in units.values())))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print('REGISTER',len(units),'EDGES',len(edges),'REVIEW_CANDIDATES',len(candidates),payload['counts'],flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();audit(a.inventory,a.output)
if __name__=='__main__':main()
