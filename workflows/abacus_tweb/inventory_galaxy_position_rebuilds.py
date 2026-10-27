#!/usr/bin/env python3
"""Bounded, metadata-only inventory of products affected by galaxy-position targets.

No model/checkpoint payload is opened, no source files are changed, and no blind
truth is inspected. Presence and lineage hints are not proof of model dependence.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import fitsio

BASE=Path('/pscratch/sd/d/dkololgi/abacus')
PHASES=('ph000','ph002','ph003','ph004','ph005','ph006')
STAGES={
 'p1_canonical':('rebuild_target_arrays_rebind_geometry','Canonical index contains eigenvalues/validity; keep row/geometry identity if unchanged.'),
 'p2_graph':('reuse_if_geometry_identical','Topology/features do not change with labels; refresh provenance after exact row/position verification.'),
 'p2_union':('reuse_if_geometry_identical','Radius union depends on geometry, not galaxy labels.'),
 'p3_fields':('reuse_if_geometry_selection_identical','Recorded input channels are target-free; changed parent hashes need verified rebinding.'),
 'p3b_random_response_v1':('reuse_if_selection_identical','Random-derived survey support is independent of host labels.'),
 'p3c_assignment_response_v1':('reuse_if_selection_identical','Assignment response is independent of host labels.'),
 'p4_patches':('review_target_bindings_preserve_folds','Spatial cores/folds can be retained with unchanged population; replace embedded targets or target hashes.'),
 'p4_rebuild':('review_target_bindings_preserve_folds','Spatial products may be reused only with unchanged geometry/membership and verified lineage.'),
 'p12f_field_targets_v1':('not_affected_by_host_lookup','Continuous voxel targets are sampled directly from T-Web grids; this does not recertify their coordinate/support choices.'),
 'targets':('retain_native_density_and_tweb','Native matter grids precede galaxy annotation; no regeneration for this defect.'),
}

def stat(p):
 s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def small_json(p):
 if p.stat().st_size>2*1024**2:return None
 try:return json.loads(p.read_text())
 except (ValueError,UnicodeError,OSError):return None

def refs(obj):
 result=[]
 if isinstance(obj,dict):
  for k,v in obj.items():
   if isinstance(v,str) and (v.startswith('/pscratch/') or v.startswith('/global/')) and any(t in k.lower() for t in ['path','catalog','cache','checkpoint','source','input','manifest','parent','target']):result.append(dict(key=k,value=v))
   elif isinstance(v,(dict,list)):result.extend(refs(v))
 elif isinstance(obj,list):
  for v in obj[:1000]:
   if isinstance(v,(dict,list)):result.extend(refs(v))
 return result[:100]

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 records=[]
 def add(p,category,reason,**extra):records.append(dict(**stat(p),category=category,reason=reason,**extra))
 for phase in PHASES:
  root=BASE/'p10_multiphase'/phase
  if not root.exists():continue
  cat=root/'catalogues'
  if cat.exists():
   for sub in sorted(cat.iterdir()):
    if not sub.is_dir():continue
    for p in sorted(sub.glob('*.fits')):
     with fitsio.FITS(p) as f:
      cols=f[1].get_colnames();n=f[1].get_nrows();hdr=f[1].read_header()
     new=(hdr.get('TWMAP')=='bgs-v01-c000-galaxy-realspace-v1' and hdr.get('TWPOLICY')=='galaxy-realspace-all-populations-v2')
     category='qualified_new_target' if new else ('rebuild_galaxy_labels' if 'LAMBDA1' in cols else 'rebuild_source_metadata')
     add(p,category,'Galaxy-position migration requires SOURCE_ROW, Z_COSMO, RES, CEN, and the new target contract.',phase=phase,rows=n,
         required_metadata_present={k:k in cols for k in ['SOURCE_ROW','Z_COSMO','RES','CEN','HALO_MASS','TWEB_STATUS','INDEP_POS']},mapping=hdr.get('TWMAP'),label_policy=hdr.get('TWPOLICY'))
  for stage,(category,reason) in STAGES.items():
   p=root/stage
   if not p.exists():continue
   receipts=[]
   for f in sorted(p.iterdir()):
    if f.is_file() and (f.suffix=='.json' or 'COMPLETE' in f.name):
     d=small_json(f)
     if isinstance(d,dict):receipts.append(dict(**stat(f),schema=d.get('schema_version'),lineage_hints=refs(d)))
   add(p,category,reason,phase=phase,receipts=receipts)
 # Historical ph000 annotation families: immediate FITS children only.
 for root in sorted(BASE.glob('mocks_with_eigs*')):
  if not root.is_dir():continue
  for p in sorted(root.glob('*.fits')):
   with fitsio.FITS(p) as f:
    cols=f[1].get_colnames();n=f[1].get_nrows();hdr=f[1].read_header()
   add(p,'rebuild_or_retire_legacy_annotations','Legacy host-centre target; low-z status/selection must be checked per downstream consumer.',phase='ph000',rows=n,
       required_metadata_present={k:k in cols for k in ['SOURCE_ROW','Z_COSMO','RES','CEN','HALO_MASS','TWEB_STATUS','INDEP_POS']},mapping=hdr.get('TWMAP'),label_policy=hdr.get('TWPOLICY'))
 # Bounded lineage receipts, at most two directory levels below named product roots.
 for name in ['sbi_caches','SBI_cache files','SBI_cache_micro','phaseB_pooled_ntilde','phaseB_tiled_ntilde','jraph_runs','sbi_runs','regression_runs','e2e_field_v1','e2e_field_v2','e2e_field_v3']:
  root=BASE/name
  if not root.exists():continue
  add(root,'audit_lineage_before_retraining','Label-bearing caches/scalers/models must be rebuilt for the new estimand; target-free or direct-field products are not automatically invalidated.')
  dirs=[root]
  for _ in range(2):
   next_dirs=[]
   for d in dirs:
    for p in sorted(d.iterdir()):
     if p.is_dir():next_dirs.append(p)
     elif p.suffix=='.json' and any(k in p.name.lower() for k in ['manifest','metadata','config','complete']):
      data=small_json(p)
      if isinstance(data,dict):add(p,'lineage_receipt','Reference for targeted cache/checkpoint rebuild; not evidence that all models consumed affected labels.',lineage_hints=refs(data))
   dirs=next_dirs
  if len(records)>5000:raise RuntimeError('Bounded inventory exceeded 5000 records')
 payload=dict(schema_version='galaxy-position-rebuild-inventory-v1',root=str(BASE),
  scope='Current P10 visible phases, immediate legacy ph000 annotation FITS, and bounded cache/model receipts; not an exhaustive scratch crawl.',
  ph001_truth_opened=False,records=records,counts=dict(Counter(r['category'] for r in records)))
 args.output.parent.mkdir(parents=True,exist_ok=True)
 if args.output.exists():raise FileExistsError(args.output)
 args.output.write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps(payload['counts'],indent=2))
if __name__=='__main__':main()
