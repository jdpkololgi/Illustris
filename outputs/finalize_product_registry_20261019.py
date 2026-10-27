from pathlib import Path
import sys,json,hashlib,csv
from collections import Counter,defaultdict
ROOT=Path('/global/u2/d/dkololgi/TNG/Illustris')
sys.path.insert(0,str(ROOT/'workflows/abacus_tweb'))
from product_lifecycle import archive_in_place,digest
p=ROOT/'outputs/product_lifecycle_optimized_20261019.json';r=json.loads(p.read_text())
legacy=[u['path'] for u in r['units'] if u['status']=='quarantine_legacy_target']
r=archive_in_place(r,legacy)
r['full_metadata_audit']={'path':str(p),'sha256':digest(p)}
r['audit_source']={'path':str(ROOT/'workflows/abacus_tweb/audit_product_lifecycle.py'),'sha256':digest(ROOT/'workflows/abacus_tweb/audit_product_lifecycle.py')}
# Compact durable registry; full receipt/path evidence remains in the audit.
for u in r['units']:
 receipts=u.pop('receipts',[])
 u['receipt_count']=len(receipts)
 u['receipt_evidence_examples']=[{k:v for k,v in x.items() if k!='refs'} for x in receipts[:5]]
 u['all_receipt_evidence_digest']=hashlib.sha256(json.dumps(receipts,sort_keys=True).encode()).hexdigest()
groups={}
for edge in r['dependencies']:
 key=(edge['consumer'],edge['dependency'],edge['direction'])
 item=groups.setdefault(key,dict(consumer=key[0],dependency=key[1],direction=key[2],occurrences=0,evidence=[]))
 item['occurrences']+=1
 if len(item['evidence'])<5:item['evidence'].append(edge)
r['dependencies']=list(groups.values())
partial=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v1/data_20260908/native_truth/ph002_partial_58068916.h5')
marker=partial.parent/'ph002_COMPLETE.json';m=json.loads(marker.read_text())
if partial.exists() and m['shard']!=str(partial) and Path(m['shard']).exists():
 r['units'].append(dict(id=hashlib.sha256(str(partial).encode()).hexdigest()[:16],path=str(partial),family='e2e_field_v1',
  status='archived_in_place',prior_status='documented_incomplete_generation',physical_path_unchanged=True,
  reason='Interrupted shard explicitly excluded from manifests; separate completed ph002 shard exists. Retain failure provenance.',
  tags=['incomplete_historical_shard'],scanned_file_bytes=partial.stat().st_size,known_consumers=[],archive_ready=False,
  evidence=[{'path':'docs/e2e_field_data_build_20260908.md','finding':'Operational record identifies interrupted shard excluded from manifests'},
            {'path':str(marker),'sha256':digest(marker),'finding':'Completed separate shard, not a scientific-domain qualification'}],
  replacement=m['shard'],replacement_scope='Historical numerical-completion replacement; not the new galaxy-label target'))
r['counts']=dict(Counter(u['status'] for u in r['units']))
r['logical_archive_count']=len(legacy)+1
r['physical_move_manifest']={'status':'not_ready','reason':'No bulk corrected replacements, complete external-consumer audit or active-job migration has been verified. Keep paths intact.'}
out=ROOT/'configs/product_lifecycle_20261019.json'
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
queue=ROOT/'docs/evidence/galaxy_position_allpop_v2_20261019';queue.mkdir(parents=True,exist_ok=True)
with (queue/'product_review_queue.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['id','path','status','family','logical_bytes_scanned','receipts','reason','tags'])
 for u in r['units']:w.writerow([u['id'],u['path'],u['status'],u['family'],u.get('scanned_file_bytes',''),u.get('receipt_count',''),u.get('reason',''),';'.join(u.get('tags',[]))])
with (queue/'retirement_candidates.json').open('x') as f:json.dump(r['retirement_candidates'],f,indent=2);f.write('\n')
report=ROOT/'docs/product_lifecycle_20261019.md'
with report.open('a') as f:
 f.write('\n## Completed inventory and tags\n\n')
 f.write(f"The registry covers {len(r['units'])} registered products/groups and {len(r['dependencies'])} grouped dependency links. {len(legacy)} legacy target products/groups plus one documented incomplete field shard are archived **in place**. {len(r['retirement_candidates'])} further retirement candidates require review. No files were moved or deleted.\n\n")
 f.write('| Status | Products/groups |\n| --- | ---: |\n')
 for status,count in r['counts'].items():f.write(f'| {status} | {count} |\n')
 f.write('\nThe exact paths and reasons are in `configs/product_lifecycle_20261019.json` and `docs/evidence/galaxy_position_allpop_v2_20261019/product_review_queue.csv`. The full raw metadata evidence is retained at `outputs/product_lifecycle_optimized_20261019.json`.\n')
print(json.dumps({'units':len(r['units']),'grouped_dependencies':len(r['dependencies']),'counts':r['counts'],'retirement_candidates':len(r['retirement_candidates'])},indent=2))
