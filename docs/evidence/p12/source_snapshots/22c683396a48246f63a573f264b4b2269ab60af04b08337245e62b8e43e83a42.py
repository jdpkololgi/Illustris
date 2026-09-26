"""Verify only the Loa full data/random inputs needed for observation fields."""
import argparse,json,hashlib,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def check(entry):
    p=Path(entry['path']);s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    t=p.stat();got=h.hexdigest();r=dict(path=str(p),bytes=s.st_size,sha256=got,pass_checks=got==entry['sha256'] and s.st_size==entry['bytes'] and (s.st_size,s.st_mtime_ns)==(t.st_size,t.st_mtime_ns))
    print(p.name,r['pass_checks'],flush=True);return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('compute allocation required')
    raw=a.registry.read_bytes();d=json.loads(raw)['desi_candidate'];entries=[d['data']['full']]+d['full_random']
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check,entries))
    a.output.write_text(json.dumps(dict(records=rows,pass_checks=all(r['pass_checks'] for r in rows),registry_sha256=hashlib.sha256(raw).hexdigest(),job=os.environ['SLURM_JOB_ID'],clustering_randoms_used=False),indent=2)+'\n')
