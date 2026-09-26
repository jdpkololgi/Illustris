"""Recover raw CEN/RES flags for the already-frozen training native-label sample."""
import argparse
import hashlib
import json
from pathlib import Path
import fitsio
import numpy as np
from workflows.abacus_tweb.p10_build_bright_parent import make_keys
from workflows.abacus_tweb.p12a_coordinate_sample_audit import require_compute, record

ROOT=Path(__file__).resolve().parents[2]

def audit(phase):
    require_compute()
    receipt=ROOT/f'docs/evidence/p12/P12A_COORDINATE_SAMPLE_{phase.upper()}_20260924_v2.json'
    r=json.loads(receipt.read_text()); n=r['numerical']
    indices=np.asarray(n['native']['observed_sample_indices'])
    rows=np.asarray(n['sampled_canonical_rows'])[indices]
    with fitsio.FITS(r['inputs']['observed']['path']) as f:
        observed=f[1].read(rows=rows,columns=['RA','DEC','Z','FILE_NUM','HALO_INDEX','BOX_INDEX'])
    keys=make_keys(observed['RA'],observed['DEC'],observed['Z'])
    order=np.argsort(keys); keys=keys[order]
    if np.any(keys[1:]==keys[:-1]): raise ValueError('duplicate sampled keys')
    registry=json.loads((ROOT/'configs/p10_phase_registry_v1.json').read_text())
    raw=Path(registry['path_templates']['cutsky'].format(phase=phase))
    before=record(raw); matches=np.zeros(len(keys),dtype=int); flags=[]
    with fitsio.FITS(raw) as f:
        for start in range(0,f[1].get_nrows(),1000000):
            rows_raw=np.arange(start,min(start+1000000,f[1].get_nrows()))
            d=f[1].read(rows=rows_raw,columns=['RA','DEC','Z','CEN','RES','FILE_NUM','HALO_INDEX','BOX_INDEX'])
            k=make_keys(d['RA'],d['DEC'],d['Z']); pos=np.searchsorted(keys,k)
            inside=np.flatnonzero(pos<len(keys)); hit=inside[keys[pos[inside]]==k[inside]]
            for j in hit:
                at=int(order[pos[j]]); matches[pos[j]]+=1
                host_match=all(observed[name][at]==d[name][j] for name in ['FILE_NUM','HALO_INDEX','BOX_INDEX'])
                flags.append(dict(sample_index=at,canonical_row=int(rows[at]),raw_row=int(rows_raw[j]),CEN=int(d['CEN'][j]),RES=int(d['RES'][j]),host_match=bool(host_match)))
    if before!=record(raw): raise ValueError('raw source changed')
    counts={}
    for row in flags:
        key=f"CEN={row['CEN']},RES={row['RES']}"; counts[key]=counts.get(key,0)+1
    return dict(schema='p12a-population-sample-v1',phase=phase,source=record(__file__,small=True),input=before,receipt=record(receipt,small=True),sample_rows=len(rows),unique_raw_match=bool(np.all(matches==1)),host_match=all(x['host_match'] for x in flags),counts=counts,records=flags,ready_for_desi_canary=False,scope='Raw flags for frozen native-label sample; flag semantics and absent populations require explicit treatment')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['ph002','ph003','ph004','ph005'],required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    r=audit(args.phase)
    with args.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print({k:r[k] for k in ['phase','unique_raw_match','host_match','counts']})
    raise SystemExit(0 if r['unique_raw_match'] and r['host_match'] else 1)
