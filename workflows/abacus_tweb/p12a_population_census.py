"""Read-only training population census, including unresolved raw CutSky flags."""
import argparse, hashlib, json
from pathlib import Path
import fitsio
import numpy as np
from workflows.abacus_tweb.p12a_coordinate_sample_audit import require_compute,record
ROOT=Path(__file__).resolve().parents[2]

def audit(phase):
    require_compute()
    registry=json.loads((ROOT/'configs/p10_phase_registry_v1.json').read_text())
    raw=Path(registry['path_templates']['cutsky'].format(phase=phase));before=record(raw)
    totals={}; negative={}; examples=[]
    with fitsio.FITS(raw) as f:
        for start in range(0,f[1].get_nrows(),1000000):
            d=f[1].read(rows=range(start,min(start+1000000,f[1].get_nrows())),columns=['R_MAG_APP','Z','CEN','RES','FILE_NUM','HALO_INDEX','BOX_INDEX'])
            bright=d['R_MAG_APP']<19.5
            for name,m in [('bright_all',bright),('bright_active',bright&(d['Z']>=.15)&(d['Z']<.55))]:
                t=totals.setdefault(name,{})
                for cen,res in np.unique(np.column_stack((d['CEN'][m],d['RES'][m])),axis=0):
                    k=f'CEN={cen},RES={res}';t[k]=t.get(k,0)+int((m&(d['CEN']==cen)&(d['RES']==res)).sum())
                n=negative.setdefault(name,{k:0 for k in ['FILE_NUM','HALO_INDEX','BOX_INDEX']})
                for k in n:n[k]+=int((m&(d[k]<0)).sum())
            rare=bright&(d['Z']>=.15)&(d['Z']<.55)&(d['RES']==0)
            for i in np.flatnonzero(rare)[:max(0,16-len(examples))]:
                examples.append(dict(raw_row=int(start+i),**{k:float(d[k][i]) for k in d.dtype.names}))
            if start%10000000==0:print(phase,start,flush=True)
    if before!=record(raw):raise ValueError('source changed')
    return dict(schema='p12a-population-census-v1',phase=phase,input=before,source=record(__file__,small=True),counts=totals,negative_host_keys=negative,unresolved_active_examples=examples,scope='Raw bright population before footprint/assignment; no assumption that all these rows enter P12',ready_for_desi_canary=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phases',nargs='+',choices=['ph000','ph002','ph003','ph004','ph005'],required=True);a=p.parse_args()
    for phase in a.phases:
        out=ROOT/f'docs/evidence/p12/P12A_POPULATION_CENSUS_{phase.upper()}_20260924.json'
        if out.exists():raise FileExistsError(out)
        r=audit(phase)
        with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
        print(r['counts'],flush=True)
