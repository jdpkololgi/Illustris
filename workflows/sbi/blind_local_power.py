"""Matched-window density auto and cross power; no model fitting."""
from pathlib import Path
import json, os
import numpy as np
import h5py

d = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/lowz_cfm_blind_confirmation_v1')
out = d.parent / ('blind_local_power_' + os.environ['SLURM_JOB_ID'])
out.mkdir(exist_ok=True)
shape = (32, 16, 16)
k = np.sqrt(sum(f*f for f in np.meshgrid(*[2*np.pi*np.fft.fftfreq(n,6.766) for n in shape], indexing='ij')))
edges = [.025,.10,.20,.30,.363,.464]
for ph in ['ph018','ph019']:
    records = []
    truth = np.load(d/'blind_score_61911489'/f'{ph}_truth.npy')
    for i,row in enumerate(json.loads((d/'GEOMETRY_FROZEN.json').read_text())['phases'][ph]['pairs']):
        cap = row['cap']
        mask = np.zeros(shape,bool)
        with h5py.File(d/'matched_reference_v1'/f'{ph}_{cap}.h5') as ref:
            for g in ref[row['pair_id']].values():
                mask.flat[g['flat_cell'][:]] = True
        def fft(a):
            assert np.isfinite(a[mask]).all()
            return np.fft.fftn(np.where(mask,a-a[mask].mean(),0))
        y = fft(truth[i].reshape(*shape,4)[...,0])
        bins = [(k>=lo)&(k<hi) for lo,hi in zip(edges,edges[1:])]
        draws = []
        for j in range(32):
            with h5py.File(d/ph/cap/'predictions'/f'draw_{j:03d}'/'density/FIELD.h5') as f:
                loc = (np.array(row['grid']['origin_mpc_h'])+(np.array(row['center'])-16)*3.383-f.attrs['origin_mpc_h'])/6.766
                assert np.allclose(loc,loc.round())
                x = fft(f['rho'][(0,*[slice(v,v+n) for v,n in zip(loc.round().astype(int),shape)])])
            draws.append([[float(np.sum(abs(x[b])**2)/np.sum(abs(y[b])**2)),float(np.vdot(x[b],y[b]).real/(np.linalg.norm(x[b])*np.linalg.norm(y[b])))] for b in bins])
        records.append(dict(pair=row['pair_id'],cells=int(mask.sum()),draws=draws))
    assert sum(r['cells'] for r in records)=={'ph018':109144,'ph019':121818}[ph]
    (out/f'{ph}.json').write_text(json.dumps(dict(edges=edges,records=records),allow_nan=False))
    print(ph,np.median([np.mean(r['draws'],axis=0) for r in records],axis=0),flush=True)
