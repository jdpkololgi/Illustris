"""Per-phase, equal-region readout of lowz_spatial_venue outputs."""
import json, sys
from pathlib import Path
import numpy as np

for phase in ('ph006','ph012','ph013'):
    rows=json.loads((Path(sys.argv[1])/(phase+'.json')).read_text())
    assert len(rows)==len({r['pair'] for r in rows})==16
    peaks=np.array([r['peak'] for r in rows if 'peak' in r])
    print(phase,'peak windows',len(peaks),'mean offsets/peak heights',peaks.mean(0))
    for lag in ('1','2','4'):
        metrics=[]
        for row in rows:
            truth,*models=row['stats'][lag]; t=truth[0];scores=[]
            for model in models:
                v=np.array(model); n=len(v); assert n==32 and np.isfinite(v).all()
                fair=np.abs(v-t).mean()-np.abs(v[:,None]-v).sum()/(2*n*(n-1))
                lo,hi=np.quantile(v,[.05,.95]);scores.append([fair,lo<=t<=hi,hi-lo])
            metrics.append(scores)
        print('lag',lag,'CFM / shuffled / spatial; fair CRPS, coverage, width')
        print(np.round(np.mean(metrics,axis=0),5))
