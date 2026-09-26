"""Small saved-score diagnostic; no targets or field arrays are loaded."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

def summarize(rows):
    result={'n_cases':len(rows),'n_pairs':len({r['pair'] for r in rows})}
    for group in ('core_mass','block_mass','fine_mass','tidal','eigengap'):
        s=[r['scores'][group] for r in rows]
        spread=math.sqrt(mean(x['posterior_variance'] for x in s))
        result[group]={'coverage90':mean(x['coverage']['0.9'] for x in s),
                       'crps':mean(x['crps'] for x in s),
                       'rmse_over_spread':math.sqrt(mean(x['mean_squared_error'] for x in s))/spread,
                       'bias_over_spread':mean(x['bias'] for x in s)/spread}
    return result

def analyze(path):
    path=Path(path);data=json.loads(path.read_text())
    for filename,checksum in data['input_hashes'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=checksum:
            raise ValueError('input receipt changed: '+filename)
    rows=[r for r in data['rows'] if r['arm']=='mixed']
    expected={(p,s) for p in ('ph012','ph013','ph014','ph015') for s in (17,29)}
    if {(r['phase'],r['seed']) for r in rows}!=expected:raise ValueError('unexpected panel')
    for phase,seed in expected:
        ids=[r['pair'].split('_',1)[1] for r in rows if (r['phase'],r['seed'])==(phase,seed)]
        want={f'{cap}_s{shell}_{support}_00' for cap in ('NGC','SGC') for shell in range(4) for support in ('boundary','interior')}
        if len(ids)!=16 or set(ids)!=want:raise ValueError('unbalanced matched strata')
    out={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'interpretation':'Exploratory descriptive cuts; correlated anchors and repeated seeds are not independent replicates.',
         'phase':{},'strata':{},'paired_013_minus_012':{}}
    for phase in sorted({r['phase'] for r in rows}):
        selected=[r for r in rows if r['phase']==phase];out['phase'][phase]=summarize(selected)
        out['strata'][phase]={}
        for name,index,values in [('cap',1,('NGC','SGC')),('shell',2,('s0','s1','s2','s3')),('support',3,('boundary','interior'))]:
            for value in values:
                out['strata'][phase][name+'_'+value]=summarize([r for r in selected if r['pair'].split('_')[index]==value])
    for key in out['strata']['ph012']:
        out['paired_013_minus_012'][key]=out['strata']['ph013'][key]['fine_mass']['coverage90']-out['strata']['ph012'][key]['fine_mass']['coverage90']
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');args=p.parse_args()
    print(json.dumps(analyze(args.source),indent=2))
