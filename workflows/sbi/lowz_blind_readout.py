"""Aggregate frozen per-pair scores; never treat draws or cells as independent phases."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(directory):
    complete=json.loads((directory/'SCORES_COMPLETE.json').read_text())
    assert complete['passed'] and set(complete['phases'])=={'ph018','ph019'}
    totals={('ph018','NGC'):56031,('ph018','SGC'):53113,
            ('ph019','NGC'):62201,('ph019','SGC'):59617}
    groups=defaultdict(list)
    index={}
    for phase,h in complete['phases'].items():
        path=directory/(phase+'_scores.json')
        assert digest(path)==h
        for r in json.loads(path.read_text())['rows']:
            assert r['phase']==phase and r['weight']>0
            key=(phase,r['pair'],r['cap'],r['shell'],r['support'],r['weighting'])
            entry=index.setdefault(key,{})
            assert r['method'] not in entry
            entry[r['method']]=r
            for ph in ('all',phase):
                for shell in ('all',r['shell']):
                    for support in ('all',r['support']):
                        for cap in ('all',r['cap']):
                            groups[ph,cap,shell,support,r['weighting'],r['method']].append(r)
    for pair in index.values():
        a,b=pair['baseline'],pair['cfm']
        assert a['weight']==b['weight'] and a['cells']==b['cells']
    for (phase,cap),n in totals.items():
        assert sum(r['cells'] for r in groups[phase,cap,'all','all','cells','cfm'])==n
    report=[]
    for key,rows in groups.items():
        weights=[r['weight'] for r in rows]
        average=lambda items: np.average(items,axis=0,weights=weights).tolist()
        data={k:average([r[k] for r in rows]) for k in ('fair_crps','coverage90','width90','bias')}
        data['order_intervals']={k:dict(nominal=rows[0]['intervals'][k]['nominal'],
            coverage=average([r['intervals'][k]['coverage'] for r in rows]),
            width=average([r['intervals'][k]['width'] for r in rows])) for k in ('1','3','8')}
        report.append(dict(zip(('phase','cap','shell','support','weighting','method'),key),
            cells=sum(r['cells'] for r in rows),weight=sum(weights),**data,
            ranks=np.sum([r['rank_counts'] for r in rows],axis=0).tolist(),
            fair_brier=average([[v['fair_brier'] for v in r['classes']] for r in rows]),
            class_reliability={k:np.sum([[v['reliability'][k] for v in r['classes']] for r in rows],axis=0).tolist()
                               for k in ('weight','sum_probability','sum_truth')}))
    return dict(score_receipt=digest(directory/'SCORES_COMPLETE.json'),fields=['rho','lambda1','lambda2','lambda3'],
                class_order=['void','sheet','filament','knot'],interval='empirical 5th and 95th percentiles',
                independent_units=['ph018','ph019'],rows=report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('directory',type=Path)
    parser.add_argument('--paper-table',type=Path,help='Also save compact matched rows for the manuscript')
    args=parser.parse_args();d=args.directory
    result=summarize(d)
    with (d/'READOUT.json').open('x') as f: json.dump(result,f,allow_nan=False)
    if args.paper_table:
        keep=('phase','cap','shell','support','weighting','method','cells','weight','fair_crps',
              'coverage90','width90','bias','fair_brier')
        table=[{**{k:r[k] for k in keep},'coverage_27_of_33':r['order_intervals']['3']['coverage']}
               for r in result['rows'] if r['support']==r['cap']=='all']
        with args.paper_table.open('x') as f:
            json.dump({**{k:v for k,v in result.items() if k!='rows'},'rows':table},f,indent=1,allow_nan=False)
