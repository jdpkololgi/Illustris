"""Bounded real-condition qualification before continuing the first phase."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_condition_products as products
from workflows.sbi import e2e_coupled_conditions as reader
from workflows.sbi import e2e_coupled_operators as op


def qualify(phase):
    c.require_compute(); c.phase_guard(phase)
    geo_path=coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
    geo=coord.verify_receipt(geo_path,payload=False)
    folder=coord.ROOT/'conditions'/phase
    checks=[]
    for row in geo['pairs']:
        marker=folder/f"{row['pair_id']}.json"
        if not marker.exists(): continue
        arrays=reader.load_pair(phase,row['pair_id'])
        source=coord.verify_receipt(coord.ROOT/'observations'/phase/f"{row['cap']}_COMPLETE.json",payload=False)
        with h5py.File(source['outputs'][0]['path'],'r') as raw:
            for side,(offset,crop) in enumerate(zip(([0,0,0],[32,0,0]),op.layout()['owned_core_crops_in_joint'])):
                center=np.array(row['center'])+offset
                sl=tuple(slice(v-16,v+16) for v in center)
                expected=np.log1p(raw['counts'][sl]).reshape(16,2,16,2,16,2).mean((1,3,5))
                owned=tuple(slice(*s) for s in crop)
                actual=arrays['joint'][0][owned]
                if not np.array_equal(actual,expected):
                    raise ValueError('owned-core count-channel coordinate mismatch')
                support=float(arrays['support'][owned].mean(dtype=np.float64))
                if support!=row['science_core_support_fractions'][side]:
                    raise ValueError('owned-core support differs from frozen geometry')
        checks.append(dict(pair_id=row['pair_id'],payload=c.file_record(marker,content_hash=True),
                           count_channel_exact=True,support_exact=True,targetless_reader_pass=True))
    if len(checks)<2: raise ValueError('two real pair shards required for initial qualification')
    result=dict(**coord.provenance(),phase=phase,geometry_sha256=c.sha256(geo_path),checks=checks,
        condition_builder_sha256=c.sha256(products.__file__),reader_sha256=c.sha256(reader.__file__),
        source_sha256=c.sha256(__file__),science_scores_evaluated=False,**{'pass':True})
    output=folder/'REAL_CONDITION_SMOKE.json'
    if output.exists():
        previous=coord.verify_receipt(output,payload=False)
        if previous['condition_builder_sha256']!=result['condition_builder_sha256']:
            raise ValueError('real smoke builder drift')
    else:
        c.atomic_json(output,result)
    return result


def run(phase,seconds):
    c.require_compute(); c.phase_guard(phase)
    frozen=json.loads((c.REPO/'SOURCE.json').read_text())
    for item in frozen['files']:
        if c.sha256(c.REPO/item['relative'])!=item['sha256']:
            raise ValueError('frozen product source changed')
    deadline=time.monotonic()+seconds
    marker=coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
    while not marker.exists():
        if time.monotonic()>deadline-600:
            return dict(phase=phase,paused=True,waiting_for=str(marker))
        print(json.dumps(dict(phase=phase,waiting_for=str(marker))),flush=True)
        time.sleep(30)
    products.build(phase,limit=2)
    proof=qualify(phase)
    print(json.dumps(dict(phase=phase,real_condition_smoke_pass=proof['pass'])),flush=True)
    if time.monotonic()>deadline-600:
        return dict(phase=phase,paused=True,smoke_pass=True)
    return products.build(phase)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',default='ph007')
    p.add_argument('--seconds',type=float,default=9000)
    a=p.parse_args(); result=run(a.phase,a.seconds)
    print(json.dumps(result),flush=True); raise SystemExit(75 if result.get('paused') else 0)
