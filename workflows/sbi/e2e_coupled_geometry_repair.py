"""Bounded support-only packing repair; never relax quotas, support or separation.

The original round-robin greedy selector can exhaust a scarce stratum after
placing less constrained domains. Failure of that heuristic is not a proof of
geometric infeasibility. Reorder the SAME finite candidate pool deterministically,
retaining the original output verbatim whenever its greedy pass succeeds.
"""
import argparse
from collections import Counter
import json
import time
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi import e2e_coupled_geometry_kernel as kernel

CONFIG=c.REPO/'configs/e2e_coupled_geometry_search_v1.json'


def select_once(buckets,quota,keys):
    cursors=Counter(); selected=[]; mids=[]; cores=[]
    for ordinal in range(quota):
        for key in keys:
            accepted=None; bucket=buckets[key]
            while cursors[key]<len(bucket):
                row=bucket[cursors[key]]; cursors[key]+=1
                midpoint=np.asarray(row['source_midpoint_mpc_h'])
                pair=np.asarray(row['source_owned_core_centers_mpc_h'])
                if geometry.admissible(midpoint,pair,mids,cores):
                    accepted=dict(row,pair_id=f"{row['phase']}_{key[0]}_s{key[1]}_{key[2]}_{ordinal:02d}")
                    mids.append(midpoint); cores.extend(pair); break
            if accepted is None:
                return None,dict(failed_stratum=list(key),failed_ordinal=ordinal,
                                 selected_pairs=len(selected))
            selected.append(accepted)
    return selected,dict(selected_pairs=len(selected))


def select(buckets,quota,phase,cfg):
    original=list(buckets)
    scarcity=sorted(buckets,key=lambda k:(len(buckets[k]),k))
    reports=[]
    for attempt in range(2+cfg['maximum_seeded_attempts']):
        candidate=buckets; keys=original if attempt==0 else scarcity
        mode='original' if attempt==0 else 'scarcity_original_order'
        if attempt>=2:
            mode='scarcity_seeded_order'
            rng=np.random.default_rng(geometry.address(c.config()['geometry']['seed'],phase,
                                                       cfg['seed_namespace'],attempt-2))
            candidate={k:[buckets[k][i] for i in rng.permutation(len(buckets[k]))] for k in original}
        rows,report=select_once(candidate,quota,keys)
        reports.append(dict(attempt=attempt,mode=mode,success=rows is not None,**report))
        if rows is not None:
            # Canonical ordinal/stratum order, independent of packing order.
            order={k:i for i,k in enumerate(original)}
            rows.sort(key=lambda r:(int(r['pair_id'].split('_')[-1]),
                                   order[(r['cap'],r['shell'],r['support_stratum'])]))
            return rows,reports
    return None,reports


def build(phase):
    c.require_compute(); c.phase_guard(phase); coord.bind(); coord.require_host_checks()
    cfg=json.loads(CONFIG.read_text())
    if (cfg['schema']!='e2e-coupled-geometry-search-v1' or cfg['candidate_pool_changed']
            or cfg['quotas_or_acceptance_criteria_changed'] or cfg['counts_or_targets_allowed']
            or cfg['scientific_training_authorized'] or cfg['maximum_seeded_attempts']!=16):
        raise ValueError('unregistered geometry feasibility policy')
    started=time.monotonic(); directory=coord.ROOT/'geometry'/phase
    with c.single_writer(directory):
        final=directory/'GEOMETRY_COMPLETE.json'
        if final.exists():
            record=coord.verify_receipt(final,payload=False)
            geometry.verify_rows(record['pairs'],phase)
            return record
        buckets={}; sources={}; diagnostics={}
        for cap in ('NGC','SGC'):
            source=coord.ROOT/'observations'/phase/f'{cap}_COMPLETE.json'
            record=coord.verify_receipt(source)
            if record['phase']!=phase or record['cap']!=cap: raise ValueError('response identity mismatch')
            rows,diagnostics[cap]=kernel.candidate_buckets(phase,cap,record)
            buckets.update({(cap,*key):value for key,value in rows.items()})
            sources[cap]=dict(receipt=str(source),sha256=c.sha256(source))
        rows,attempts=select(buckets,8 if c.ROLES[phase]=='train' else 1,phase,cfg)
        evidence=dict(**coord.provenance(),phase=phase,role=c.ROLES[phase],sources=sources,
            source_code_sha256=c.sha256(__file__),candidate_kernel_sha256=c.sha256(kernel.__file__),
            reference_code_sha256=c.sha256(geometry.__file__),selection_config_sha256=c.sha256(CONFIG),
            diagnostics=diagnostics,packing_attempts=attempts,counts_or_targets_read=False,
            elapsed_seconds=time.monotonic()-started)
        if rows is None:
            c.atomic_json(directory/f'PACKING_FAILED_{time.time_ns()}.json',dict(**evidence,**{'pass':False}))
            raise ValueError('bounded packing failed without weakening the registered geometry')
        result=dict(**evidence,pairs=rows,qa=geometry.verify_rows(rows,phase),**{'pass':True})
        c.atomic_json(final,result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',required=True)
    a=p.parse_args(); result=build(a.phase)
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','sources','diagnostics')}),flush=True)
