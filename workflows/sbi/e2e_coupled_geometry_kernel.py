"""Equivalent support-only candidate kernel with one authority check per cap.

Centers are eight-cell aligned. Owned-core support means can therefore be
computed exactly from integer sums on an eight-cell grid, avoiding repeated
32-cubed reductions and hundreds of thousands of shared-filesystem metadata IOs.
"""
import argparse
from collections import Counter
import json
import time
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_geometry as geometry


def support_block_counts(support):
    shape=support.shape
    out=np.empty(tuple((n+7)//8 for n in shape),dtype=np.uint16)
    for start in range(0,shape[0],8):
        slab=support[start:start+8]
        if np.any((slab!=0)&(slab!=1)): raise ValueError('binary random support required')
        padded=np.pad(slab,[(0,8-len(slab)),(0,(-shape[1])%8),(0,(-shape[2])%8)])
        out[start//8]=padded.reshape(8,out.shape[1],8,out.shape[2],8).sum((0,2,4),dtype=np.uint16)
    return out


def candidate_buckets(phase,cap,record):
    c.phase_guard(phase)
    authority=c.sha256(c.CONFIG),c.sha256(coord.CONFIG)
    cfg=c.config()['geometry']; grid=coord.validate_grid(record['grid'])
    rng=np.random.default_rng(geometry.address(cfg['seed'],phase,cap,'pair-geometry'))
    n=cfg['candidates_per_cap']
    centers=np.column_stack([rng.integers(2,(side-margin)//8+1,n)*8
                             for side,margin in zip(grid['shape'],(96,64,64))])
    buckets={(shell,kind):[] for shell in range(4) for kind in ('interior','boundary')}
    rejected=Counter(); seen=set()
    with h5py.File(c.guarded(record['outputs'][0]['path'],phase),'r') as f:
        support=f['support_random']
        if list(support.shape)!=grid['shape']: raise ValueError('response grid shape mismatch')
        counts=support_block_counts(support)
    origin=np.asarray(grid['origin_mpc_h']); cell=grid['cell_mpc_h']
    for center in centers:
        key=tuple(map(int,center))
        if key in seen: rejected['duplicate']+=1; continue
        seen.add(key)
        midpoint=(origin+(center+[16,0,0])*cell-1000.)%2000.
        cores=(origin+np.stack((center,center+[32,0,0]))*cell-1000.)%2000.
        xyz=origin+(center+[16,0,0])*cell
        redshift=float(coord.redshift(np.linalg.norm(xyz)))
        shell=int(np.searchsorted((.15,.25,.35,.45,.55),redshift,side='right')-1)
        if not 0<=shell<4: rejected['outside_shell']+=1; continue
        fractions=[]
        for offset in (np.zeros(3,dtype=int),np.array([32,0,0])):
            start=(center+offset-16)//8
            fractions.append(float(counts[tuple(slice(v,v+4) for v in start)].sum(dtype=np.uint64))/32768.)
        if min(fractions)<.25: rejected['low_support']+=1; continue
        kind='interior' if min(fractions)>=.95 else 'boundary'
        buckets[shell,kind].append(dict(phase=phase,role=c.ROLES[phase],cap=cap,
            center=list(key),grid=grid,shell=shell,redshift=redshift,support_stratum=kind,
            science_core_support_fractions=fractions,source_midpoint_mpc_h=midpoint.tolist(),
            source_owned_core_centers_mpc_h=cores.tolist()))
    if authority!=(c.sha256(c.CONFIG),c.sha256(coord.CONFIG)):
        raise ValueError('coordinate/data authority changed during candidate generation')
    return buckets,dict(rejected=dict(rejected),eligible={f'{s}:{k}':len(v) for (s,k),v in buckets.items()})


def build(phase):
    """Same candidate seed/order and greedy ownership decisions as the reference."""
    c.require_compute(); c.phase_guard(phase); coord.bind(); coord.require_host_checks()
    started=time.monotonic(); directory=coord.ROOT/'geometry'/phase
    with c.single_writer(directory):
        marker=directory/'GEOMETRY_COMPLETE.json'
        if marker.exists():
            record=coord.verify_receipt(marker,payload=False)
            geometry.verify_rows(record['pairs'],phase)
            return record
        buckets={}; sources={}; diagnostics={}
        for cap in ('NGC','SGC'):
            source=coord.ROOT/'observations'/phase/f'{cap}_COMPLETE.json'
            record=coord.verify_receipt(source)
            if record['phase']!=phase or record['cap']!=cap: raise ValueError('response identity mismatch')
            candidate,diagnostics[cap]=candidate_buckets(phase,cap,record)
            buckets.update({(cap,*key):value for key,value in candidate.items()})
            sources[cap]=dict(receipt=str(source),sha256=c.sha256(source))
            print(json.dumps(dict(phase=phase,cap=cap,geometry_candidates_complete=True)),flush=True)
        quota=8 if c.ROLES[phase]=='train' else 1
        cursors=Counter(); selected=[]; mids=[]; cores=[]
        for ordinal in range(quota):
            for key,bucket in buckets.items():
                accepted=None
                while cursors[key]<len(bucket):
                    row=bucket[cursors[key]]; cursors[key]+=1
                    midpoint=np.asarray(row['source_midpoint_mpc_h']); pair=np.asarray(row['source_owned_core_centers_mpc_h'])
                    if geometry.admissible(midpoint,pair,mids,cores):
                        accepted=dict(row,pair_id=f'{phase}_{key[0]}_s{key[1]}_{key[2]}_{ordinal:02d}')
                        mids.append(midpoint); cores.extend(pair); break
                if accepted is None:
                    failure=dict(**coord.provenance(),phase=phase,failed_stratum=list(key),
                        failed_ordinal=ordinal,diagnostics=diagnostics,selected_pairs=len(selected),
                        source_code_sha256=c.sha256(__file__),**{'pass':False})
                    c.atomic_json(directory/f'GEOMETRY_FAILED_{time.time_ns()}.json',failure)
                    raise ValueError(f'insufficient nonoverlapping pair candidates: {phase} {key}')
                selected.append(accepted)
        record=dict(**coord.provenance(),phase=phase,role=c.ROLES[phase],sources=sources,
            source_code_sha256=c.sha256(__file__),reference_code_sha256=c.sha256(geometry.__file__),
            pairs=selected,diagnostics=diagnostics,qa=geometry.verify_rows(selected,phase),
            counts_or_targets_read=False,elapsed_seconds=time.monotonic()-started,**{'pass':True})
        c.atomic_json(marker,record)
        return record


def verify_existing_candidates(phase):
    """Full real-data replay against the originally committed slow selection."""
    c.require_compute(); c.phase_guard(phase)
    path=coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
    reference=coord.verify_receipt(path,payload=False)
    started=time.monotonic(); buckets={}; diagnostics={}
    for cap in ('NGC','SGC'):
        source=coord.ROOT/'observations'/phase/f'{cap}_COMPLETE.json'
        if c.sha256(source)!=reference['sources'][cap]['sha256']:
            raise ValueError('parity input differs from original geometry')
        record=coord.verify_receipt(source)
        rows,diagnostics[cap]=candidate_buckets(phase,cap,record)
        buckets.update({(cap,*key):value for key,value in rows.items()})
    if diagnostics!=reference['diagnostics']: raise ValueError('candidate diagnostic parity failed')
    cursors=Counter(); selected=[]; mids=[]; cores=[]
    for ordinal in range(8 if c.ROLES[phase]=='train' else 1):
        for key,bucket in buckets.items():
            accepted=None
            while cursors[key]<len(bucket):
                row=bucket[cursors[key]]; cursors[key]+=1
                midpoint=np.asarray(row['source_midpoint_mpc_h']); pair=np.asarray(row['source_owned_core_centers_mpc_h'])
                if geometry.admissible(midpoint,pair,mids,cores):
                    accepted=dict(row,pair_id=f'{phase}_{key[0]}_s{key[1]}_{key[2]}_{ordinal:02d}')
                    mids.append(midpoint); cores.extend(pair); break
            if accepted is None: raise ValueError('parity replay cannot select original panel')
            selected.append(accepted)
    if selected!=reference['pairs']: raise ValueError('selected real pair panel is not exactly equal')
    result=dict(**coord.provenance(),phase=phase,geometry_receipt_sha256=c.sha256(path),
        kernel_sha256=c.sha256(__file__),pairs=len(selected),all_pair_rows_exact=True,
        candidate_diagnostics_exact=True,elapsed_seconds=time.monotonic()-started,**{'pass':True})
    destination=path.parent/f'FAST_PARITY_{time.time_ns()}.json'
    c.atomic_json(destination,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',required=True)
    p.add_argument('--verify-existing',action='store_true')
    a=p.parse_args()
    result=verify_existing_candidates(a.phase) if a.verify_existing else build(a.phase)
    print(json.dumps({key:value for key,value in result.items()
                      if key not in ('pairs','sources','diagnostics')}),flush=True)
