"""Random-support-only adjacent-pair selection with periodic ownership checks."""
import argparse
from collections import Counter
import hashlib
import json

import h5py
import numpy as np

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord


def address(*parts):
    return int(hashlib.sha256(':'.join(map(str,parts)).encode()).hexdigest()[:16],16)%(2**63-1)


def source_position(center,grid):
    coord.validate_grid(grid)
    return (np.asarray(grid['origin_mpc_h'])+np.asarray(center)*grid['cell_mpc_h']-1000.)%2000.


def periodic_delta(a,b):
    return (np.asarray(a)-np.asarray(b)+1000.)%2000.-1000.


def domain_positions(center,grid):
    center=np.asarray(center)
    cores=np.stack([source_position(center,grid),source_position(center+[32,0,0],grid)])
    midpoint=source_position(center+[16,0,0],grid)
    return midpoint,cores


def admissible(midpoint,cores,prior_midpoints,prior_cores):
    if not len(prior_midpoints):
        return True
    distances=np.linalg.norm(periodic_delta(midpoint,prior_midpoints),axis=-1)
    if np.any(distances<162.384-1e-8):
        return False
    difference=np.abs(periodic_delta(cores[:,None,:],np.asarray(prior_cores)[None,:,:]))
    # A shared face is allowed, a shared positive-volume science voxel is not.
    return not bool(np.any(np.all(difference<108.256-1e-8,axis=-1)))


def candidate_buckets(phase,cap,record):
    geometry=c.config()['geometry']
    grid=record['grid']
    rng=np.random.default_rng(address(geometry['seed'],phase,cap,'pair-geometry'))
    n=geometry['candidates_per_cap']
    centers=np.column_stack([rng.integers(2,(side-margin)//8+1,n)*8
                             for side,margin in zip(grid['shape'],(96,64,64))])
    buckets={(shell,kind):[] for shell in range(4) for kind in ('interior','boundary')}
    rejected=Counter()
    # Deliberately open only support_random; this function must not read counts.
    with h5py.File(c.guarded(record['outputs'][0]['path'],phase),'r') as f:
        support=f['support_random'][:].astype(bool)
    if list(support.shape)!=grid['shape']:
        raise ValueError('response grid shape mismatch')
    seen=set()
    for center in centers:
        key=tuple(map(int,center))
        if key in seen:
            rejected['duplicate']+=1; continue
        seen.add(key)
        midpoint,cores=domain_positions(center,grid)
        xyz=np.asarray(grid['origin_mpc_h'])+(center+[16,0,0])*grid['cell_mpc_h']
        redshift=float(coord.redshift(np.linalg.norm(xyz)))
        shell=int(np.searchsorted((.15,.25,.35,.45,.55),redshift,side='right')-1)
        if not 0<=shell<4:
            rejected['outside_shell']+=1; continue
        fractions=[]
        for offset in (np.zeros(3,dtype=int),np.array([32,0,0])):
            slices=tuple(slice(int(v)-16,int(v)+16) for v in center+offset)
            fractions.append(float(support[slices].mean()))
        if min(fractions)<.25:
            rejected['low_support']+=1; continue
        kind='interior' if min(fractions)>=.95 else 'boundary'
        buckets[shell,kind].append(dict(phase=phase,role=c.ROLES[phase],cap=cap,
            center=list(key),grid=grid,shell=shell,redshift=redshift,support_stratum=kind,
            science_core_support_fractions=fractions,source_midpoint_mpc_h=midpoint.tolist(),
            source_owned_core_centers_mpc_h=cores.tolist()))
    return buckets,dict(rejected=dict(rejected),eligible={f'{s}:{k}':len(v) for (s,k),v in buckets.items()})


def verify_rows(rows,phase):
    c.phase_guard(phase)
    target=128 if c.ROLES[phase]=='train' else 16
    counts=Counter((row['cap'],row['shell'],row['support_stratum']) for row in rows)
    keys={(cap,shell,kind) for cap in ('NGC','SGC') for shell in range(4)
          for kind in ('interior','boundary')}
    if len(rows)!=target or set(counts)!=keys or set(counts.values())!={target//16}:
        raise ValueError('pair panel/strata are incomplete')
    mids,cores=[],[]
    for row in rows:
        if row['phase']!=phase or row['role']!=c.ROLES[phase]:
            raise ValueError('pair phase/role mismatch')
        fractions=np.asarray(row['science_core_support_fractions'])
        if (fractions.shape!=(2,) or not np.isfinite(fractions).all()
                or np.min(fractions)<.25 or np.max(fractions)>1.
                or (np.min(fractions)>=.95) != (row['support_stratum']=='interior')):
            raise ValueError('invalid owned-core support fractions')
        midpoint,pair=domain_positions(row['center'],row['grid'])
        if not np.allclose(midpoint,row['source_midpoint_mpc_h'],atol=1e-9,rtol=0):
            raise ValueError('source midpoint mismatch')
        if not np.allclose(pair,row['source_owned_core_centers_mpc_h'],atol=1e-9,rtol=0):
            raise ValueError('source core mismatch')
        if not admissible(midpoint,pair,mids,cores):
            raise ValueError('duplicate/overlapping source science regions')
        mids.append(midpoint); cores.extend(pair)
    return dict(pairs=target,owned_cores=2*target,source_nonoverlap=True,
                midpoint_separation_mpc_h=162.384,stratum_counts={str(k):v for k,v in counts.items()})


def build(phase):
    c.require_compute(); c.phase_guard(phase); coord.bind(); coord.require_host_checks()
    directory=coord.ROOT/'geometry'/phase
    with c.single_writer(directory):
        marker=directory/'GEOMETRY_COMPLETE.json'
        if marker.exists():
            result=coord.verify_receipt(marker,payload=False)
            verify_rows(result['pairs'],phase)
            return result
        buckets,sources,diagnostics={},{},{}
        for cap in ('NGC','SGC'):
            source=coord.ROOT/'observations'/phase/f'{cap}_COMPLETE.json'
            record=coord.verify_receipt(source)
            if record['phase']!=phase or record['cap']!=cap:
                raise ValueError('response identity mismatch')
            candidate,diagnostics[cap]=candidate_buckets(phase,cap,record)
            buckets.update({(cap,*key):value for key,value in candidate.items()})
            sources[cap]=dict(receipt=str(source),sha256=c.sha256(source))
        per_stratum=8 if c.ROLES[phase]=='train' else 1
        cursors=Counter(); selected=[]; mids=[]; cores=[]
        for ordinal in range(per_stratum):
            for key,bucket in buckets.items():
                accepted=None
                while cursors[key]<len(bucket):
                    row=bucket[cursors[key]]; cursors[key]+=1
                    midpoint=np.asarray(row['source_midpoint_mpc_h'])
                    pair=np.asarray(row['source_owned_core_centers_mpc_h'])
                    if admissible(midpoint,pair,mids,cores):
                        accepted=dict(row,pair_id=f'{phase}_{key[0]}_s{key[1]}_{key[2]}_{ordinal:02d}')
                        mids.append(midpoint); cores.extend(pair); break
                if accepted is None:
                    raise ValueError(f'insufficient nonoverlapping pair candidates: {phase} {key}')
                selected.append(accepted)
        qa=verify_rows(selected,phase)
        record=dict(**coord.provenance(),phase=phase,role=c.ROLES[phase],sources=sources,
                    source_code_sha256=c.sha256(__file__),pairs=selected,
                    diagnostics=diagnostics,qa=qa,counts_or_targets_read=False,**{'pass':True})
        c.atomic_json(marker,record)
        return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',required=True)
    a=p.parse_args(); print(json.dumps(build(a.phase)))
