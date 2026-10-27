#!/usr/bin/env python3
"""Inspect failed ph007 native-particle identity without changing qualification.

One existing audit and one field-A slab only. This diagnostic never changes the
coordinate transform, tolerance, certificate, catalogue, or training readiness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
from galaxy_position_mapping import periodic_delta, field_indices
from validate_galaxy_position_mapping import Field

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/ph007')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not (os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):
        parser.error('Approved CPU compute step required')
    if args.output.exists():
        raise FileExistsError(args.output)
    audit_path=ROOT/'qualification_audit/report.json'
    sample_path=ROOT/'qualification_audit/ph007_sample.npz'
    audit=json.loads(audit_path.read_text())
    info=audit['phases']['ph007']
    if not audit['completed'] or audit['tolerance_mpc_h'] != .005:
        raise ValueError('Unexpected audit or tolerance')
    checks=[c for c in info['native_checks'] if c['population']=='unresolved_central' and c['slab']==33]
    if len(checks)!=1 or checks[0]['accepted']!=139 or checks[0]['rows']!=140:
        raise ValueError('Diagnostic does not match the frozen failed case')
    source=Path(checks[0]['source']['path']);stat=source.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(checks[0]['source']['bytes'],checks[0]['source']['mtime_ns']):
        raise ValueError('Native source changed')
    with np.load(sample_path,allow_pickle=False) as f:
        s={k:f[k] for k in f.files}
    mask=(s['BOX_INDEX']<0)&(s['CEN']==1)&(s['RES']==0)&(s['FILE_NUM']==33)&(s['reference_kind']=='uncovered')
    failed=np.flatnonzero(mask)
    if len(failed)!=1 or int(s['original_row'][failed[0]])!=64130602:
        raise ValueError('Unexpected failed sample identity')
    i=int(failed[0]);position=s['reconstructed_native'][i]
    from abacusnbody.data.read_abacus import read_asdf
    table=read_asdf(source,load=['pos'],verbose=False)
    xyz=np.asarray(table['pos'])
    rows=[];points=[];distances=[];nearest=[]
    for start in range(0,len(xyz),1_000_000):
        block=xyz[start:start+1_000_000]
        distance=np.linalg.norm(periodic_delta(block,position),axis=1)
        local=np.flatnonzero(distance < audit['tolerance_mpc_h'])
        rows.extend((local+start).tolist());points.extend(block[local].tolist());distances.extend(distance[local].tolist())
        ids=np.argpartition(distance,min(7,len(distance)-1))[:8]
        nearest.extend((float(distance[j]),int(start+j)) for j in ids)
        nearest=sorted(nearest)[:8]
    candidates=np.asarray(points,dtype=float).reshape(-1,3)
    voxels=field_indices(candidates).tolist() if len(candidates) else []
    query_voxel=field_indices(position[None,:])[0].tolist()
    field=Field('ph007',Path(info['field_path']))
    eigenvalues=field.sample(candidates).tolist() if len(candidates) else []
    query_eigenvalues=field.sample(position[None,:])[0].tolist()
    report=dict(completed=True,diagnostic_only=True,qualification_changed=False,training_ready=False,
                job_id=os.environ['SLURM_JOB_ID'],script_sha256=digest(__file__),
                audit_sha256=digest(audit_path),sample_sha256=digest(sample_path),
                source=checks[0]['source'],original_row=int(s['original_row'][i]),
                query_position=position.tolist(),tolerance_mpc_h=audit['tolerance_mpc_h'],
                nearest_eight=[dict(distance_mpc_h=d,source_row=j) for d,j in nearest],
                candidates_within_tolerance=len(rows),candidate_source_rows=rows,
                candidate_positions=candidates.tolist(),candidate_distances_mpc_h=distances,
                candidate_voxels=voxels,query_voxel=query_voxel,
                candidate_eigenvalues=eigenvalues,query_eigenvalues=query_eigenvalues,
                distinct_candidate_positions=len(np.unique(candidates,axis=0)),
                all_candidates_same_query_voxel=bool(len(voxels) and all(v==query_voxel for v in voxels)),
                all_candidates_same_query_eigenvalues=bool(len(eigenvalues) and all(v==query_eigenvalues for v in eigenvalues)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':main()
