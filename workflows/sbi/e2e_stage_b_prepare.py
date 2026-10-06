"""Source-bound galaxy-to-cell alignment and train-only prevalence for Stage B."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_stage_a_calibration as a
b=a.b;c=b.c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi import e2e_coupled_conditions as conditions
from workflows.sbi.e2e_coupled_product_audit import read_target
from workflows.abacus_tweb.p3br_build_random_response import galactic_cap
import fitsio
import h5py
import numpy as np

CONFIG=Path(__file__).resolve().parents[2]/'configs/e2e_stage_b_20261006.json'
ROOT=a.ROOT/'b1b2'
THRESHOLDS=(.2,0.)


def guard(phase,training=False):
    allowed=c.TRAIN if training else a.PHASES
    if phase not in allowed:raise PermissionError('phase outside Stage B role')


def cell_addresses(xyz,origin):
    u=(np.asarray(xyz)-np.asarray(origin))/6.766
    inside=np.all((u>=0)&(u<np.array([32,16,16])),axis=-1)
    index=np.floor(u[inside]).astype(np.int64)
    side=index[:,0]//16
    index[:,0]%=16
    flat=side*4096+np.ravel_multi_index(index.T,(16,16,16))
    return inside,flat


def observed(phase):
    p=c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json'
    q=c.verify_receipt(p,payload=True)
    if q['phase']!=phase or not q['exact_id_sky_rsd_join'] or q['target_columns_present']:
        raise ValueError('unverified observation identity')
    parent=p.parent/'PARENT_COMPLETE.json'
    if c.sha256(parent)!=q['parent_receipt_sha256']:raise ValueError('parent binding drift')
    data=fitsio.read(q['outputs'][0]['path'],columns=['TARGETID','RA','DEC','Z'])
    ids=data['TARGETID']
    if len(np.unique(ids))!=len(ids):raise ValueError('duplicate observation IDs')
    if not all(np.isfinite(data[k]).all() for k in ('RA','DEC','Z')):
        raise ValueError('nonfinite observation coordinates')
    selected=(data['Z']>=.15)&(data['Z']<.55)
    data=data[selected]
    xyz=coord.sky_mpc_h(data['RA'],data['DEC'],data['Z'])
    caps=galactic_cap(data['RA'],data['DEC'])
    with fitsio.FITS(json.loads(parent.read_text())['outputs'][0]['path']) as f:
        parent_columns=f[1].get_colnames()
    return data,xyz,caps,dict(receipt=c.file_record(p,content_hash=True),payload=q['outputs'][0],
        original_rows=q['observed_rows'],selected_redshift_rows=len(data),
        parent_receipt=c.file_record(parent,content_hash=True),parent_columns=parent_columns,
        observed_columns=list(data.dtype.names),coordinate_config=c.file_record(coord.CONFIG,content_hash=True))


def target(phase,pid,training):
    guard(phase,training)
    if not training:return b.auth.truth(phase,pid,b.FREEZE)
    p=coord.ROOT/'targets'/phase/(pid+'.json');r=coord.verify_receipt(p,payload=True)
    if r['phase']!=phase or r['pair_id']!=pid:raise ValueError('wrong train target')
    return read_target(r['outputs'][0]['path'],phase,pid),c.file_record(p,content_hash=True)


def prepare(pilot=False):
    c.require_compute();start=time.monotonic()
    config=json.loads(CONFIG.read_text())
    if config['development_phases']!=list(a.PHASES) or config['baseline_training_phases']!=list(c.TRAIN):
        raise ValueError('phase manifest changed')
    dest=ROOT/('prepare_pilot' if pilot else 'prepare');dest.mkdir(parents=True,exist_ok=False)
    binding=dict(config=c.file_record(CONFIG,content_hash=True),source=c.file_record(__file__,content_hash=True),
        freeze=c.file_record(b.FREEZE,content_hash=True),job=os.environ['SLURM_JOB_ID'],pilot=pilot,
        selection='All development pairs; training _00 geometry only, fixed before target reads')
    c.atomic_json(dest/'BINDING.json',binding)
    allrows=[];sources=[];class_counts={str(t):np.zeros(4,dtype=np.int64) for t in THRESHOLDS}
    phases=(*a.PHASES,) if pilot else (*c.TRAIN,*a.PHASES)
    for phase in phases:
        before=time.monotonic();training=phase in c.TRAIN;guard(phase,training)
        gp=coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
        g=coord.verify_receipt(gp,payload=False);geometry.verify_rows(g['pairs'],phase)
        rows=[v for v in g['pairs'] if not training or v['pair_id'].endswith('_00')]
        if len(rows)!=16:raise ValueError('incomplete matched geometry strata')
        if pilot:rows=rows[:1]
        data,xyz,caps,source=observed(phase)
        source['geometry']=c.file_record(gp,content_hash=True);sources.append(source)
        used=[];phase_rows=[]
        for row in rows:
            pid=row['pair_id'];origin=np.array(row['grid']['origin_mpc_h'])+3.383*(np.array(row['center'])-16)
            capmask=caps==(1 if row['cap']=='NGC' else 0)
            addresses=np.flatnonzero(capmask)
            inside,flat=cell_addresses(xyz[capmask],origin);addresses=addresses[inside]
            observations=conditions.load_pair(phase,pid)
            support=np.stack([observations['support'][tuple(slice(*s) for s in cut)]
                for cut in b.op.layout()['owned_core_crops_in_joint']]).reshape(-1)
            keep=support[flat]>0;zero=int((~keep).sum());flat=flat[keep];addresses=addresses[keep]
            values,truthsource=target(phase,pid,training)
            tensor=values['fullbox_tensor_cores'];eigen=b.tidal.eigensystem(tensor).reshape(8192,3)
            truth=np.stack([(eigen>t).sum(-1) for t in THRESHOLDS]).astype('u1')
            counts=np.bincount(flat,minlength=8192).reshape(2,4096)
            ids=data['TARGETID'][addresses];used.append(ids)
            if len(np.unique(ids))!=len(ids):raise ValueError('duplicate selected ID')
            own_centers=origin+(np.column_stack(np.unravel_index(flat%4096,(16,16,16)))+
                np.stack((flat//4096*16,np.zeros(len(flat)),np.zeros(len(flat))),axis=-1)+.5)*6.766
            if len(flat) and np.max(abs(xyz[addresses]-own_centers))>3.383+1e-8:
                raise ValueError('point-to-cell geometry mismatch')
            cp=coord.ROOT/'conditions'/phase/(pid+'.json')
            rec=dict(phase=phase,training=training,pair_id=pid,cap=row['cap'],shell=row['shell'],
                support_stratum=row['support_stratum'],core_galaxies=counts.sum(-1).tolist(),
                galaxies=len(ids),zero_support_excluded=zero,origin_mpc_h=origin.tolist(),
                target=truthsource,condition=c.file_record(cp,content_hash=True))
            if training:
                for j,t in enumerate(THRESHOLDS):
                    class_counts[str(t)]+=np.bincount(truth[j,flat],minlength=4)
                rec['class_counts']={str(t):np.bincount(truth[j,flat],minlength=4).tolist() for j,t in enumerate(THRESHOLDS)}
            else:
                path=dest/(pid+'.npz')
                np.savez_compressed(path,targetid=ids,z=data['Z'][addresses],xyz=xyz[addresses],
                    flat_index=flat,cell_counts=counts,truth_cell_labels=truth.reshape(2,2,4096),
                    support=support.reshape(2,4096),truth_eigenvalues=eigen.reshape(2,4096,3))
                rec['output']=c.file_record(path,content_hash=True)
            phase_rows.append(rec);allrows.append(rec)
            del observations,values,tensor,eigen
        all_ids=np.concatenate(used)
        if len(np.unique(all_ids))!=len(all_ids):raise ValueError('cross-core galaxy duplication')
        c.atomic_json(dest/(phase+'.json'),dict(phase=phase,cases=phase_rows,source=source,
            selected_galaxies=len(all_ids),seconds=time.monotonic()-before,passed=True))
        print(json.dumps(dict(stage='b_prepare',phase=phase,pairs=len(rows),galaxies=len(all_ids),seconds=time.monotonic()-before)),flush=True)
        del data,xyz,caps;gc.collect()
    baseline={str(t):dict(counts=class_counts[str(t)].tolist(),
        probabilities=(class_counts[str(t)]/class_counts[str(t)].sum()).tolist()) for t in THRESHOLDS} if not pilot else None
    out=dict(passed=True,panel_complete=not pilot,cases=allrows,sources=sources,baseline=baseline,
        binding=binding,seconds=time.monotonic()-start,posterior_qualified=False,
        property_audit='Observed payload has TARGETID/RA/DEC/Z only. Parent column inventory recorded; no sSFR or stellar mass assumed, no property contrast scored.')
    c.atomic_json(dest/'COMPLETE.json',out)
    print(json.dumps(dict(stage='b_prepare_complete',seconds=out['seconds'],out=str(dest))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args();prepare(args.pilot)
