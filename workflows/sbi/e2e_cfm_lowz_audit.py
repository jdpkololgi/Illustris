"""Read frozen CFM exposure and stage target-free low-z development catalogues.

Never modifies old products, reads final-test phases, fits a model or silently
equates field-domain exposure with conditioned low-redshift evaluation.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket

import fitsio
import numpy as np
from scipy.interpolate import CubicSpline

REPO=Path(__file__).resolve().parents[2]
BASE=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
COUPLED=BASE/'coupled_20260918_v1'
WHOLE=BASE/'wholefield_pilot_20261013_v1'
DEPLOY=Path('/pscratch/sd/d/dkololgi/graphweb_desi/outputs/loa_shared_sixfit_20261026_v1/production_v1/NGC_seed17/BINDING.json')
BINS=np.array([0,.01,.03,.05,.075,.1,.15,.25,.35,.45,.55,.6])

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def radial_bounds(origin,shape,cell):
    lo=np.asarray(origin);hi=lo+np.asarray(shape)*cell
    near=np.where((lo<=0)&(hi>=0),0,np.minimum(abs(lo),abs(hi)))
    return float(np.linalg.norm(near)),float(np.linalg.norm(np.maximum(abs(lo),abs(hi))))

def north(ra,dec):
    a=np.deg2rad(ra);d=np.deg2rad(dec)
    return np.sin(d)*np.sin(np.deg2rad(27.12825))+np.cos(d)*np.cos(np.deg2rad(27.12825))*np.cos(a-np.deg2rad(192.85948))>0

def domain(row,shape,cell):
    midpoint=np.asarray(row['center'])+[16,0,0]
    centre=np.asarray(row['grid']['origin_mpc_h'])+midpoint*3.383
    origin=centre-np.asarray(shape)*cell/2
    return dict(origin=origin.tolist(),shape=list(shape),cell=cell,radius_bounds=radial_bounds(origin,shape,cell))

def main(args):
    assert os.environ.get('SLURM_JOB_ID') and socket.gethostname().startswith('nid')
    args.output.mkdir(parents=True,exist_ok=False)
    table=np.loadtxt(REPO/'configs/data/desi_distance_701f498.dat')
    radius=CubicSpline(table[:,0],table[:,2],extrapolate=False)
    redshift=CubicSpline(table[:,2],table[:,0],extrapolate=False)
    deployed=read(DEPLOY);config=read(REPO/'configs/e2e_coupled_data_v1.json')
    train=tuple(p for p,r in config['phase_roles'].items() if r=='train')
    allowed=set(train)|{'ph012','ph013'}
    assert not allowed&{'ph018','ph019'}
    report=dict(schema='cfm-lowz-audit-v1',job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
        source_sha256=sha(__file__),deployed_binding_sha256=sha(DEPLOY),checkpoints={},
        bin_edges=BINS.tolist(),catalogues={},phase_policy='Frozen checkpoint training phases; ph012/ph013 development only; ph018/ph019 sealed',
        source_cosmic_epoch=.2,old_science_centres=[.15,.55],old_observed_context=[.1,.6],
        lowz_science_candidate=[.01,.1],selection={},passed=False)
    import torch
    torch.set_num_threads(1)
    for key in ('coarse_17','unet_x_17','coarse_29','unet_x_29'):
        record=deployed['checkpoints'][key];path=Path(record['path'])
        assert sha(path)==record['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        binding=state['binding'];assert json.loads(json.dumps(binding))==read(path.parent/'BINDING.json')
        assert binding['normalizer']==deployed['normalizer']
        ids=binding.get('ids',binding.get('train_ids'));assert set(p for p,_ in ids)==set(train)
        phases={};rows=[]
        for phase,pair in ids:
            if phase not in phases:
                source=(WHOLE/'data'/phase/'COMPLETE.json') if key.startswith('unet') else (COUPLED/'cartesian_v2/geometry'/phase/'GEOMETRY_COMPLETE.json')
                data=read(source)
                if key.startswith('unet'):assert sha(source)==binding['data'][phase]
                phases[phase]={r['pair_id']:r for r in data['pairs']}
            row=phases[phase][pair]
            fine=domain(row,(96,64,64),6.766) if key.startswith('unet') else domain(row,(64,48,48),6.766)
            wide=domain(row,(48,48,48),27.064)
            rows.append(dict(phase=phase,pair=pair,cap=row['cap'],centre_z=row['redshift'],fine=fine,wide=wide))
        summary=dict(path=str(path),sha256=record['sha256'],step=state['step'],normalizer=binding['normalizer'],
            training_phases=list(train),domains=len(rows),centre_z=[min(r['centre_z'] for r in rows),max(r['centre_z'] for r in rows)],
            fine_domain_min_z=float(redshift(min(r['fine']['radius_bounds'][0] for r in rows))),
            wide_domain_min_z=float(redshift(min(r['wide']['radius_bounds'][0] for r in rows))),
            fine_boxes_intersect_z_lt_01=int(sum(r['fine']['radius_bounds'][0]<radius(.1) for r in rows)),
            wide_boxes_containing_observer=int(sum(r['wide']['radius_bounds'][0]==0 for r in rows)),
            target_box_exposure_is_not_conditioned_lowz_validation=True)
        report['checkpoints'][key]=summary
        if key in ('coarse_17','unet_x_17'):save(args.output/f'{key}_domains.json',rows)
        del state
        print(json.dumps(dict(event='checkpoint',model=key,**summary)),flush=True)
    npath=COUPLED/'cartesian_v2/normalization/NORMALIZATION_COMPLETE.json'
    assert sha(npath)==deployed['normalizer']
    norm=read(npath)
    save(args.output/'FROZEN_NORMALIZER.json',norm)
    selection=read(config['observation']['fixed_selection_manifest'])
    report['selection']=dict(path=config['observation']['fixed_selection_manifest'],sha256=sha(config['observation']['fixed_selection_manifest']),
        fit_phases=selection['fit_phases'],fit_redshift_source_code=[.15,.55],
        interpolation_domain={c:[min(v['grid_z']),max(v['grid_z'])] for c,v in selection['rotations']['0']['caps'].items()},
        volume_jacobian_redshift_domain=[min(selection['cosmology']['redshift_grid']),max(selection['cosmology']['redshift_grid'])],
        extrapolation_below_domain_prohibited=True)
    for phase in (*train,'ph012','ph013'):
        assert phase in allowed
        root=COUPLED/'observations'/phase
        rec=read(root/'OBSERVED_COMPLETE.json');src=Path(rec['outputs'][0]['path'])
        assert rec['pass'] and not rec['target_columns_present'] and sha(src)==rec['outputs'][0]['sha256']
        g=fitsio.read(src,columns=['TARGETID','RA','DEC','Z'])
        assert len(np.unique(g['TARGETID']))==len(g) and np.isfinite(g['Z']).all()
        cap=north(g['RA'],g['DEC'])
        counts={c:np.histogram(g['Z'][cap==val],BINS)[0].tolist() for c,val in [('NGC',True),('SGC',False)]}
        output=dict(source=str(src),source_sha256=rec['outputs'][0]['sha256'],rows=len(g),counts=counts,
                    observed_z_min=float(g['Z'].min()),observed_z_max=float(g['Z'].max()),unique_targetids=True)
        arpath=root/'angular/ANGULAR_COMPLETE.json';ar=read(arpath)
        output['angular_receipt']=dict(path=str(arpath),sha256=sha(arpath),outputs=ar['outputs'])
        if phase in ('ph012','ph013'):
            # The old target-free parent contains real-space redshift; never use
            # HALO_INDEX/BOX_INDEX for truth or validity in the new extraction.
            parentrec=read(root/'PARENT_COMPLETE.json');pp=Path(parentrec['outputs'][0]['path'])
            assert sha(pp)==parentrec['outputs'][0]['sha256']
            parent=fitsio.read(pp,columns=['TARGETID','RA','DEC','Z','Z_COSMO','R_MAG_APP'])
            keep=(g['Z']>=.01)&(g['Z']<.15);low=g[keep];ids=low['TARGETID']
            assert np.array_equal(parent['TARGETID'],np.arange(1,len(parent)+1))
            p=parent[ids-1]
            assert np.array_equal(p['TARGETID'],ids) and np.allclose(p['RA'],low['RA'],atol=1e-10,rtol=0)
            assert np.allclose(p['DEC'],low['DEC'],atol=1e-10,rtol=0) and np.allclose(p['Z'],low['Z'],atol=2e-7,rtol=0)
            unit=np.column_stack((np.cos(np.deg2rad(low['DEC']))*np.cos(np.deg2rad(low['RA'])),np.cos(np.deg2rad(low['DEC']))*np.sin(np.deg2rad(low['RA'])),np.sin(np.deg2rad(low['DEC']))))
            xyz=radius(low['Z'])[:,None]*unit;real=radius(p['Z_COSMO'])[:,None]*unit
            assert np.isfinite(xyz).all() and np.isfinite(real).all()
            path=args.output/f'{phase}_LOWZ_POINTS.npz'
            np.savez_compressed(path,TARGETID=ids,RA=low['RA'],DEC=low['DEC'],Z=low['Z'],Z_COSMO=p['Z_COSMO'],R_MAG_APP=p['R_MAG_APP'],CAP=cap[keep].astype('u1'),xyz_observed=xyz,xyz_realspace=real,native_xyz=(real-1000)%2000)
            output.update(staged_lowz=dict(path=str(path),rows=len(low),sha256=sha(path),parent_sha256=parentrec['outputs'][0]['sha256'],
                uses_own_realspace_position=True,halo_indices_used=False,contains_tidal_targets=False))
            del parent
        report['catalogues'][phase]=output
        print(json.dumps(dict(event='catalogue',phase=phase,counts=counts)),flush=True)
    report['passed']=True
    report['model_ready']=False
    report['remaining']=['low-z train-only selection/response with a validated volume definition',
        'origin-crossing geometry and support masks', 'native-density/periodic-tidal low-z targets',
        'frozen-normalizer condition shards and separate low-z scoring', 'GPU posterior evaluation not authorized by CPU preparation']
    save(args.output/'AUDIT.json',report)
    print('AUDIT_AND_POINT_STAGING_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args())
