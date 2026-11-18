"""Prepare physical whole-field arrays separately from fitted chart publication.

Raw staging never publishes COMPLETE.json. Finalization requires the current
full-20 normalizer. Historical raw arrays may be reused only with unchanged
roles, exact geometry, source arithmetic and fresh payload/parity checks.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_wholefield_pilot_data as data
from workflows.sbi import e2e_coupled_conditions as conditions
from workflows.sbi import e2e_coupled_condition_products as products
from workflows.sbi import e2e_coupled_observations as obs
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_views as views

OLD=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_pilot_20261013_v1')


def rows_for(phase):
    c.require_compute();c.phase_guard(phase);data.guard(phase)
    if not c.SHARED_POLICY or data.ROOT==OLD:raise ValueError('Explicit new whole-field campaign required')
    p=coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
    geo=coord.verify_receipt(p,payload=False)
    return data.select_rows(phase,geo['pairs']),p


def source_map():
    return {str(Path(m.__file__).resolve()):c.sha256(m.__file__) for m in (op,conditions,products,obs,views,data)}


def checked(path,phase,pair=None):
    value=json.loads(path.read_text())
    if value.get('passed') is not True or value.get('phase')!=phase or (pair and value.get('pair_id')!=pair):
        raise ValueError('Whole-field receipt identity/gate failed')
    item=value.get('output')
    if item:
        p=c.guarded(item['path'],phase)
        if p.stat().st_size!=item['bytes'] or c.sha256(p)!=item['sha256']:raise ValueError('Whole-field payload changed')
    return value


def validate_pair(phase,row,directory):
    pair=row['pair_id'];a=checked(directory/'OBSERVATIONS.json',phase,pair);b=checked(directory/'TARGETS.json',phase,pair)
    with np.load(a['output']['path'],allow_pickle=False) as f:observed={k:f[k] for k in f.files}
    with np.load(b['output']['path'],allow_pickle=False) as f:target={k:f[k] for k in f.files}
    if set(observed)!={'joint','wide'} or observed['joint'].shape!=(12,*data.SHAPE) or observed['wide'].shape!=(12,48,48,48):
        raise ValueError('Whole-field observation shape/schema mismatch')
    if set(target)!={'rho','coarse_wide','residual'} or target['rho'].shape!=data.SHAPE or target['coarse_wide'].shape!=(48,48,48) or target['residual'].shape!=data.SHAPE:
        raise ValueError('Whole-field target shape/schema mismatch')
    if any(not np.isfinite(x).all() for x in (*observed.values(),*target.values())):raise ValueError('Nonfinite whole-field payload')
    if min(target['rho'].min(),target['coarse_wide'].min())<=0:raise ValueError('Nonpositive physical density')
    central=conditions.crop_context(conditions.load_pair(phase,pair),phase)
    channel_error=float(np.max(abs(observed['joint'][(slice(None),*data.OLD_CROP)]-central['joint'])))
    if channel_error>2e-5 or not np.array_equal(observed['wide'],central['wide']):raise ValueError('Whole-field observation central parity failed')
    reference=coord.verify_receipt(coord.ROOT/'targets'/phase/f'{pair}.json',payload=False)
    with h5py.File(reference['outputs'][0]['path'],'r') as f:rho=f['rho_joint'][:];wide=f['coarse_rho_extended'][4:52,4:52,4:52]
    density_error=float(np.max(abs(target['rho'][data.OLD_CROP]-rho)/rho))
    wide_error=float(np.max(abs(target['coarse_wide']-wide)/wide))
    coarse,residual=op.encode(target['rho'].astype(float))
    mass_error=float(np.max(abs(coarse-target['coarse_wide'][data.COARSE_CROP])/coarse))
    residual_error=float(np.max(abs(residual-target['residual'])))
    if max(density_error,wide_error,mass_error,residual_error)>2e-6:raise ValueError('Whole-field target parity/chart failed')
    return dict(pair_id=pair,channel_maxabs=channel_error,density_relative=density_error,
                wide_relative=wide_error,mass_relative=mass_error,residual_maxabs=residual_error)


def reuse(phase):
    rows,geometry=rows_for(phase);prior=checked(OLD/'data'/phase/'COMPLETE.json',phase)
    if prior['training']!=(phase in c.TRAIN) or prior['pairs']!=rows or prior['shape']!=list(data.SHAPE):
        raise ValueError('Whole-field role/geometry changed; rebuild raw arrays')
    # The committed source exactly matches the historical producer. Compare the
    # numerical function ASTs, allowing only current routing/phase-list changes.
    rel='workflows/sbi/e2e_wholefield_pilot_data.py'
    historical=subprocess.check_output(['git','show','HEAD:'+rel],cwd=c.REPO)
    import hashlib
    if hashlib.sha256(historical).hexdigest()!=prior['sources'][str((c.REPO/rel).resolve())]:
        raise ValueError('Historical whole-field source not available at pinned revision')
    def functions(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    old_ast,new_ast=functions(historical),functions(Path(data.__file__).read_text())
    for name in ('build_fine_observations','observations','targets'):
        if old_ast[name]!=new_ast[name]:raise ValueError('Whole-field numerical function changed: '+name)
    for m in (op,conditions,products,obs):
        if prior['sources'][str(Path(m.__file__).resolve())]!=c.sha256(m.__file__):raise ValueError('Whole-field dependency changed')
    directory=data.ROOT/'data'/phase
    with c.single_writer(directory):
        marker=directory/'RAW_REUSE_COMPLETE.json'
        if marker.exists():return checked(marker,phase)
        checks=[];receipts=[]
        for row in rows:
            pair=row['pair_id'];old_dir=OLD/'data'/phase/pair;dest=directory/pair;dest.mkdir(exist_ok=True)
            for kind in ('OBSERVATIONS','TARGETS'):
                src=old_dir/(kind+'.json');value=checked(src,phase,pair);payload=Path(value['output']['path']);out=dest/(kind+'.npz')
                if not out.exists():shutil.copy2(payload,out)
                if c.sha256(out)!=value['output']['sha256']:raise ValueError('Private whole-field copy mismatch')
                value=copy.deepcopy(value);value['output']=c.file_record(out,content_hash=True)
                if kind=='OBSERVATIONS':
                    value['raw_receipt']=c.file_record(coord.ROOT/'observations'/phase/(row['cap']+'_COMPLETE.json'),content_hash=True)
                    value['old_condition']=c.file_record(coord.ROOT/'conditions'/phase/(pair+'.json'),content_hash=True)
                else:
                    value['native_receipt']=c.file_record(c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json',content_hash=True)
                    value['old_target']=c.file_record(coord.ROOT/'targets'/phase/(pair+'.json'),content_hash=True)
                value['reuse']=dict(source_receipt=c.file_record(src,content_hash=True),raw_arrays_not_normalized=True,
                                    source_sha256=c.sha256(__file__))
                marker_pair=dest/(kind+'.json')
                if marker_pair.exists():
                    existing=checked(marker_pair,phase,pair)
                    if existing['reuse']['source_receipt']['sha256']!=c.sha256(src):raise ValueError('Whole-field reuse conflict')
                else:c.atomic_json(marker_pair,value)
                receipts.append(c.file_record(marker_pair,content_hash=True))
            checks.append(validate_pair(phase,row,dest))
        result=dict(phase=phase,pairs=rows,passed=True,checks=checks,receipts=receipts,
            geometry=c.file_record(geometry,content_hash=True),historical_complete=c.file_record(OLD/'data'/phase/'COMPLETE.json',content_hash=True),
            source_sha256=c.sha256(__file__),normalizer_bound=False)
        c.atomic_json(marker,result)
    print(json.dumps(dict(phase=phase,wholefield_raw_reused=len(rows))),flush=True)
    return result


def fresh(phase,action,workers):
    rows,_=rows_for(phase);directory=data.ROOT/'data'/phase
    with c.single_writer(directory):
        if action=='observations':data.observations(phase,rows)
        elif action=='targets':data.targets(phase,rows,workers)
    print(json.dumps(dict(phase=phase,wholefield_raw_stage=action,domains=len(rows))),flush=True)


def finalize(phase):
    rows,geometry=rows_for(phase)
    normalizer_path=coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json'
    normalizer=coord.verify_receipt(normalizer_path,payload=False)
    if normalizer['fit_phases']!=list(c.TRAIN) or len(c.TRAIN)!=20:raise ValueError('Full-20 normalization required')
    digest=c.sha256(normalizer_path);views.load_chart(digest)
    directory=data.ROOT/'data'/phase
    with c.single_writer(directory):
        marker=directory/'COMPLETE.json'
        if marker.exists():
            result=checked(marker,phase)
            if result['normalizer']!=digest or result['pairs']!=rows or result['sources']!=source_map():raise ValueError('Finalized whole-field binding changed')
            return result
        checks=[validate_pair(phase,row,directory/row['pair_id']) for row in rows]
        result=dict(phase=phase,pairs=rows,shape=list(data.SHAPE),normalizer=digest,sources=source_map(),
            geometry=c.file_record(geometry,content_hash=True),passed=True,training=phase in c.TRAIN,
            checks=checks,finalizer_sha256=c.sha256(__file__),raw_arrays_not_normalized=True)
        c.atomic_json(marker,result)
    print(json.dumps(dict(phase=phase,wholefield_complete=len(rows))),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['reuse','observations','targets','finalize']);p.add_argument('--phases',nargs='+',required=True);p.add_argument('--workers',type=int,default=32);a=p.parse_args()
    for phase in a.phases:
        if a.action=='reuse':reuse(phase)
        elif a.action=='finalize':finalize(phase)
        else:fresh(phase,a.action,a.workers)
