"""Restartable positive R7 density charts and independent full-box tidal truth.

Stages commit individually. Interruption requires recomputing the in-memory FFT,
not rebuilding completed density/component products. No model fitting or scoring.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy import fft

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi import e2e_coupled_observations as obs
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_field_error_budget as native_ops

SCHEMA='e2e-coupled-target-v1'


def starts_for(row,kind):
    cfg=op.layout(); center=np.asarray(row['center'])
    if kind=='joint':
        start=center+cfg['joint_raw_start_from_first_center']; shape=cfg['joint_density_shape']; average=2
    elif kind=='wide':
        start=center+cfg['joint_midpoint_from_first_center']+cfg['wide_raw_start_from_joint_midpoint']
        shape=cfg['wide_extended_shape']; average=8
    else:
        raise ValueError('unknown target domain')
    return [s+average*np.arange(n) for s,n in zip(start,shape)],average


def owned_core_arrays(joint):
    return np.stack([joint[tuple(slice(*s) for s in crop)]
                     for crop in op.layout()['owned_core_crops_in_joint']])


def density_qa(rho,wide):
    cfg=op.layout()
    if rho.shape!=tuple(cfg['joint_density_shape']) or wide.shape!=tuple(cfg['wide_extended_shape']):
        raise ValueError('density target shape mismatch')
    if not np.isfinite(wide).all() or np.any(wide<=0):
        raise ValueError('wide density must be strictly positive without clipping')
    coarse,residual=op.encode(rho)
    rebuilt=op.decode(coarse,residual)
    roundtrip=float(np.max(np.abs(rebuilt-rho)/rho))
    errors=[]
    for offset in cfg['context_offsets_raw']:
        shifted=np.array(cfg['wide_crop_base_start'])+np.array(offset)//8
        cut=wide[tuple(slice(s,s+48) for s in shifted)]
        crop=np.array(cfg['joint_coarse_crop_in_wide'])-np.array(offset)[:,None]//8
        match=cut[tuple(slice(*s) for s in crop)]
        if match.shape!=coarse.shape:
            raise ValueError('coarse/fine grid alignment failed')
        errors.append(float(np.max(np.abs(match-coarse)/coarse)))
    maximum=max(errors)
    if roundtrip>2e-6 or maximum>2e-6:
        raise ValueError('density chart/mass agreement failed')
    return dict(roundtrip_max_relative=roundtrip,coarse_fine_max_relative=maximum,
                minimum_rho=float(np.min(rho)),minimum_wide_rho=float(np.min(wide)),
                all_seven_context_crops_verified=True)


def fullbox_qa(rho,tensor):
    if tensor.shape!=(2,16,16,16,6) or not np.isfinite(tensor).all():
        raise ValueError('invalid full-box tensor reference')
    trace=tensor[...,[0,3,5]].sum(-1)
    error=float(np.max(np.abs(trace-(owned_core_arrays(rho)-1.))))
    if error>2e-6:
        raise ValueError('independent full-box tensor trace does not match density')
    return dict(fullbox_trace_max_abs=error,physical_representation_gate_pass=None)


def verified_stage(path,binding):
    if not path.exists(): return None
    record=coord.verify_receipt(path)
    if record['binding']!=binding:
        raise ValueError('target stage source/code drift')
    return record


def stage_record(phase,binding,output,**extra):
    return dict(**coord.provenance(),phase=phase,role=c.ROLES[phase],binding=binding,
                outputs=[c.file_record(output,content_hash=True)],**extra,**{'pass':True})


def make_density_stage(field,rows,phase,binding,directory,mean):
    output=directory/f'density_generation_{time.time_ns()}.h5'
    checks=[]
    with h5py.File(output,'x') as saved:
        saved.attrs.update(schema=SCHEMA,phase=phase,contains_observations=False,
                           coordinate_sha256=c.sha256(coord.CONFIG),distance_unit='Mpc/h',
                           smoothing_mpc_h=7.,native_count_mean=mean)
        for cap in ('NGC','SGC'):
            selected=[r for r in rows if r['cap']==cap]
            mids=np.array([r['center'] for r in selected])+[16,0,0]
            if np.any(mids%8): raise ValueError('wide sample lattice misaligned')
            lo,hi=(mids//8).min(0)-28,(mids//8).max(0)+28
            starts=[np.arange(a,b)*8 for a,b in zip(lo,hi)]
            wide_cap=1+op.sample_averaged_local(field,starts,8,selected[0]['grid'])
            if np.any(wide_cap<=0) or not np.isfinite(wide_cap).all():
                raise ValueError('interpolated R7 density is nonpositive/nonfinite')
            for row in selected:
                starts,average=starts_for(row,'joint')
                rho=1+op.sample_averaged_local(field,starts,average,row['grid'])
                begin=(np.array(row['center'])+[16,0,0])//8-28-lo
                wide=wide_cap[tuple(slice(v,v+56) for v in begin)]
                qa=density_qa(rho,wide)
                group=saved.create_group(row['pair_id'])
                group.create_dataset('rho_joint',data=rho,compression='lzf',shuffle=True)
                group.create_dataset('coarse_rho_extended',data=wide,compression='lzf',shuffle=True)
                checks.append(dict(pair_id=row['pair_id'],**qa))
            print(json.dumps(dict(phase=phase,stage='density',cap=cap,pairs=len(selected))),flush=True)
        saved.flush()
    obs.durable(output)
    record=stage_record(phase,binding,output,checks=checks,native_count_mean=mean)
    c.atomic_json(directory/'DENSITY_STAGE.json',record)
    return record


def make_tensor_stage(field,rows,phase,binding,directory,index):
    values=np.empty((len(rows),2,16,16,16),dtype=np.float64)
    for ordinal,row in enumerate(rows):
        for side,offset in enumerate((np.array([0,0,0]),np.array([32,0,0]))):
            center=np.asarray(row['center'])+offset
            starts=[v-16+2*np.arange(16) for v in center]
            values[ordinal,side]=op.sample_averaged_local(field,starts,2,row['grid'])
    if not np.isfinite(values).all(): raise ValueError('nonfinite independent reference')
    output=directory/f'tensor_{index}_generation_{time.time_ns()}.npy'
    with output.open('xb') as stream: np.save(stream,values,allow_pickle=False)
    obs.durable(output)
    record=stage_record(phase,binding,output,component=list(op.COMPONENTS[index]),
                        pair_ids=[row['pair_id'] for row in rows])
    c.atomic_json(directory/f'TENSOR_{index}_STAGE.json',record)
    return record


def build(phase,workers=32,stop_after_seconds=12000):
    c.require_compute(); c.phase_guard(phase); coord.bind(); coord.require_host_checks()
    started=time.monotonic()
    directory=coord.ROOT/'targets'/phase
    geometry_path=coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
    geo=coord.verify_receipt(geometry_path,payload=False); rows=geo['pairs']
    geometry.verify_rows(rows,phase)
    native_path=c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json'
    native=c.verify_receipt(native_path)
    if (native['phase']!=phase or native['ngrid']!=2048 or native['box_mpc_h']!=2000.
            or native['subsample_fraction']!=.10 or native['processed_files']!=136):
        raise ValueError('unqualified/incomplete A+B native density')
    binding=dict(geometry_receipt_sha256=c.sha256(geometry_path),
                 native_receipt_sha256=c.sha256(native_path),builder_sha256=c.sha256(__file__),
                 operators_sha256=c.sha256(op.__file__),native_operator_sha256=c.sha256(native_ops.__file__),
                 layout_sha256=c.sha256(op.LAYOUT))
    with c.single_writer(directory):
        final=directory/'TARGETS_COMPLETE.json'
        if final.exists():
            result=verified_stage(final,binding)
            for item in result['pair_receipts']:
                if c.sha256(item['path'])!=item['sha256']: raise ValueError('target receipt drift')
                verified_stage(Path(item['path']),binding)
            return result
        density=verified_stage(directory/'DENSITY_STAGE.json',binding)
        tensors=[verified_stage(directory/f'TENSOR_{i}_STAGE.json',binding) for i in range(6)]
        if density is None or any(t is None for t in tensors):
            print(json.dumps(dict(phase=phase,stage='fullbox_R7_FFT')),flush=True)
            spectrum,mean=op.smoothed_spectrum(native['outputs'][0]['path'],workers)
            if density is None:
                field=fft.irfftn(spectrum.copy(),s=(2048,)*3,workers=workers,overwrite_x=True)
                density=make_density_stage(field,rows,phase,binding,directory,mean)
                del field; gc.collect()
            for index,component in enumerate(op.COMPONENTS):
                if tensors[index] is not None: continue
                if time.monotonic()-started>stop_after_seconds:
                    return dict(phase=phase,paused=True,completed_components=sum(t is not None for t in tensors))
                field=native_ops.inverse_component(spectrum,2000.,component,workers=workers)
                tensors[index]=make_tensor_stage(field,rows,phase,binding,directory,index)
                del field; gc.collect()
                print(json.dumps(dict(phase=phase,stage='fullbox_tensor',component=component)),flush=True)
            del spectrum; gc.collect()
        pair_ids=[r['pair_id'] for r in rows]
        if any(t['pair_ids']!=pair_ids for t in tensors): raise ValueError('tensor pair ordering drift')
        components=[np.load(t['outputs'][0]['path'],mmap_mode='r',allow_pickle=False) for t in tensors]
        records=[]; qa=[]
        with h5py.File(density['outputs'][0]['path'],'r') as src:
            for ordinal,row in enumerate(rows):
                pair_id=row['pair_id']; marker=directory/f'{pair_id}.json'
                if marker.exists():
                    record=verified_stage(marker,binding)
                else:
                    rho=src[pair_id]['rho_joint'][:]; wide=src[pair_id]['coarse_rho_extended'][:]
                    tensor=np.stack([v[ordinal] for v in components],axis=-1)
                    checks=dict(**density_qa(rho,wide),**fullbox_qa(rho,tensor))
                    output=directory/f'{pair_id}_generation_{time.time_ns()}.h5'
                    with h5py.File(output,'x') as dst:
                        dst.attrs.update(schema=SCHEMA,phase=phase,pair_id=pair_id,role=c.ROLES[phase],
                            contains_observations=False,coordinate_sha256=c.sha256(coord.CONFIG),
                            distance_unit='Mpc/h',smoothing_mpc_h=7.,smoothing_count=1)
                        for key,value in (('rho_joint',rho),('coarse_rho_extended',wide),
                                          ('fullbox_tensor_cores',tensor)):
                            dst.create_dataset(key,data=value,compression='lzf',shuffle=True)
                        dst.flush()
                    obs.durable(output)
                    record=stage_record(phase,binding,output,pair_id=pair_id,checks=checks,
                                        science_scores_evaluated=False)
                    c.atomic_json(marker,record)
                records.append(dict(path=str(marker),sha256=c.sha256(marker))); qa.append(record['checks'])
        stages=[directory/'DENSITY_STAGE.json']+[directory/f'TENSOR_{i}_STAGE.json' for i in range(6)]
        result=dict(**coord.provenance(),phase=phase,role=c.ROLES[phase],binding=binding,
            pair_receipts=records,stage_receipts=[c.file_record(p,content_hash=True) for p in stages],
            outputs=[],numerical_checks=qa,science_scores_evaluated=False,
            physical_representation_gate_pass=None,elapsed_seconds=time.monotonic()-started,**{'pass':True})
        c.atomic_json(final,result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',required=True)
    p.add_argument('--workers',type=int,default=32); p.add_argument('--stop-after-seconds',type=float,default=12000)
    a=p.parse_args(); result=build(a.phase,a.workers,a.stop_after_seconds)
    print(json.dumps(result),flush=True); raise SystemExit(75 if result.get('paused') else 0)
