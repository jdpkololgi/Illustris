"""Loa observation-only field atlas; independent tiles are NOT a global draw."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import fitsio
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.abacus_tweb import p3a_build_canonical_fields as grid_ops
from workflows.sbi.e2e_cfm_field_gallery import LOA

PROPERTY_ROOT=Path('/pscratch/sd/d/dkololgi/graphweb_desi/outputs/p12a_loa_cigale_20260929_v1')
STRIDE=np.array([64,32,32],dtype=int) # two owned 16^3 fine cores, exactly once


def tile_keys(xyz,grid):
    return np.floor((xyz-np.asarray(grid['origin_mpc_h']))/grid['cell_mpc_h']/STRIDE).astype(np.int32)


def prepare(output):
    c.require_compute();output.mkdir(parents=True,exist_ok=False)
    source=json.loads((LOA/'INPUTS_READY.json').read_text())
    for name,sha in source['arrays_sha256'].items():
        if c.sha256(LOA/name)!=sha: raise ValueError('observation cache changed: '+name)
    product=json.loads((PROPERTY_ROOT/'PRODUCT.json').read_text())
    if c.sha256(product['path'])!=product['sha256']: raise ValueError('enriched VAC changed')
    columns=['TARGETID','RA','DEC','Z','CAP','SUPPORTED','P_WEB','CIGALE_MATCHED',
             'CIGALE_AMBIGUOUS','CIGALE_SPECTYPE','MASS_CG_15','SFR_CG_15','MASS_CG_5','SFR_CG_5']
    d=fitsio.read(product['path'],columns=columns)
    d=d.astype(d.dtype.newbyteorder('='))
    with np.load(LOA/'catalogue.npz') as f:
        ra,dec,z,cap,ids=(f[k] for k in ('ra','dec','z','cap','targetid'))
    order=np.argsort(ids);at=np.searchsorted(ids[order],d['TARGETID'])
    if np.any(at==len(ids)) or not np.array_equal(ids[order[at]],d['TARGETID']):
        raise ValueError('VAC is not a subset of the observation catalogue')
    idx=order[at]
    for a,b in ((ra[idx],d['RA']),(dec[idx],d['DEC']),(z[idx],d['Z'])):
        if not np.allclose(a,b,atol=1e-10,rtol=0): raise ValueError('VAC/observation coordinate mismatch')
    xyz=coord.sky_mpc_h(ra,dec,z);vxyz=xyz[idx]
    gal=np.char.strip(d['CIGALE_SPECTYPE'].astype('U'))=='GALAXY'
    ok=d['CIGALE_MATCHED']&d['SUPPORTED']&gal
    for name in ('MASS_CG_15','SFR_CG_15'):ok &= np.isfinite(d[name])&(d[name]>0)
    np.savez_compressed(output/'catalogue.npz',xyz=xyz,cap=cap,targetid=ids)
    np.savez_compressed(output/'vac_properties.npz',xyz=vxyz,**{n:d[n] for n in columns})
    tiles=[];grids={};counts={}
    for capid,name in ((1,'NGC'),(0,'SGC')):
        grid=coord.grid_record(grid_ops.grid_from_xyz(xyz[cap==capid],3.383,coord.config()['padding_mpc_h']))
        grids[name]=grid;rows=np.flatnonzero(d['CAP']==capid)
        keys=tile_keys(vxyz[rows],grid);unique,inverse,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        for j,(key,n) in enumerate(zip(unique,count)):
            rr=rows[inverse==j];midpoint=key*STRIDE+STRIDE//2
            tiles.append(dict(id=name+'_'+'_'.join(f'{v:03d}' for v in key),cap=name,cap_id=capid,
                key=key.tolist(),midpoint=midpoint.tolist(),rows=int(n),supported=int(d['SUPPORTED'][rr].sum()),
                usable_cg15=int(ok[rr].sum()),lowz_rows=int(np.sum((d['Z'][rr]>=.2)&(d['Z'][rr]<.3)))))
        counts[name]=dict(tiles=len(unique),vac_rows=len(rows),usable_cg15=int(ok[rows].sum()),
                         raw_grid_voxels=int(np.prod(grid['shape'])))
    record=dict(schema='loa-field-atlas-plan-v1',grids=grids,tiles=tiles,counts=counts,
        property_product=product,observation_receipt=c.file_record(LOA/'INPUTS_READY.json',content_hash=True),
        source=c.file_record(__file__,content_hash=True),science_redshift=[.15,.55],
        ownership_raw_stride=STRIDE.tolist(),independent_tiles=True,global_joint_posterior=False,
        catalogue_sha256=c.sha256(output/'catalogue.npz'),properties_sha256=c.sha256(output/'vac_properties.npz'))
    c.atomic_json(output/'PLAN.json',record)
    print(json.dumps(dict(prepared=True,counts=counts,usable_cg15=int(ok.sum()))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','smoke','details','run','merge'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int,default=0)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--draws',type=int,default=8)
    p.add_argument('--seconds',type=float,default=14000);a=p.parse_args()
    if a.action=='prepare':prepare(a.root)
    elif a.action=='merge':merge(a.root)
    else:run(a)


def groups(properties,plan):
    result={}
    for cap,capid in [('NGC',1),('SGC',0)]:
        rows=np.flatnonzero(properties['CAP']==capid)
        keys=tile_keys(properties['xyz'][rows],plan['grids'][cap])
        code=keys[:,0]*10000+keys[:,1]*100+keys[:,2];order=np.argsort(code)
        sorted_rows=rows[order];kk=keys[order]
        breaks=np.r_[0,np.flatnonzero(np.diff(code[order]))+1,len(rows)]
        for lo,hi in zip(breaks[:-1],breaks[1:]):
            key=kk[lo];name=cap+'_'+'_'.join(f'{v:03d}' for v in key)
            result[name]=sorted_rows[lo:hi]
    return result


def binding(root,draws,hashes,normalizer):
    from workflows.sbi import e2e_loa_field_conditions as cond
    from workflows.sbi import e2e_loa_field_sampling as sampling
    from workflows.sbi import e2e_loa_field_tensor as tensor
    from workflows.sbi import e2e_coupled_operators as op
    return dict(draws=draws,nfe=128,batch=4,precision='TF32 float32 network; float64 FFT',
        checkpoints=hashes,normalizer=normalizer,plan=c.sha256(root/'PLAN.json'),
        source={name:c.sha256(module.__file__) for name,module in
                [('conditions',cond),('sampling',sampling),('tensor',tensor),('operators',op)]},
        runner=c.sha256(__file__),independent_tiles=True,global_joint_posterior=False)


def run_case(root,case,rows,props,plan,models,chart,run_binding,detail=False,precision_check=False):
    import hashlib
    import h5py
    from scipy.ndimage import map_coordinates
    from workflows.sbi import e2e_loa_field_conditions as cond
    from workflows.sbi import e2e_loa_field_sampling as sampling
    from workflows.sbi import e2e_loa_field_tensor as tensor
    from workflows.sbi import e2e_coupled_observations as obs
    from workflows.sbi import e2e_coupled_operators as op
    from workflows.sbi.e2e_coupled_physical_gate import eigenvalues
    from workflows.sbi.e2e_cfm_field_gallery import render,overlay
    directory=root/('details' if detail else 'tiles')/case['id']
    directory.mkdir(parents=True,exist_ok=True);marker=directory/'COMPLETE.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['binding']!=c.digest(run_binding):raise ValueError('tile resume binding mismatch')
        for item in old['outputs']:
            if c.sha256(item['path'])!=item['sha256']:raise ValueError('tile output changed')
        return old
    tick=time.monotonic();cap=case['cap'];grid=plan['grids'][cap];mid=np.asarray(case['midpoint'])
    legacy=detail and cap=='NGC'
    if legacy:
        old=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_gallery_20260930_v2')
        with np.load(old/'loa_conditions.npz') as f:arrays={k:f[k] for k in f.files}
    else:
        with h5py.File(root/(cap+'_observation_cache.h5'),'r') as f:
            arrays=cond.build(grid,mid,cap,f['counts_raw'],f['wide'],cond.load_response(cap))
    conditioning_seconds=time.monotonic()-tick
    digest=hashlib.sha256()
    for name in ('joint','wide'):digest.update(arrays[name].tobytes())
    origin=np.asarray(grid['origin_mpc_h'])+(mid-[64,48,48])*3.383
    xyz=props['xyz'][rows];position=((xyz-origin)/6.766-.5).T
    if np.any(position<15.49) or np.any(position>np.array([47.51,31.51,31.51])[:,None]):
        raise ValueError('ownership or interpolation coordinates outside owned cores')
    gs=map_coordinates(arrays['support'],position,order=1,mode='nearest',prefilter=False)
    core_support=[float(arrays['support'][x:x+16,16:32,16:32].mean()) for x in (16,32)]
    rho=[];wide=[];labels=[];labels0=[];gdensity=[];errors=[];prec=None
    sampling.setup(not legacy)
    count=run_binding['draws'];batch=2 if legacy else 4
    for first in range(0,count,batch):
        val=sampling.generate(models,arrays,chart,case['id'],first,min(batch,count-first),128,legacy)
        if precision_check and first==0:
            sampling.setup(False)
            ref=sampling.generate(models,arrays,chart,case['id'],0,2,128,False)
            sampling.setup(True);prec={}
            for name in ('rho','wide'):
                scale=float(np.sqrt(np.mean((ref[name][0]-ref[name][1])**2)/2))
                err=float(np.sqrt(np.mean((val[name][:2]-ref[name])**2)))
                prec[name+'_rmse_over_spread']=err/max(scale,1e-12)
            ee=[];rr=[]
            for i in range(2):
                ee.append((eigenvalues(tensor.consistent(val['rho'][i],val['wide'][i])[16:48,16:32,16:32])>.2).sum(-1))
                rr.append((eigenvalues(tensor.consistent(ref['rho'][i],ref['wide'][i])[16:48,16:32,16:32])>.2).sum(-1))
            prec['class_disagreement']=float(np.mean(np.asarray(ee)!=np.asarray(rr)))
            prec['passed']=max(prec.values())<.01
            if not prec['passed']:
                c.atomic_json(directory/'PRECISION_FAILED.json',prec);raise ValueError('expanded precision check failed')
        for r,w in zip(val['rho'],val['wide']):
            field=tensor.consistent(r,w);e=tensor.at_galaxies(field,xyz,grid,mid)
            density=map_coordinates(r,position,order=1,mode='nearest',prefilter=False)
            if np.max(abs(e.sum(-1)-(density-1)),initial=0)>1e-8*max(1.,np.max(abs(density),initial=0)):
                raise ValueError('galaxy-interpolated tensor trace failed')
            labels.append((e>.2).sum(-1).astype('u1'));labels0.append((e>0).sum(-1).astype('u1'))
            gdensity.append(density.astype('f4'))
            rho.append(r.astype('f4') if detail else r[16:48,16:32,16:32].astype('f4'))
            if detail:wide.append(w.astype('f4'))
        errors.extend(val['mass_error'].tolist())
    values=dict(rho=np.asarray(rho),rows=rows,classes02=np.asarray(labels),classes0=np.asarray(labels0),
        galaxy_density=np.asarray(gdensity),galaxy_support=gs.astype('f4'),
        support=arrays['support'] if detail else arrays['support'][16:48,16:32,16:32])
    if detail:values['wide']=np.asarray(wide)
    output=obs.publish_npz(directory,'fields',**values)
    record=dict(case=case,binding=c.digest(run_binding),condition_sha256=digest.hexdigest(),
        rows=len(rows),draws=count,core_support=core_support,max_mass_error=max(errors),
        conditioning_seconds=conditioning_seconds,seconds=time.monotonic()-tick,
        numerical_precision_check=prec,outputs=[c.file_record(output,content_hash=True)],
        field_property_eligible_rows=int(np.sum(gs>=.5)) if min(core_support)>=.25 else 0,
        independent_tiles=True,global_joint_posterior=False,legacy_density_replay=False)
    if legacy:
        previous=json.loads((old/'COMPLETE.json').read_text());oldrho=[]
        for name,sha in sorted(previous['loa']['chunks'].items()):
            if c.sha256(old/name)!=sha:raise ValueError('legacy density changed')
            with np.load(old/name) as f:oldrho.extend(f['rho'].astype('f4'))
        record['legacy_density_replay']=bool(np.array_equal(np.asarray(oldrho),values['rho']))
        if not record['legacy_density_replay']:raise ValueError('original NGC sample replay mismatch')
    if detail:
        with np.load(root/'catalogue.npz') as f:observed=f['xyz'][f['cap']==case['cap_id']]
        points=overlay(observed,grid,mid)
        render(directory/'gallery.png',cap+' Loa: exploratory local posterior',rho,arrays['support'],points)
        local=observed-origin;keep=np.all((local>=0)&(local<np.array([64,48,48])*6.766),axis=1)
        gp=obs.publish_npz(directory,'galaxies',xyz=observed[keep].astype('f4'))
        record['outputs'].append(c.file_record(gp,content_hash=True))
    c.atomic_json(marker,record)
    print(json.dumps(dict(tile=case['id'],rows=len(rows),draws=count,seconds=record['seconds'],
                          core_support=core_support,precision_check=prec)),flush=True)
    return record


def run(a):
    from workflows.sbi import e2e_loa_field_sampling as sampling
    from workflows.sbi import e2e_cfm_pilot_evaluate as evaluate
    from workflows.sbi import e2e_coupled_views as views
    c.require_compute();sampling.setup(True)
    plan=json.loads((a.root/'PLAN.json').read_text())
    for name in ('CONDITION_CHECKS.json','TENSOR_CHECKS.json','benchmark/BENCHMARK.json'):
        if not json.loads((a.root/name).read_text())['passed']:raise ValueError('upstream technical gate failed')
    for cap in ('NGC','SGC'):
        receipt=json.loads((a.root/(cap+'_CACHE.json')).read_text())
        if c.sha256(receipt['outputs'][0]['path'])!=receipt['outputs'][0]['sha256']:
            raise ValueError('cap condition cache changed')
    if c.sha256(a.root/'vac_properties.npz')!=plan['properties_sha256']:raise ValueError('property cache changed')
    with np.load(a.root/'vac_properties.npz') as f:props={k:f[k] for k in f.files}
    grouped=groups(props,plan)
    models,hashes,normalizer=evaluate.load_models(17,13312,26624);chart=views.load_chart(normalizer)
    count=16 if a.action=='details' else a.draws;bind=binding(a.root,count,hashes,normalizer)
    cfg=a.root/(a.action+f'_worker{a.worker:02d}_BINDING.json')
    if cfg.exists():
        if json.loads(cfg.read_text())!=bind:raise ValueError('worker binding drift')
    else:c.atomic_json(cfg,bind)
    if a.action=='details':
        cases=[]
        for cap,ra,dec,capid in [('NGC',180,10,1),('SGC',0,-10,0)]:
            grid=plan['grids'][cap]
            mid=(np.rint((coord.sky_mpc_h(np.array(ra),np.array(dec),np.array(.22))-
                           grid['origin_mpc_h'])/3.383/8)*8).astype(int)
            cases.append(dict(id='detail_'+cap,cap=cap,cap_id=capid,midpoint=mid.tolist(),center_radec_z=[ra,dec,.22]))
    elif a.action=='smoke':
        cases=[]
        for cap,ra,dec,z in [('NGC',180,10,.22),('SGC',0,-10,.22),('SGC',0,-10,.5)]:
            pos=coord.sky_mpc_h(np.array(ra),np.array(dec),np.array(z));grid=plan['grids'][cap]
            choices=[r for r in plan['tiles'] if r['cap']==cap]
            selected=min(choices,key=lambda r:np.linalg.norm(np.asarray(grid['origin_mpc_h'])+np.asarray(r['midpoint'])*3.383-pos))
            if selected not in cases:cases.append(selected)
    else:
        if not (0<=a.worker<a.workers):raise ValueError('invalid worker partition')
        if not json.loads((a.root/'SMOKE_COMPLETE.json').read_text())['passed']:raise ValueError('smoke not passed')
        cases=plan['tiles'][a.worker::a.workers]
    completed=[];start=time.monotonic()
    for case in cases:
        if time.monotonic()-start>a.seconds:raise TimeoutError('bounded atlas segment elapsed; valid receipts retained')
        if a.action=='details':
            grid=plan['grids'][case['cap']];raw=(props['xyz']-grid['origin_mpc_h'])/3.383
            mid=np.asarray(case['midpoint']);rows=np.flatnonzero((props['CAP']==case['cap_id'])&
                np.all((raw>=mid-STRIDE//2)&(raw<mid+STRIDE//2),axis=1))
        else:rows=grouped[case['id']]
        completed.append(run_case(a.root,case,rows,props,plan,models,chart,bind,
                                  a.action=='details',a.action=='smoke'))
    summary=dict(passed=True,cases=len(completed),draws=count,worker=a.worker,workers=a.workers,
                 seconds=time.monotonic()-start,binding=bind,results=completed)
    name='SMOKE_COMPLETE.json' if a.action=='smoke' else a.action+f'_worker{a.worker:02d}_COMPLETE.json'
    c.atomic_json(a.root/name,summary);print(json.dumps(dict(completed=a.action,cases=len(completed))),flush=True)


def merge(root):
    """Exact once-only VAC row coverage; retain unavailable field probabilities."""
    c.require_compute();plan=json.loads((root/'PLAN.json').read_text());n=plan['property_product']['rows']
    seen=np.zeros(n,dtype='u1');prob=np.full((n,4),np.nan,dtype='f4');prob0=prob.copy()
    density=np.full(n,np.nan,dtype='f4');spread=density.copy();support=density.copy()
    eligible=np.zeros(n,dtype=bool);tile_index=np.full(n,-1,dtype='i4');classes=None;receipts=[];bindings=set()
    for i,case in enumerate(plan['tiles']):
        path=root/'tiles'/case['id']/'COMPLETE.json';receipt=json.loads(path.read_text());bindings.add(receipt['binding'])
        item=receipt['outputs'][0]
        if c.sha256(item['path'])!=item['sha256']:raise ValueError('tile checksum changed')
        with np.load(item['path']) as f:
            rows=f['rows'];lab=f['classes02'];lab0=f['classes0'];d=f['galaxy_density'];gs=f['galaxy_support']
            if classes is None:classes=np.full((len(lab),n),255,dtype='u1')
            if lab.shape!=(len(classes),len(rows)) or np.any(seen[rows]) or np.any(lab>3):
                raise ValueError('tile rows or class samples invalid/duplicated')
            seen[rows]=1;tile_index[rows]=i;classes[:,rows]=lab
            prob[rows]=np.stack([(lab==j).mean(0) for j in range(4)],axis=1)
            prob0[rows]=np.stack([(lab0==j).mean(0) for j in range(4)],axis=1)
            density[rows]=d.mean(0);spread[rows]=d.std(0,ddof=1);support[rows]=gs
            eligible[rows]=(gs>=.5)&(min(receipt['core_support'])>=.25)
        receipts.append(dict(path=str(path),sha256=c.sha256(path)))
    if not np.all(seen==1) or len(bindings)!=1 or not np.allclose(prob.sum(1),1):
        raise ValueError('incomplete or mixed atlas; no merged product allowed')
    output=root/'FIELD_MARGINALS.npz'
    if output.exists():raise FileExistsError('merged output already exists')
    np.savez_compressed(output,p_web=prob,p_web_threshold0=prob0,rho_mean=density,rho_sd=spread,
        support=support,field_eligible=eligible,tile_index=tile_index,classes02=classes)
    record=dict(passed=True,rows=n,tiles=len(receipts),draws=len(classes),eligible_rows=int(eligible.sum()),
        independent_tiles=True,global_joint_posterior=False,receipts=receipts,
        output=c.file_record(output,content_hash=True),source=c.file_record(__file__,content_hash=True))
    c.atomic_json(root/'ATLAS_COMPLETE.json',record);print(json.dumps({k:v for k,v in record.items() if k!='receipts'}),flush=True)


if __name__=='__main__':main()
