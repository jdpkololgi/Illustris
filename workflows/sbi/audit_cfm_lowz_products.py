"""Independent low-z source lineage and matched-CIC response checks."""
import json
from pathlib import Path

import h5py
import numpy as np

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import prepare_cfm_lowz_campaign as campaign
from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi.prepare_cfm_lowz_local_condition_canary import project_crop, density_at


def adopted(record, phase):
    cfg,_ = campaign.authority()
    item = record['lowz_adoption']['source_receipt']
    path = c.guarded(item['path'],phase)
    if c.sha256(path) != item['sha256']:
        raise ValueError('Adopted source receipt changed')
    original = campaign.checked_source(path,phase,cfg['lowz']['source_config_sha256'])
    if original.get('outputs',[]) != record.get('outputs',[]):
        raise ValueError('Adoption changed source outputs')
    return original


def base_sources(audit,parent,observed,angular):
    from workflows.sbi import audit_shared_phase_sources as shared
    cfg,old = campaign.authority()
    originals=[adopted(r,audit.phase) for r in (parent,observed,angular)]
    return dict(helper_sha256=c.sha256(shared.__file__),lowz_helper_sha256=c.sha256(__file__),
        parent=shared.parent_sources(audit,originals[0],originals[1]),
        angular=shared.angular_sources(audit,originals[2],source_root=old['root'],
            source_config_sha256=cfg['lowz']['source_config_sha256'],reuse_source_root=old['reuse_source_root']))


def response(audit,record):
    cfg,_ = campaign.authority();phase=audit.phase;cap=record['cap']
    if (record['schema']!='cfm-lowz-adopted-matched-response-v1'
            or record['source_code_sha256']!=c.sha256(campaign.__file__)
            or record['selection']['fit_phases']!=list(c.TRAIN)):
        raise ValueError('Invalid low-z response authority')
    source=audit.file(record['matched_source']);raw=json.loads(source.read_text())
    if raw['phase']!=phase or raw['cap']!=cap or not raw['pass'] or raw['contains_matter']:
        raise ValueError('Wrong matched cap identity')
    output=audit.file(raw['output']);external=audit.file(raw['external_source'])
    if str(output)!=str(Path(record['outputs'][0]['path']).resolve()):
        raise ValueError('Response adapter points to different payload')
    selection_path=audit.file(record['selection']);selection=json.loads(selection_path.read_text())
    if selection['fit_phases']!=list(c.TRAIN) or selection['omitted_phase'] is not None:
        raise PermissionError('Response used nontraining selection')
    if raw['binding']['selection_sha256']!=c.sha256(selection_path):
        raise ValueError('Matched selection binding changed')
    angular=c.verify_receipt(c.ROOT/'observations'/phase/'angular/ANGULAR_COMPLETE.json')
    angular_path=audit.file(angular['outputs'][0])
    with np.load(angular_path,allow_pickle=False) as saved:
        amap={k:saved[k] for k in ('support','domain','angular_response')}
    plan,_,_=lowz.load_plan()
    grid=record['grid'];origin=np.asarray(grid['origin_mpc_h']);cell=grid['cell_mpc_h']
    # Independent scatter-add cubature, not the builder's separable contraction.
    rng=np.random.default_rng(8311+int(phase[2:])+(cap=='NGC'))
    sample=list(rng.integers(0,np.asarray(grid['shape']),size=(24,3)))
    edges=np.asarray(plan['diagnostic_bin_edges'],float)
    radii=coord.radius_mpc_h(edges);supported_samples={}
    with h5py.File(output,'r') as handle:
        for key in handle:
            link=handle.get(key,getlink=True)
            if isinstance(link,h5py.ExternalLink) and Path(link.filename).resolve()!=external:
                raise PermissionError('Unbound external geometry payload')
        expected_sum=0.;count_sum=0.
        for start in range(0,grid['shape'][0],8):
            a=handle['expected_counts_random'][start:start+8]
            b=handle['counts'][start:start+8]
            if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(),b.min())<0:
                raise ValueError('Invalid matched response values')
            expected_sum+=float(a.sum(dtype=float));count_sum+=float(b.sum(dtype=float))
            if len(supported_samples)<len(edges)-1:
                valid=handle['selection_valid'][start:start+len(a)].astype(bool)
                axes=[origin[0]+(start+np.arange(len(a))+.5)*cell,
                      origin[1]+(np.arange(a.shape[1])+.5)*cell,
                      origin[2]+(np.arange(a.shape[2])+.5)*cell]
                radius=np.sqrt(axes[0][:,None,None]**2+axes[1][None,:,None]**2+axes[2][None,None,:]**2)
                for bin_id,(lo,hi) in enumerate(zip(radii[:-1],radii[1:])):
                    if bin_id in supported_samples:continue
                    choices=np.flatnonzero(valid&(radius>=lo)&(radius<hi))
                    if len(choices):
                        local=np.array(np.unravel_index(choices[rng.integers(len(choices))],a.shape))
                        supported_samples[bin_id]=local+np.array([start,0,0])
        if set(supported_samples)!=set(range(len(edges)-1)):
            raise ValueError('Missing supported response audit sample in a declared redshift bin')
        sample.extend(supported_samples.values())
        if not np.isclose(count_sum,raw['count_source_audit']['supported_source_rows'],rtol=2e-6):
            raise ValueError('Full supported-count conservation failed')
        if not np.isclose(expected_sum,raw['sums']['expected'],rtol=2e-6):
            raise ValueError('Full expected-count sum differs')
        checks=[]
        for index in sample:
            ijk=tuple(index)
            value=project_crop(origin+index*cell,(1,1,1),cell,amap,cap,plan,selection,4).item()
            stored=float(handle['expected_counts_random'][ijk])
            if not np.isclose(stored,value,rtol=2e-6,atol=1e-7):
                raise ValueError('Independent midpoint CIC integral failed')
            z=coord.redshift(np.linalg.norm(origin+(index+.5)*cell))
            nbar,known=density_at(z,selection,cap)
            valid=bool(known and handle['support_random'][ijk] and lowz.radial_mask(z,plan))
            count=float(handle['counts'][ijk])
            ratio=np.log((count+.001)/(stored+.001)) if valid else 0.
            if (bool(handle['selection_valid'][ijk])!=valid
                    or not np.isclose(handle['ntilde_h3_mpc3'][ijk],nbar,rtol=2e-6)
                    or not np.isclose(handle['log_count_ratio_random'][ijk],ratio,rtol=2e-5,atol=2e-6)):
                raise ValueError('Independent response channel identity failed')
            checks.append(dict(index=index.tolist(),expected=stored,independent=value))
    return dict(source=str(source),expected_sum=expected_sum,count_sum=count_sum,
                independent_cic_samples=checks,external_geometry_bound=True,
                supported_samples_by_redshift_bin={str(k):v.tolist() for k,v in supported_samples.items()},
                diagnostic_bin_edges=edges.tolist())
