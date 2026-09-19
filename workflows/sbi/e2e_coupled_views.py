"""Common, train-normalized data views for the later four-arm comparison.

Observation inference never opens a target path. Training-target access checks
the phase role before any file IO. This module packages data, not authorization
to fit a scientific model; the later experiment needs its own approved manifest.
"""
import json
from pathlib import Path
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_conditions as conditions
from workflows.sbi import e2e_coupled_condition_products as products
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi.e2e_coupled_normalization import IDENTITY


def validate_chart(chart):
    channels={'joint':products.LOCAL_CHANNELS,'wide':products.WIDE_CHANNELS}
    for key,n in (('joint',12),('wide',12),('coarse_logrho',1),('fine_residual',1)):
        mean=np.asarray(chart[key]['mean']); std=np.asarray(chart[key]['std'])
        if (mean.shape!=(n,) or std.shape!=(n,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or np.any(std<=0)):
            raise ValueError('invalid global normalization: '+key)
        if key in channels and chart[key]['channels']!=list(channels[key]):
            raise ValueError('normalizer channel ordering mismatch')
        if key in channels:
            for index,name in enumerate(channels[key]):
                if name in IDENTITY and (mean[index]!=0. or std[index]!=1.):
                    raise ValueError('physical support/LOS identity scale changed')
    if chart['fine_residual']['mean']!=[0.]: raise ValueError('fine chart must preserve its zero-mean subspace')
    return chart


def load_chart(expected_sha256):
    """Read the pinned global transform only; never follow its target sources.

Full source-chain verification belongs to preparation readiness, not inference:
an observation-only installation need not have any target payload mounted.
"""
    path=coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json'
    if c.sha256(path)!=expected_sha256: raise ValueError('global normalizer differs from frozen model binding')
    record=coord.verify_receipt(path,payload=False)
    if record['fit_phases']!=list(c.TRAIN) or record['phase_weight']!=1/13:
        raise PermissionError('normalizer must use exactly the thirteen equally weighted training phases')
    return validate_chart(record['normalization'])


def crop_slices(phase,offset):
    c.phase_guard(phase); cfg=op.layout(); offset=list(offset)
    if offset not in cfg['context_offsets_raw']: raise ValueError('unregistered context offset')
    if c.ROLES[phase]!='train' and offset!=[0,0,0]: raise PermissionError('no held-out context augmentation')
    start=np.array(cfg['wide_crop_base_start'])+np.asarray(offset)//8
    wide=tuple(slice(int(v),int(v)+48) for v in start)
    coarse=np.asarray(cfg['joint_coarse_crop_in_wide'])-np.asarray(offset)[:,None]//8
    return wide,tuple(slice(int(a),int(b)) for a,b in coarse)


def scale_channels(values,stats):
    x=np.asarray(values,dtype=np.float32)
    mean=np.asarray(stats['mean'],dtype=np.float32)[:,None,None,None]
    std=np.asarray(stats['std'],dtype=np.float32)[:,None,None,None]
    return (x-mean)/std


def observation_view(arrays,phase,offset,chart):
    validate_chart(chart)
    cropped=conditions.crop_context(arrays,phase,offset)
    # No phase IDs, file names, targets or HOD parameters enter model channels.
    return dict(joint=scale_channels(cropped['joint'],chart['joint']),
                wide=scale_channels(cropped['wide'],chart['wide']),
                support=cropped['support'].copy(),
                joint_center_from_wide_mpc_h=-np.asarray(offset,dtype=np.float32)*3.383)


def load_observations(phase,pair_id,normalization_sha256,offset=(0,0,0)):
    c.phase_guard(phase)
    chart=load_chart(normalization_sha256)
    return observation_view(conditions.load_pair(phase,pair_id),phase,offset,chart)


def target_view(rho,wide_extended,phase,offset,chart):
    c.phase_guard(phase); validate_chart(chart)
    wide_slice,coarse_slice=crop_slices(phase,offset)
    targets.density_qa(rho,wide_extended)
    coarse,residual=op.encode(rho)
    wide=np.asarray(wide_extended[wide_slice],dtype=np.float64)
    if not np.allclose(wide[coarse_slice],coarse,rtol=2e-6,atol=0):
        raise ValueError('wide/fine coarse alignment mismatch')
    cs,rs=chart['coarse_logrho'],chart['fine_residual']
    return dict(coarse_logrho=((np.log(wide)-cs['mean'][0])/cs['std'][0])[None].astype('f4'),
                fine_residual=(residual/rs['std'][0])[None].astype('f4'))


def decode_view(coarse_standardized,residual_standardized,phase,offset,chart):
    validate_chart(chart); _,crop=crop_slices(phase,offset)
    coarse=np.asarray(coarse_standardized,dtype=np.float64)
    residual=np.asarray(residual_standardized,dtype=np.float64)
    if coarse.shape!=(1,48,48,48) or residual.shape!=(1,64,48,48):
        raise ValueError('coarse/joint latent shape mismatch')
    cs,rs=chart['coarse_logrho'],chart['fine_residual']
    with np.errstate(over='raise',invalid='raise'):
        wide=np.exp(coarse[0]*cs['std'][0]+cs['mean'][0])
    return op.decode(wide[crop],residual[0]*rs['std'][0])


def independent_parents(joint):
    """Deterministic crops; each later I arm uses independent residual noise.

The COMMON full joint observation view must still be supplied to I and J.
Cropping the I arm's observation information would confound the coupling test.
"""
    x=np.asarray(joint)
    if x.shape[-3:]!=(64,48,48): raise ValueError('joint trailing dimensions required')
    return tuple(x[(Ellipsis,*tuple(slice(*sl) for sl in crop))]
                 for crop in op.layout()['independent_parent_crops_in_joint'])


def assemble_independent(left,right):
    """One block-aligned ownership seam, not overlap averaging or extra coupling.

Both generated parents must be conditioned on the SAME coarse draw. Left owns
joint x<32, right owns x>=32 (right local x>=16). The seam is block4-aligned and
separates the owned science cores. This conserves coarse masses and keeps the
same 64x48x48 physical operator available for both I and J primary diagnostics.
"""
    left,right=np.asarray(left),np.asarray(right)
    if left.shape!=right.shape or left.shape[-3:]!=(48,48,48):
        raise ValueError('matched independent parent trailing dimensions required')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('nonfinite independent parents')
    return np.concatenate((left[...,0:32,:,:],right[...,16:48,:,:]),axis=-3)


def load_training_pair(phase,pair_id,normalization_sha256,offset=(0,0,0)):
    # Role check deliberately precedes chart, observation and target IO.
    if phase not in c.TRAIN: raise PermissionError('only registered training phases may form fit batches')
    chart=load_chart(normalization_sha256)
    observation=observation_view(conditions.load_pair(phase,pair_id),phase,offset,chart)
    directory=coord.ROOT/'targets'/phase
    receipt=coord.verify_receipt(directory/f'{pair_id}.json',payload=False)
    if receipt['phase']!=phase or receipt['pair_id']!=pair_id or len(receipt['outputs'])!=1:
        raise ValueError('target receipt identity mismatch')
    item=receipt['outputs'][0]; path=c.guarded(item['path'],phase)
    if path.parent!=directory.resolve() or not path.name.startswith(pair_id+'_generation_'):
        raise PermissionError('target payload outside pair target directory')
    if path.stat().st_size!=item['bytes'] or c.sha256(path)!=item['sha256']:
        raise ValueError('target payload checksum mismatch')
    # Reuse the independently tested strict HDF5 link/schema guard. Import only
    # on the target-enabled path; inference never invokes or follows this code.
    from workflows.sbi.e2e_coupled_product_audit import read_target
    values=read_target(path,phase,pair_id)
    return observation,target_view(values['rho_joint'],values['coarse_rho_extended'],phase,offset,chart)
