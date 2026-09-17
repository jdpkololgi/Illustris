"""Training-only physical representation gate; no learned scores or promotion."""
import argparse
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
import torch

from workflows.sbi.e2e_vdm_context_data import output_root, spec, CONFIG, read_json
from workflows.sbi.e2e_vdm_context_dataset import Products, coarse_local_crop, fit_normalization
from workflows.sbi.e2e_vdm_context_models import encode_density, decode_density
from workflows.sbi.e2e_field_build_products import require_compute, sha256, tensor_from_delta, eigs
from workflows.sbi.e2e_durable import publish_json

OPERATOR = 'consistent-expanded-density-v2'
FAILED_V1_SHA256 = 'c72e7d9a6169299aa8beab877f156316bf6d6d56701aedefdc87fd2591e61581'


def repeat_spatial(x, factor=4):
    for axis in range(3):
        x = np.repeat(x,factor,axis=axis)
    return x


def consistent_tensor(delta, coarse_delta, coarse_crop, cell, factor=4):
    """The wide tide and local subtraction use exactly the same lifted density.

    This removes the lifting commutator, not unknown exterior/boundary error.
    A small geometry-independent helper permits matched-domain analytic tests.
    """
    background = repeat_spatial(coarse_delta[coarse_crop],factor)
    if background.shape != delta.shape:
        raise ValueError('fine parent and coarse crop do not align')
    crop = tuple(slice(s.start*factor,s.stop*factor) for s in coarse_crop)
    wide_tensor = tensor_from_delta(repeat_spatial(coarse_delta,factor),cell)
    return wide_tensor[crop]+tensor_from_delta(delta-background,cell)


def composite_tensor(delta, coarse_delta, offset_raw=(0,0,0), cell=6.766):
    """Approved v2 finite-domain closure; not full exterior truth."""
    delta,coarse_delta = np.asarray(delta,dtype=np.float64),np.asarray(coarse_delta,dtype=np.float64)
    if delta.shape != (48,)*3 or coarse_delta.shape != (48,)*3:
        raise ValueError('registered fine/coarse geometries required')
    sl = coarse_local_crop(offset_raw,(0,0,0))
    return consistent_tensor(delta,coarse_delta,sl,cell)


def tensor_controls():
    n,factor,cell=16,4,6.766
    axes=np.meshgrid(*[(np.arange(n)+.5)/n]*3,indexing='ij',sparse=True)
    records=[]
    for mode in ((0,0,0),(1,0,0),(1,1,0),(1,1,1)):
        delta=np.broadcast_to(.2*np.cos(2*np.pi*sum(k*x for k,x in zip(mode,axes))),(n,)*3).copy()
        coarse=delta.reshape(4,4,4,4,4,4).mean((1,3,5))
        actual=consistent_tensor(delta,coarse,(slice(0,4),)*3,cell,factor)
        truth=tensor_from_delta(delta,cell)
        error=float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))
        if error>1e-12:
            raise ValueError('matched-domain tensor recovery failed')
        records.append(dict(mode=mode,rms_tensor_error_over_reference=error))
    return records


def verify_representation_release(root):
    release=read_json(root/'data/REPRESENTATION_RELEASE.json')
    if release.get('operator')!=OPERATOR or not release.get('representation_pass'):
        raise PermissionError('approved operator release missing')
    required={'data/REPRESENTATION_GATE.json','data/REPRESENTATION_GATE_V2.json',
              'data/NORMALIZATION.json','PHYSICS_V2_SOURCE.json'}
    if set(release['receipts'])!=required:
        raise ValueError('incomplete representation provenance')
    for path,digest in release['receipts'].items():
        if sha256(root/path)!=digest:
            raise ValueError('representation provenance drift: '+path)
    if release['receipts']['data/REPRESENTATION_GATE.json']!=FAILED_V1_SHA256:
        raise ValueError('failed v1 result changed')
    gate=read_json(root/'data/REPRESENTATION_GATE_V2.json')
    source=read_json(root/'PHYSICS_V2_SOURCE.json')
    if (not gate['scientific_representation_pass'] or not gate['training_launch_allowed']
            or gate['operator']!=OPERATOR or gate['evaluator_sha256']!=sha256(__file__)
            or gate['config_sha256']!=sha256(CONFIG)
            or gate['geometry_sha256']!=sha256(root/'data/GEOMETRY.json')
            or gate['source_receipt_sha256']!=sha256(root/'PHYSICS_V2_SOURCE.json')
            or source['source_sha256']['workflows/sbi/e2e_vdm_context_physics.py']!=sha256(__file__)
            or gate['failed_v1_sha256']!=FAILED_V1_SHA256
            or len(gate['records'])!=32 or gate['heldout_used']
            or not np.all(np.asarray(gate['median_per_anchor_relative_rmse_reduction'])>=.25)):
        raise PermissionError('physical representation gate not passed/matched')
    return release


def plane_controls():
    c = spec()
    raw_cell = c['fine_cell_mpc_h']/2
    raw_position = (np.arange(384)+.5)*raw_cell-192*raw_cell
    coarse_position = (np.arange(48)+.5)*8*raw_cell-192*raw_cell
    records = []
    for k in (0.,.01,.02,.04,.08):
        sampled = (1.4+.2*np.cos(k*raw_position)).reshape(48,8).mean(1)
        gain = 1. if k==0 else np.sin(8*k*raw_cell/2)/(8*np.sin(k*raw_cell/2))
        expected = 1.4+.2*gain*np.cos(k*coarse_position)
        error = float(np.max(np.abs(sampled-expected)))
        if error > 2e-12 or gain <= 0:
            raise ValueError('declared block-average plane/DC transfer failed')
        records.append(dict(k=k,amplitude_transfer=float(gain),max_abs_error=error,
            convention='exact eight-point average of raw R7 samples, no extra Gaussian smoothing'))
    dc = tensor_from_delta(np.full((8,8,8),.4),27.064)
    if np.max(np.abs(dc[...,[0,3,5]]-.4/3))>1e-12:
        raise ValueError('DC tensor completion failed')
    return records


def topology(eigen, truth):
    threshold = .2  # inherited physical T-web convention; descriptive, not a new gate
    label = (eigen>threshold).sum(-1)
    true_label = (truth>threshold).sum(-1)
    void_labels,_ = ndimage.label(label==0,structure=ndimage.generate_binary_structure(3,1))
    sizes = np.bincount(void_labels.ravel())[1:]
    structure_labels,_ = ndimage.label(label>=2,structure=ndimage.generate_binary_structure(3,1))
    connected = []
    for axis in range(3):
        first,last = [slice(None)]*3,[slice(None)]*3
        first[axis],last[axis] = 0,-1
        connected.append(bool((set(np.unique(structure_labels[tuple(first)]))-{0})
                              & (set(np.unique(structure_labels[tuple(last)]))-{0})))
    return dict(class_disagreement=float((label!=true_label).mean()),
        class_fractions=[float((label==i).mean()) for i in range(4)],
        largest_void_fraction=float(sizes.max(initial=0)/label.size),
        connected_structure_axes=connected,threshold=threshold)


def evaluate(root):
    require_compute()
    root = output_root(root)
    destination = root/'data/REPRESENTATION_GATE_V2.json'
    if destination.exists():
        raise FileExistsError('representation gate is immutable')
    if sha256(root/'data/REPRESENTATION_GATE.json')!=FAILED_V1_SHA256:
        raise ValueError('preserved failed v1 gate required')
    source=read_json(root/'PHYSICS_V2_SOURCE.json')
    if Path(source['source']).resolve()!=Path(__file__).resolve().parents[2]:
        raise ValueError('execute only frozen v2 evaluator')
    for name,digest in source['source_sha256'].items():
        if sha256(Path(source['source'])/name)!=digest:
            raise ValueError('v2 source drift')
    dataset = Products(root,['ph000','ph002'],targets=True)
    anchors = sorted(k for k,r in dataset.rows.items() if r['small_train'])
    if len(anchors)!=32:
        raise ValueError('exact A32 training-only physical panel required')
    started = time.monotonic()
    controls = plane_controls()
    tensor_checks = tensor_controls()
    records = []
    core = (slice(16,32),)*3
    for anchor in anchors:
        target = dataset.raw_targets(anchor)
        rho,coarse,truth = target['rho'],target['coarse'],target['tensor']
        mean,u = encode_density(torch.from_numpy(rho)[None,None])
        coarse_local = coarse[(slice(18,30),)*3]
        reconstruction = decode_density(torch.from_numpy(coarse_local)[None,None],u).numpy()[0,0]
        mass_error = float(np.max(np.abs(mean.numpy()[0,0]-coarse_local)))
        roundtrip = float(np.max(np.abs(reconstruction-rho)))
        parent = tensor_from_delta(rho-1,6.766)
        composite = composite_tensor(rho-1,coarse-1)
        trace_error = float(np.max(np.abs(composite[...,[0,3,5]].sum(-1)-(rho-1))))
        if max(mass_error,roundtrip,trace_error)>2e-6:
            raise ValueError('mass/roundtrip/trace representation identity failed')
        reference = eigs(truth)
        scale = np.maximum(reference.std((0,1,2)),1e-12)
        metrics = {}
        for key,tensor in (('parent_only',parent),('shared_coarse_fine',composite)):
            eigen = eigs(tensor[core])
            rmse = np.sqrt(np.mean((eigen-reference)**2,axis=(0,1,2)))
            metrics[key] = dict(rmse_over_truth_std=(rmse/scale).tolist(),
                bias=(eigen-reference).mean((0,1,2)).tolist(),topology=topology(eigen,reference))
        baseline = np.asarray(metrics['parent_only']['rmse_over_truth_std'])
        candidate = np.asarray(metrics['shared_coarse_fine']['rmse_over_truth_std'])
        reduction = 1-candidate/np.maximum(baseline,1e-12)
        records.append(dict(anchor=anchor,phase=dataset.rows[anchor]['phase'],
            metrics=metrics,reduction=reduction.tolist(),mass_max_abs=mass_error,
            roundtrip_max_abs=roundtrip,trace_max_abs=trace_error))
    reductions = np.array([r['reduction'] for r in records])
    median = np.median(reductions,axis=0)
    passed = bool(np.all(median>=.25))
    result = dict(operator=OPERATOR,failed_v1_sha256=FAILED_V1_SHA256,
        source_receipt_sha256=sha256(root/'PHYSICS_V2_SOURCE.json'),
        tensor_controls=tensor_checks,
        config_sha256=sha256(CONFIG),geometry_sha256=sha256(root/'data/GEOMETRY.json'),
        product_receipts=dataset.receipts,evaluator_sha256=sha256(__file__),records=records,
        plane_controls=controls,median_per_anchor_relative_rmse_reduction=median.tolist(),
        reduction_quantiles=np.quantile(reductions,[0,.1,.5,.9,1],axis=0).tolist(),
        scientific_representation_pass=passed,training_launch_allowed=passed,
        exact_identity_pass=True,heldout_used=False,elapsed_seconds=time.monotonic()-started,
        gate='median of per-anchor relative normalized-eigenvalue-RMSE reductions >=25% for EACH eigenvalue',
        caveat='representation sufficiency screen, not learned posterior calibration or full exterior recovery')
    publish_json(destination,result)
    print('REPRESENTATION_GATE',passed,'median_reduction',median.tolist(),flush=True)
    if not passed:
        raise RuntimeError('predeclared physical representation gate failed; no fits authorized')
    fit_normalization(root)
    paths=['data/REPRESENTATION_GATE.json','data/REPRESENTATION_GATE_V2.json',
           'data/NORMALIZATION.json','PHYSICS_V2_SOURCE.json']
    publish_json(root/'data/REPRESENTATION_RELEASE.json',dict(operator=OPERATOR,
        representation_pass=True,receipts={p:sha256(root/p) for p in paths},
        gpu_launch_still_requires_full_manifest_and_smoke=True))
    verify_representation_release(root)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    evaluate(p.parse_args().root)


if __name__=='__main__':
    main()
