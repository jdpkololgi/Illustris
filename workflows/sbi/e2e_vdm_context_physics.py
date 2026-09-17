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


def repeat_spatial(x, factor=4):
    for axis in range(3):
        x = np.repeat(x,factor,axis=axis)
    return x


def composite_tensor(delta, coarse_delta, offset_raw=(0,0,0), cell=6.766):
    """No double counting; a declared finite-domain closure, not exterior truth."""
    delta,coarse_delta = np.asarray(delta,dtype=np.float64),np.asarray(coarse_delta,dtype=np.float64)
    if delta.shape != (48,)*3 or coarse_delta.shape != (48,)*3:
        raise ValueError('registered fine/coarse geometries required')
    sl = coarse_local_crop(offset_raw,(0,0,0))
    background = repeat_spatial(coarse_delta[sl])
    wide_tensor = tensor_from_delta(coarse_delta,4*cell)
    return repeat_spatial(wide_tensor[sl])+tensor_from_delta(delta-background,cell)


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
    destination = root/'data/REPRESENTATION_GATE.json'
    if destination.exists():
        raise FileExistsError('representation gate is immutable')
    dataset = Products(root,['ph000','ph002'],targets=True)
    anchors = sorted(k for k,r in dataset.rows.items() if r['small_train'])
    if len(anchors)!=32:
        raise ValueError('exact A32 training-only physical panel required')
    started = time.monotonic()
    controls = plane_controls()
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
    result = dict(config_sha256=sha256(CONFIG),geometry_sha256=sha256(root/'data/GEOMETRY.json'),
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
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    evaluate(p.parse_args().root)


if __name__=='__main__':
    main()
