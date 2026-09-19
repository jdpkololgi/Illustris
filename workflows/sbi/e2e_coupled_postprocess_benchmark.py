"""Bounded synthetic CPU component costs, not posterior-performance evidence.

Measures the common rectangular decode/tidal path and full-size scoring arrays.
No catalogue, native matter, training target or held-out payload is opened.
This is not a complete production-analysis implementation or cold-cache IO test.
"""
import gc
import json
import os
from pathlib import Path
import resource
import time

import h5py
import numpy as np
from scipy import fft

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_physical_gate as physical
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi import e2e_vdm_context_metrics as metrics
from workflows.sbi.e2e_coupled_stage_b import used_bytes


def specification():
    return dict(operator_draws=8, score_draws=[32,128,256], workers=4,
                max_seconds=600, artifact_reserve_bytes=256*1024**2,
                joint_shape=[64,48,48], wide_shape=[48,48,48],
                owned_shape=[2,16,16,16], synthetic_seed=20260919)


def binding():
    return dict(specification=specification(), layout_sha256=c.sha256(op.LAYOUT),
                source_hashes={Path(module.__file__).name:c.sha256(module.__file__)
                               for module in (op,physical,targets,metrics)},
                runner_sha256=c.sha256(__file__), synthetic_only=True,
                scientific_payloads_read=False, posterior_performance_evaluated=False)


def require_time(start,seconds):
    if time.monotonic()-start>=seconds:
        raise TimeoutError('bounded synthetic postprocessing benchmark expired')


def checked_score(draws,truth):
    result=metrics.calibration(draws,truth)
    if any(not np.isfinite(result[key]).all() for key in ('mean','std','bias','crps','rank')):
        raise FloatingPointError('nonfinite synthetic calibration calculation')
    return result


def score_probe(size,owned_shape,rng):
    """Time actual metric functions; discard synthetic score values."""
    density=rng.normal(size=(size,*owned_shape))
    eigen=np.sort(rng.normal(size=(size,*owned_shape,3)),axis=-1)
    vector=np.concatenate((eigen,np.diff(eigen,axis=-1)),axis=-1)
    density_truth=np.zeros(owned_shape); vector_truth=np.zeros((*owned_shape,5))
    start=time.monotonic()
    checked_score(density,density_truth)
    density_seconds=time.monotonic()-start
    start=time.monotonic()
    checked_score(vector,vector_truth)
    marginal_seconds=time.monotonic()-start
    start=time.monotonic()
    energy=metrics.fair_energy(vector,vector_truth)
    if not np.isfinite(energy).all(): raise FloatingPointError('nonfinite vector energy score')
    energy_seconds=time.monotonic()-start
    # Six summaries per owned core: density, three eigenvalues and two gaps.
    summaries=np.concatenate((density.mean(axis=(2,3,4))[...,None],
                              vector.mean(axis=(2,3,4))),axis=-1).reshape(size,12)
    start=time.monotonic()
    checked_score(summaries,np.zeros(12))
    joint=metrics.fair_energy(summaries,np.zeros(12))
    variogram=metrics.variogram_score(summaries,np.zeros(12))
    if not np.isfinite([joint,variogram]).all(): raise FloatingPointError('nonfinite joint scores')
    joint_seconds=time.monotonic()-start
    return dict(draws=size, owned_voxels=int(np.prod(owned_shape)), vector_components=5,
                density_calibration_seconds=density_seconds,
                eigen_gap_marginal_seconds=marginal_seconds,
                pointwise_vector_energy_seconds=energy_seconds,
                adjacent_summary_scores_seconds=joint_seconds,
                score_values_retained=False)


def write_field(path,arrays):
    start=time.monotonic()
    with h5py.File(path,'x') as saved:
        saved.attrs['synthetic_only']=True
        for key,value in arrays.items(): saved.create_dataset(key,data=np.asarray(value,dtype='f4'))
        saved.flush()
    with path.open('rb') as stream: os.fsync(stream.fileno())
    item=c.file_record(path,content_hash=True)
    with h5py.File(path,'r') as saved:
        for key,value in arrays.items():
            if not np.array_equal(saved[key][:],np.asarray(value,dtype='f4')):
                raise ValueError('synthetic field serialization roundtrip differs')
    return item,time.monotonic()-start


def run():
    c.require_compute(); c.config()
    for item in json.loads((c.REPO/'SOURCE.json').read_text())['files']:
        if c.sha256(c.REPO/item['relative'])!=item['sha256']:
            raise ValueError('postprocessing benchmark frozen source drift')
    cfg=specification(); bound=binding()
    if used_bytes(c.ROOT)+cfg['artifact_reserve_bytes']>c.config()['approval']['scratch_bytes']:
        raise RuntimeError('insufficient approved postprocessing artifact reserve')
    directory=coord.ROOT/'technical_cpu/postprocess'/c.digest(bound)[:16]
    with c.single_writer(directory):
        final=directory/'POSTPROCESS_BENCHMARK_COMPLETE.json'
        if final.exists():
            result=coord.verify_receipt(final)
            if result['binding']!=bound: raise ValueError('postprocessing benchmark binding drift')
            return result
        attempt=directory/f'attempt_{time.time_ns()}'; attempt.mkdir()
        start=time.monotonic(); rng=np.random.default_rng(cfg['synthetic_seed'])
        rows=[]; outputs=[]; crop=np.asarray(op.layout()['joint_coarse_crop_in_wide'])
        slices=tuple(slice(a,b) for a,b in crop)
        for index in range(cfg['operator_draws']):
            require_time(start,cfg['max_seconds'])
            wide=np.exp(.1*rng.normal(size=cfg['wide_shape']))
            residual=.2*op.project(rng.normal(size=cfg['joint_shape']))
            before=time.monotonic(); rho=op.decode(wide[slices],residual)
            decode_seconds=time.monotonic()-before
            relative=float(np.max(np.abs(op.mean_pool(rho)-wide[slices])/wide[slices]))
            if relative>2e-6: raise ValueError('synthetic coarse mass conservation failed')
            before=time.monotonic()
            tensor=op.consistent_tensor(rho-1,wide-1,crop,workers=cfg['workers'])
            tensor_seconds=time.monotonic()-before
            owned=targets.owned_core_arrays(tensor)
            if np.max(np.abs(owned[...,[0,3,5]].sum(-1)-targets.owned_core_arrays(rho-1)))>2e-6:
                raise ValueError('synthetic physical trace identity failed')
            before=time.monotonic(); eigen=physical.eigenvalues(owned)
            eigen_seconds=time.monotonic()-before
            before=time.monotonic()
            spectrum=fft.rfftn(rho-1,workers=cfg['workers'])
            np.abs(spectrum)**2
            np.quantile(rho,[.01,.1,.5,.9,.99])
            fft_quantile_seconds=time.monotonic()-before
            item,io_seconds=write_field(attempt/f'synthetic_field_{index:02d}.h5',
                dict(rho_joint=rho,rho_wide=wide,eigen_core=eigen))
            outputs.append(item)
            rows.append(dict(index=index,decode_seconds=decode_seconds,
                common_tensor_seconds=tensor_seconds,owned_eigen_seconds=eigen_seconds,
                fft_quantile_seconds=fft_quantile_seconds,write_fsync_hash_verify_seconds=io_seconds,
                coarse_mass_relative_error=relative,output_bytes=item['bytes']))
            print(json.dumps(dict(synthetic_operator_draw=index,seconds=rows[-1])),flush=True)
            del wide,residual,rho,tensor,owned,eigen,spectrum; gc.collect()
        scores=[]
        for size in cfg['score_draws']:
            require_time(start,cfg['max_seconds'])
            scores.append(score_probe(size,cfg['owned_shape'],rng))
            print(json.dumps(dict(synthetic_score_timing=scores[-1])),flush=True)
            gc.collect()
        require_time(start,cfg['max_seconds'])
        result=dict(**coord.provenance(),binding=bound,operator_measurements=rows,
            score_measurements=scores,elapsed_seconds=time.monotonic()-start,
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            numpy_version=np.__version__,outputs=outputs,synthetic_only=True,
            scientific_payloads_read=False,scientific_training_authorized=False,
            posterior_performance_evaluated=False,full_preparation_complete=False,
            limitations=['Synthetic inputs, not a trained-posterior evaluation',
                'Four-worker common rectangular operator; no multi-job scaling claim',
                'Uncompressed per-draw HDF5 including hash/roundtrip; no cold-cache claim',
                'FFT/quantiles are component probes, not the final band/window implementation',
                'GPU-to-host copy, orchestration and output contention are not measured'],**{'pass':True})
        c.atomic_json(final,result)
        return result


if __name__=='__main__':
    result=run()
    print(json.dumps(dict(elapsed_seconds=result['elapsed_seconds'],
                          peak_host_rss_bytes=result['peak_host_rss_bytes'],synthetic_only=True)),flush=True)
