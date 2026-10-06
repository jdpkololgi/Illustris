"""Development-only true-density closure of the frozen cap tidal operator.

Fill the ENTIRE inference source cuboid with true R7 density, including holes
and padding. Beyond that cuboid the operator still assumes zero contrast.
This tests finite-domain AND discretisation error; no posterior is modified.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1')
SOURCE = ROOT/'source_78156ac3417555ef269f'
OUT = ROOT/'verification/true_density_closure_20261005_v1'
sys.path.insert(0, str(SOURCE))
import h5py
import numpy as np
from scipy import fft
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_exterior_final_score as auth
from workflows.sbi import e2e_footprint_tidal_operator as tidal
from workflows.sbi.e2e_exterior_metrics import joint_rank

FREEZE = ROOT/'SELECTION_FREEZE.json'


def cuts(row, origin):
    raw = np.asarray(row['center'])-16
    pos = np.asarray(row['grid']['origin_mpc_h'])+3.383*raw
    offset = (pos-origin)/6.766
    start = np.rint(offset).astype(int)
    if np.max(abs(offset-start)) > 1e-8:
        raise ValueError('Misaligned core')
    return tuple(slice(int(a), int(a+b)) for a, b in zip(start, (32, 16, 16)))


def prepare(phase):
    rows = auth.geometry(phase, FREEZE)
    native_path = c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json'
    native = c.verify_receipt(native_path)
    if native['phase'] != phase or native['ngrid'] != 2048:
        raise ValueError('Wrong native source')
    directory = OUT/phase
    directory.mkdir(parents=True, exist_ok=True)
    spectrum, mean = op.smoothed_spectrum(native['outputs'][0]['path'], 32)
    field = fft.irfftn(spectrum, s=(2048,)*3, workers=32, overwrite_x=True)
    del spectrum
    gc.collect()
    records = []
    for cap in ('NGC', 'SGC'):
        receipt = ROOT/'fullcap_tides'/phase/cap/'d000/COMPLETE.json'
        rec = json.loads(receipt.read_text())
        with h5py.File(rec['output']['path']) as f:
            shape = f['support'].shape
            source_shape = f['rho_completed'].shape[1:]
            origin = np.asarray(f.attrs['origin_mpc_h'])
            source_origin = np.asarray(f.attrs['completed_origin_mpc_h'])
            pad = int(f.attrs['padding_cells'])
            if pad != 32 or float(f.attrs['cell_mpc_h']) != 6.766:
                raise ValueError('Unregistered operator geometry')
            assert tuple(n+2*pad for n in shape) == source_shape
            assert np.max(abs(source_origin-(origin-pad*6.766))) < 1e-8
        selected = [r for r in rows if r['cap'] == cap]
        grid = selected[0]['grid']
        raw_offset = (source_origin-np.asarray(grid['origin_mpc_h']))/3.383
        raw_start = np.rint(raw_offset).astype(int)
        assert np.max(abs(raw_start-raw_offset)) < 1e-8
        starts = [a+2*np.arange(n) for a, n in zip(raw_start, source_shape)]
        rho = 1+op.sample_averaged_local(field, starts, 2, grid)
        if not np.isfinite(rho).all() or rho.min() <= 0:
            raise ValueError('Invalid oracle density')
        qa = []
        for row in selected:
            truth, _ = auth.truth(phase, row['pair_id'], FREEZE)
            local = rho[cuts(row, source_origin)].reshape(2,16,16,16)
            expected = np.stack([truth['rho_joint'][tuple(slice(*s) for s in crop)]
                                 for crop in op.layout()['owned_core_crops_in_joint']])
            error = float(np.max(abs(local-expected)))
            if error > 2e-6:
                raise ValueError(('Truth grid mismatch', row['pair_id'], error))
            qa.append(dict(pair_id=row['pair_id'], density_maxabs=error))
        path = directory/(cap+'_TRUE_SOURCE.h5')
        with h5py.File(path, 'x') as f:
            f.create_dataset('rho', data=rho, chunks=(8,32,32), compression='lzf')
            f.attrs.update(phase=phase, cap=cap, origin_mpc_h=origin,
                           source_origin_mpc_h=source_origin, padding=pad,
                           target_shape=shape, smoothing_mpc_h=7., smoothing_count=1)
        records.append(dict(cap=cap, source=c.file_record(path, content_hash=True),
                            geometry_receipt=c.file_record(receipt, content_hash=True), checks=qa))
        print(json.dumps(dict(stage='prepared', phase=phase, cap=cap, checks=qa)), flush=True)
        del rho
    c.atomic_json(directory/'PREPARED.json', dict(phase=phase, records=records,
        native_receipt=c.file_record(native_path, content_hash=True), native_count_mean=mean,
        freeze_sha256=c.sha256(FREEZE), script_sha256=c.sha256(__file__),
        job=os.environ['SLURM_JOB_ID'], passed=True))


def comparison(draws, full, matched):
    mu = draws.mean(0)
    sd = draws.std(0, ddof=1)
    gap = matched-full
    residual = mu-full
    cov = np.cov(draws.T)
    cov = .8*cov+.2*np.diag(np.diag(cov))+np.eye(6)*1e-8*max(np.trace(cov)/6,1e-12)
    norm = lambda x: float(np.sqrt(x@np.linalg.solve(cov,x)))
    rank_full, rank_matched = joint_rank(draws,full), joint_rank(draws,matched)
    ordered = np.sort(draws,axis=0)
    return dict(fullbox_target=full.tolist(), operator_target=matched.tolist(),
        posterior_mean=mu.tolist(), posterior_sd=sd.tolist(), operator_gap=gap.tolist(),
        operator_gap_sd=(gap/sd).tolist(), original_bias_sd=(residual/sd).tolist(),
        matched_bias_sd=((mu-matched)/sd).tolist(),
        operator_mahalanobis=norm(gap), original_mahalanobis=norm(residual),
        matched_mahalanobis=norm(mu-matched),
        gap_residual_alignment=float(gap@np.linalg.solve(cov,residual)/max(norm(gap)*norm(residual),1e-30)),
        fullbox_joint=rank_full, matched_joint=rank_matched,
        fullbox_marginal90=((full>=ordered[1])&(full<=ordered[30])).tolist(),
        matched_marginal90=((matched>=ordered[1])&(matched<=ordered[30])).tolist())


def evaluate(phase):
    import torch
    from workflows.sbi.e2e_exterior_cap_tides import CapTidal
    from workflows.sbi.e2e_loa_field_sampling import setup
    setup(True)
    directory = OUT/phase
    prepared = json.loads((directory/'PREPARED.json').read_text())
    assert prepared['freeze_sha256'] == c.sha256(FREEZE)
    rows = auth.geometry(phase, FREEZE)
    cases = []
    for record in prepared['records']:
        cap = record['cap']
        assert c.sha256(record['source']['path']) == record['source']['sha256']
        with h5py.File(record['source']['path']) as f:
            rho = f['rho'][:]
            shape = tuple(map(int, f.attrs['target_shape']))
            origin = np.asarray(f.attrs['origin_mpc_h'])
            pad = int(f.attrs['padding'])
        selected = [r for r in rows if r['cap'] == cap]
        tensors = {r['pair_id']: np.empty((2,16,16,16,6)) for r in selected}
        operator = CapTidal(shape,pad)
        for k, component in enumerate(operator.components(rho)):
            for row in selected:
                tensors[row['pair_id']][...,k] = component[cuts(row,origin)].reshape(2,16,16,16)
        del component, operator
        gc.collect(); torch.cuda.empty_cache()
        for row in selected:
            pair = row['pair_id']
            truth, provenance = auth.truth(phase,pair,FREEZE)
            full = tidal.eigensystem(truth['fullbox_tensor_cores']).mean((1,2,3)).reshape(6)
            matched = tidal.eigensystem(tensors[pair]).mean((1,2,3)).reshape(6)
            core = cuts(row,origin)
            wide = tuple(slice(s.start+pad,s.stop+pad) for s in core)
            trace_error = float(np.max(abs(tensors[pair][...,[0,3,5]].sum(-1)-(rho[wide].reshape(2,16,16,16)-1))))
            assert trace_error < 1e-8
            chunks=[]
            for first in range(0,32,4):
                path=ROOT/'fullcap_tides'/phase/cap/f'd{first:03d}'/'COMPLETE.json'
                rec=json.loads(path.read_text())
                assert rec['passed'] and rec['first']==first and rec['draws']==4
                with h5py.File(rec['output']['path']) as f:
                    eig=f['eigenvalues'][(slice(None),*core,slice(None))].reshape(4,2,16,16,16,3)
                    # Match the frozen scorer's float32 mean arithmetic exactly.
                    chunks.append(eig.mean((2,3,4)).reshape(4,6))
            result=comparison(np.concatenate(chunks),full,matched)
            result.update(pair_id=pair,cap=cap,trace_maxabs=trace_error,truth_receipt=provenance)
            cases.append(result)
            print(json.dumps(dict(stage='evaluated',phase=phase,**result)),flush=True)
        del rho,tensors
    saved=json.loads((ROOT/'interim_scores'/phase/'SCORES.json').read_text())
    reference={r['pair_id']:r['metrics']['joint_tidal_regional']['rank'] for r in saved['cases']}
    assert all(abs(v['fullbox_joint']['rank']-reference[v['pair_id']])<1e-10 for v in cases)
    c.atomic_json(directory/'CLOSURE.json',dict(phase=phase,cases=cases,
        fullbox_joint90=float(np.mean([v['fullbox_joint']['covered']['0.9'] for v in cases])),
        matched_joint90=float(np.mean([v['matched_joint']['covered']['0.9'] for v in cases])),
        prepared=c.file_record(directory/'PREPARED.json',content_hash=True),
        operator_sha256=c.sha256(sys.modules[CapTidal.__module__].__file__),
        script_sha256=c.sha256(__file__),job=os.environ['SLURM_JOB_ID'],passed=True,
        posterior_qualified=False,scope='development only; finite-domain and discretisation closure, not retraining'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','evaluate'],required=True)
    p.add_argument('--phase',choices=['ph012','ph013'],required=True)
    a=p.parse_args();c.require_compute();auth.frozen(a.phase,FREEZE)
    tick=time.monotonic()
    (prepare if a.stage=='prepare' else evaluate)(a.phase)
    print(json.dumps(dict(stage=a.stage,phase=a.phase,seconds=time.monotonic()-tick)),flush=True)
