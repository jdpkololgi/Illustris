"""New campaign authority; legacy phase guards are not relaxed or patched."""
import argparse
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_spatial_exterior as spatial

CONFIG = c.REPO/'configs/e2e_exterior_campaign_20261002.json'
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1')


def phase_guard(phase, purpose):
    config = json.loads(CONFIG.read_text())
    if purpose == 'fit':
        allowed = config['training_phases']
    elif purpose == 'development':
        allowed = config['development_selection_phases']
    else:
        raise PermissionError('Final evaluation requires a separately frozen selection manifest')
    if phase not in allowed:
        raise PermissionError(f'{purpose} access prohibited for {phase}')


def target(phase, pair_id, purpose):
    phase_guard(phase, purpose)  # before any target receipt/payload access
    from workflows.sbi.e2e_coupled_product_audit import read_target
    path = coord.ROOT/'targets'/phase/(pair_id+'.json')
    receipt = coord.verify_receipt(path, payload=False)
    if receipt['phase'] != phase or receipt['pair_id'] != pair_id or len(receipt['outputs']) != 1:
        raise ValueError('Target identity mismatch')
    item = receipt['outputs'][0]
    payload = c.guarded(item['path'], phase)
    if payload.parent != path.parent.resolve() or c.sha256(payload) != item['sha256']:
        raise ValueError('Target lineage mismatch')
    return read_target(payload, phase, pair_id), c.file_record(path, content_hash=True)


def fit(out):
    c.require_compute()
    out.mkdir(parents=True, exist_ok=False)
    moments, sources = [], []
    start = time.monotonic()
    for phase in c.TRAIN:
        phase_guard(phase, 'fit')
        geometry_path = coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
        geometry = coord.verify_receipt(geometry_path, payload=False)
        phase_rows = []
        for row in geometry['pairs']:
            values, source = target(phase, row['pair_id'], 'fit')
            phase_rows.append(spatial.training_moments(values['rho_joint']))
            sources.append(source)
        moments.append(phase_rows)
        print(json.dumps(dict(fit_phase=phase, pairs=len(phase_rows), seconds=time.monotonic()-start)), flush=True)
    result = spatial.fit_prior(moments)
    result.update(training_phases=list(c.TRAIN), sources=sources,
                  pair_counts=[len(x) for x in moments], job=os.environ['SLURM_JOB_ID'],
                  code_sha256=c.sha256(spatial.__file__), runner_sha256=c.sha256(__file__),
                  protocol_sha256=c.sha256(c.REPO/'docs/e2e_exterior_campaign_20261002.md'))
    c.atomic_json(out/'PRIOR.json', result)
    c.atomic_json(out/'MOMENTS.json', dict(phases=list(c.TRAIN), values=moments))
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources',)}), flush=True)


def verify(out):
    c.require_compute()
    out.mkdir(parents=True, exist_ok=False)
    import torch
    fit = dict(log_mean=-.25, lengths_cells=[1., 2.], weights=[.3, .2])
    shape = (12, 10, 8)
    cpu, gpu = spatial.SpatialPrior(shape, fit), spatial.CudaSpatialPrior(shape, fit)
    rng = np.random.default_rng(3381)
    known = rng.uniform(size=shape) < .4
    interior = rng.normal(size=shape)
    a, ar = cpu.conditional(interior, known, np.random.default_rng(37), rtol=1e-9)
    b, br = gpu.conditional(interior, known, np.random.default_rng(37), rtol=1e-9)
    error = float(np.max(abs(a-b)))
    if error > 1e-7:
        raise ValueError('CPU/GPU Gaussian conditional parity failed')
    torch.cuda.synchronize()
    c.atomic_json(out/'PARITY.json', dict(passed=True, cpu=ar, gpu=br, maxabs=error,
                  code_sha256=c.sha256(spatial.__file__), job=os.environ['SLURM_JOB_ID']))
    print('CPU/GPU exact-law parity passed', error, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['fit', 'verify'])
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    globals()[args.action](args.out)


if __name__ == '__main__':
    main()
