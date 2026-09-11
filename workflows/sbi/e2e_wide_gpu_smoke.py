"""User-approved two-update full-size engineering verification, not a science fit."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import h5py
import numpy as np
import torch

from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_data import fit_normalization


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = p.runtime()
    c = p.read_config()
    out = p.output_path(c, args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source = {k: p.sha256(p.REPO/k) for k in p.SOURCE_FILES}
    registration = {'job_id': os.environ['SLURM_JOB_ID'], 'node': socket.gethostname(),
                    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=p.REPO, text=True).strip(),
                    'source_sha256': source, 'harness_sha256': p.sha256(__file__),
                    'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(device), 'config_sha256': p.digest(c),
                    'scope': 'two-update engineering checks only; no science training or holdout access'}
    p.write_json(out/'STARTED.json', registration)
    normalization = out/'normalization.json'
    ds = p.dataset_for(c, None, verify=False)
    print('NORMALIZATION: verify training payloads and fit 96 anchors', flush=True)
    fit_normalization(ds, normalization)
    ds = p.dataset_for(c, normalization)
    binding = p.provenance(c, ds)
    p.write_json(out/'NORMALIZATION_VERIFIED.json', {'sha256': p.sha256(normalization),
                 'parents': len(ds), 'payloads_verified': ds.verified})
    checks, paths = [], []
    for method in ('cfm', 'diffusion'):
        checkpoints = {}
        for stage in ('coarse', 'fine'):
            print(f'TRAIN/RESUME CHECK: {method}/{stage}', flush=True)
            base = out/f'{method}_{stage}'
            full, first, resumed = (Path(str(base)+suffix) for suffix in ('_full', '_first', '_resumed'))
            p.train(c, normalization, stage, method, full, stop_after=2)
            p.train(c, normalization, stage, method, first, stop_after=1)
            p.train(c, normalization, stage, method, resumed, stop_after=2, resume=first/'step_000001.pt')
            a = p.load_checkpoint(full/'step_000002.pt', binding, stage, method)
            b = p.load_checkpoint(resumed/'step_000002.pt', binding, stage, method)
            parity = {key: equal_tree(a[key], b[key]) for key in ('model', 'optimizer', 'rng', 'history', 'step')}
            if not all(parity.values()):
                raise ValueError(f'CUDA resume mismatch: {method}/{stage}: {parity}')
            checks.append({'method': method, 'stage': stage, 'resume_exact': parity})
            checkpoints[stage] = full/'step_000002.pt'
            del a, b
            torch.cuda.empty_cache()
        anchor = ds.rows[0]['anchor_id']
        outputs = [out/f'{method}_draw_{i}.h5' for i in (1, 2)]
        for output in outputs:
            p.sample(c, normalization, checkpoints['coarse'], checkpoints['fine'], anchor, 'smoke-0', output)
        with h5py.File(outputs[0], 'r') as a, h5py.File(outputs[1], 'r') as b:
            replay = {name: np.array_equal(a[name][()], b[name][()]) for name in a}
            density = a['delta_local96'][:]
            overlap = np.array_equal(density[32:64,32:64,40:64], density[32:64,32:64,40:72][..., :24])
        if not all(replay.values()) or not overlap:
            raise ValueError('generated replay/overlap mismatch')
        checks.append({'method': method, 'replay_exact': replay, 'sibling_overlap_exact': overlap})
        paths.append(outputs[0])
    p.diagnose(c, normalization, paths, out/'TRAINING_DRAW_DIAGNOSTICS.json')
    if source != {k: p.sha256(p.REPO/k) for k in source}:
        raise ValueError('source changed during verification')
    result = {'registration': registration, 'checks': checks, 'technical_pass': True,
              'elapsed_seconds': time.monotonic()-start,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'normalization_sha256': p.sha256(normalization),
              'diagnostics_sha256': p.sha256(out/'TRAINING_DRAW_DIAGNOSTICS.json'),
              'training_ready': False, 'r0_physics_pass': False, 'science_training_started': False,
              'heldout_payloads_read': False}
    p.write_json(out/'SMOKE_COMPLETE.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
