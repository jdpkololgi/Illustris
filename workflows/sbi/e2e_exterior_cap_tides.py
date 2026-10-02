"""A complete cap density draw and its free-space tidal pushforward.

Each cap is a separate modular posterior approximation. It is NOT a joint
two-cap posterior and does not condition its exterior on the opposite cap.
The approximation must remain explicit in every receipt and in the website.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_footprint_tidal_operator as tidal
from workflows.sbi import e2e_spatial_exterior as spatial
from workflows.sbi.e2e_exterior_fullcap import inference_guard


class CapTidal:
    """Exact cuboid free-space convolution, using float64 FFT arithmetic."""
    def __init__(self, shape, padding):
        import torch
        self.t = torch
        self.target_shape = tuple(shape)
        self.padding = padding
        self.source_shape = tuple(n+2*padding for n in shape)
        before = time.monotonic()
        kernels = tidal.kernel(self.source_shape, shape, offset=(padding,)*3)
        self.fft_shape = tuple(kernels.shape[1:])
        self.spectra = [torch.fft.rfftn(torch.as_tensor(k, device='cuda')) for k in kernels]
        del kernels; gc.collect()
        print(json.dumps(dict(tidal_kernels_seconds=time.monotonic()-before,
                              source_shape=self.source_shape, fft_shape=self.fft_shape)), flush=True)

    def components(self, density):
        t = self.t
        if density.shape != self.source_shape or not np.isfinite(density).all():
            raise ValueError('Invalid complete density source')
        spectrum = t.fft.rfftn(t.as_tensor(density-1, device='cuda', dtype=t.float64), s=self.fft_shape)
        cut = tuple(slice(0, n) for n in self.target_shape)
        for kernel in self.spectra:
            yield t.fft.irfftn(spectrum*kernel, s=self.fft_shape)[cut].cpu().numpy()


def run(field_receipt, prior_path, out, method='learned', padding=32, freeze=None, limit=None):
    c.require_compute()
    import torch
    from workflows.sbi import e2e_loa_field_sampling as sampling
    record = json.loads(field_receipt.read_text())
    cap = record['cap']
    old_loa = (field_receipt.resolve() == Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_global_canvas_20261002_v1')/cap/'COMPLETE.json')
    if 'phase' not in record and not old_loa:
        raise ValueError('Missing field phase identity')
    phase = 'Loa' if old_loa else record['phase']
    inference_guard(phase, freeze)
    if method not in ('learned', 'conditional', 'unconditional', 'zero') or padding not in (16, 32, 64):
        raise ValueError('Unregistered density completion')
    if not record['passed'] or c.sha256(record['field']['path']) != record['field']['sha256']:
        raise ValueError('Full-cap density source changed')
    if phase in ('ph018', 'ph019'):
        selected = json.loads(freeze.read_text())
        if method not in (selected['method'], 'zero') or padding != selected['fullcap_padding']:
            raise PermissionError('Frozen full-cap completion changed')
    fit = json.loads(prior_path.read_text())
    if fit['training_phases'] != list(c.TRAIN):
        raise ValueError('Spatial prior is not training-only')
    sampling.setup(True)
    out.mkdir(parents=True, exist_ok=False)
    binding = dict(phase=phase, cap=cap, method=method, padding=padding,
                   input_receipt=c.file_record(field_receipt, content_hash=True),
                   prior=c.file_record(prior_path, content_hash=True),
                   sources={Path(v.__file__).name: c.sha256(v.__file__) for v in (spatial, tidal)},
                   runner_sha256=c.sha256(__file__), posterior_qualified=False,
                   cap_joint_law='independent capwise modular approximation',
                   opposite_cap_observations_condition_exterior=False,
                   freeze_sha256=None if freeze is None else c.sha256(freeze))
    c.atomic_json(out/'BINDING.json', binding)
    with h5py.File(record['field']['path']) as source:
        shape = source['rho'].shape[1:]
        support = source['support'][:]
        count = source['rho'].shape[0] if limit is None else min(limit, source['rho'].shape[0])
        origin = np.asarray(source.attrs['origin_mpc_h'])
        first = 0 if old_loa else int(source.attrs['first_draw'])
        start = time.monotonic()
        prior = spatial.CudaSpatialPrior(tuple(n+2*padding for n in shape), fit, workers=4)
        operator = CapTidal(shape, padding)
        setup_seconds = time.monotonic()-start
        diagnostics = []
        output = out/'COMPLETED_FIELD.h5'
        with h5py.File(output, 'x') as f:
            complete = f.create_dataset('rho_completed', shape=(count, *prior.shape), dtype='f4',
                                        chunks=(1, 16, 32, 32), compression='lzf')
            eigen = f.create_dataset('eigenvalues', shape=(count, *shape, 3), dtype='f4',
                                     chunks=(1, 8, 32, 32, 3), compression='lzf')
            f.create_dataset('support', data=support, compression='lzf')
            f.create_dataset('coverage', data=source['coverage'][:], compression='lzf')
            f.attrs.update(phase=phase, cap=cap, cell_mpc_h=6.766, origin_mpc_h=origin,
                           completed_origin_mpc_h=origin-padding*6.766, first_draw=first,
                           method=method, padding_cells=padding, posterior_qualified=False,
                           cap_joint_law=binding['cap_joint_law'], smoothing_mpc_h=7., smoothing_count=1)
            for i in range(count):
                tick = time.monotonic()
                rho = source['rho'][i]
                seed = int(c.digest(['spatial-exterior-cap-v1', phase, cap, first+i])[:15], 16)
                field, diagnostic = spatial.complete_density(rho, support, prior, np.random.default_rng(seed), method, padding)
                completion_seconds = time.monotonic()-tick
                complete[i] = field.astype('f4')
                # The measured full-cap smoke peak is below 15 GiB host RSS.
                # This 1–2 GiB array avoids unreclaimed HDF5 scratch chunks
                # inflating every completed product by a temporary tensor.
                temp = np.empty((*shape, 6), dtype=np.float64)
                for component, values in enumerate(operator.components(field)):
                    temp[..., component] = values
                    del values
                trace_error = 0.
                for lo in range(0, shape[0], 8):
                    hi = min(lo+8, shape[0]); tensor = temp[lo:hi]
                    local = field[padding+lo:padding+hi, padding:padding+shape[1], padding:padding+shape[2]]
                    trace_error = max(trace_error, float(np.max(abs(tensor[..., [0, 3, 5]].sum(-1)-(local-1)))))
                    eigen[i, lo:hi] = tidal.eigensystem(tensor).astype('f4')
                if trace_error > 1e-9*max(1., float(field.max())):
                    raise ValueError('Full-cap free-space trace identity failed')
                del temp
                diagnostic.update(draw=first+i, seconds=time.monotonic()-tick,
                                  completion_seconds=completion_seconds, trace_maxabs=trace_error,
                                  minimum_density=float(field.min()))
                diagnostics.append(diagnostic); print(json.dumps(diagnostic), flush=True)
                del field, rho; gc.collect()
        final = dict(phase=phase, cap=cap, first=first, draws=count, method=method,
                     binding=c.sha256(out/'BINDING.json'), diagnostics=diagnostics,
                     setup_seconds=setup_seconds, seconds=time.monotonic()-start,
                     output=c.file_record(output, content_hash=True), passed=True,
                     posterior_qualified=False, cap_joint_law=binding['cap_joint_law'],
                     job=os.environ['SLURM_JOB_ID'])
        c.atomic_json(out/'COMPLETE.json', final)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--field-receipt', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--method', default='learned'); p.add_argument('--padding', type=int, default=32)
    p.add_argument('--freeze', type=Path); p.add_argument('--limit', type=int)
    a = p.parse_args(); run(a.field_receipt, a.prior, a.out, a.method, a.padding, a.freeze, a.limit)
