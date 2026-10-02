"""Numerical padding follow-up, reusing identical development density draws."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_spatial_exterior as spatial
from workflows.sbi import e2e_footprint_tidal_operator as tidal
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi import e2e_exterior_campaign as campaign
from workflows.sbi.e2e_exterior_panel import CoreTidal, load, METHODS


def run(root, old, out, prior_path, padding=32):
    c.require_compute()
    from workflows.sbi.e2e_loa_field_sampling import setup
    setup(True)
    prepared = json.loads((root/'PREPARED.json').read_text())
    campaign.phase_guard(prepared['phase'], 'development')
    previous = json.loads((old/'BINDING.json').read_text())
    if previous['prepared_sha256'] != c.sha256(root/'PREPARED.json') or previous['draws'] != 32 or padding != 32:
        raise ValueError('Changed development refinement contract')
    fit = json.loads(prior_path.read_text())
    if c.sha256(prior_path) != previous['prior_sha256']:
        raise ValueError('Refinement prior changed')
    prior = spatial.CudaSpatialPrior((160, 112, 112), fit, workers=4)
    operator = CoreTidal(prior.shape, padding)
    binding = dict(previous, padding=padding, runner_sha256=c.sha256(__file__),
                   original_run_sha256=c.sha256(old/'BINDING.json'), same_interior_draws=True)
    out.mkdir(parents=True, exist_ok=False)
    c.atomic_json(out/'BINDING.json', binding)
    for case in prepared['cases']:
        directory = old/case['pair_id']
        record = json.loads((directory/'COMPLETE.json').read_text())
        if record['binding'] != c.digest(previous):
            raise ValueError('Original development binding changed')
        full_item = next(v for v in record['outputs'] if Path(v['path']).stem == 'shared_density')
        full = load(full_item)
        target = out/case['pair_id']; target.mkdir()
        outputs = [full_item]
        # Unchanged physical controls and shared density features retain their
        # original bytes. Padding cannot change a zero exterior source field.
        for name in ('zero', 'periodic'):
            item = next(v for v in record['outputs'] if Path(v['path']).stem == name)
            load(item); outputs.append(item)
        diagnostics, rows = [], {name: [] for name in ('unconditional', 'conditional', 'learned')}
        started = time.monotonic()
        for draw, rho in enumerate(full['rho']):
            for name in rows:
                seed = int(c.digest(['spatial-exterior-v1', case['pair_id'], draw])[:15], 16)
                field, numerical = spatial.complete_density(rho, full['support'], prior, np.random.default_rng(seed), name, padding)
                tensor = operator(field)
                density = targets.owned_core_arrays(field[padding:padding+64, padding:padding+48, padding:padding+48])
                error = float(np.max(abs(tensor[..., [0, 3, 5]].sum(-1)-(density-1))))
                if error > 1e-9*max(1., float(density.max())):
                    raise ValueError('Refined trace identity failed')
                rows[name].append(dict(tidal=tidal.eigensystem(tensor), density=density-1))
                diagnostics.append(dict(numerical, draw=draw, trace_maxabs=error))
        for name, values in rows.items():
            path = target/(name+'.npz')
            np.savez_compressed(path, **{key: np.stack([v[key] for v in values]) for key in values[0]})
            outputs.append(c.file_record(path, content_hash=True))
        result = dict(pair_id=case['pair_id'], phase=case['phase'], binding=c.digest(binding),
                      diagnostics=diagnostics, outputs=outputs, seconds=time.monotonic()-started,
                      original_case=c.file_record(directory/'COMPLETE.json', content_hash=True))
        c.atomic_json(target/'COMPLETE.json', result)
        print(json.dumps(dict(case=case['pair_id'], padding=padding, seconds=result['seconds'])), flush=True)
    c.atomic_json(out/'RUN_COMPLETE.json', dict(cases=len(prepared['cases']), total=len(prepared['cases']),
                                              panel_complete=True, binding=c.digest(binding)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--old', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True); a = p.parse_args(); run(a.inputs, a.old, a.out, a.prior)
