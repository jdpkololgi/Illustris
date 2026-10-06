"""Matched shortened full-cap trajectories: comparability, NOT coverage repair.

No truth access, physical field output, fitted mixture weight or scientific score.
Reference and extended seed17 must match exactly at NFE2 on both dev SGC caps.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflows.sbi.e2e_cfm_mixture_contract import PHASES, component_seed, pilot_guard

BASE = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1')
FROZEN = BASE/'source_78156ac3417555ef269f'
OUT = BASE.parent/'cfm_mixture_pilot_20261006_v1'
REFERENCE_SHA = '7940a599d41a865543a53a32fadc96c0e0491b8269e986a769237a57e080803c'


def dependencies(c):
    """Every loaded shared project dependency must equal the frozen baseline."""
    receipts = []
    excluded = {'workflows.sbi.e2e_exterior_fullcap',
                'workflows.sbi.e2e_cfm_mixture_contract'}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if not path or not name.startswith(('workflows.', 'shared.')) or name in excluded:
            continue
        relative = Path(*name.split('.'))
        relative = relative/'__init__.py' if hasattr(module, '__path__') else relative.with_suffix('.py')
        reference = FROZEN/relative
        if not reference.is_file():
            raise ValueError('Dependency missing from frozen reference: '+name)
        sha = c.sha256(path)
        if sha != c.sha256(reference):
            raise ValueError('Shared dependency changed: '+name)
        receipts.append(dict(module=name, sha256=sha))
    return receipts


def run(phase, mode):
    import numpy as np
    import torch
    from workflows.sbi import e2e_coupled_contract as c
    from workflows.sbi import e2e_global_canvas as canvas
    from workflows.sbi import e2e_exterior_fullcap as current
    from workflows.sbi import e2e_coupled_views as views
    from workflows.sbi.e2e_vdm_context_models import decode_density, block_mean

    seed = 29 if mode == 'seed29' else 17
    pilot_guard(phase, seed)
    c.require_compute()
    live_sources = [Path(__file__), Path(current.__file__),
                    Path(__file__).with_name('e2e_cfm_mixture_contract.py')]
    initial_hashes = {str(p): c.sha256(p) for p in live_sources}
    cap = 'SGC'  # predetermined smaller cap in BOTH phases, not score-selected
    destination = OUT/phase/mode
    reference_path = FROZEN/'workflows/sbi/e2e_exterior_fullcap.py'
    if c.sha256(reference_path) != REFERENCE_SHA:
        raise ValueError('Frozen sampler identity changed')
    baseline_path = BASE/'fullcap_density'/phase/cap/'d000/BINDING.json'
    baseline = json.loads(baseline_path.read_text())
    input_root = BASE/'fullcap_inputs'/phase/cap
    if baseline['prepared_sha256'] != c.sha256(input_root/'PREPARED.json'):
        raise ValueError('Prepared inputs differ from existing NFE128 baseline')
    if baseline['runner_sha256'] != REFERENCE_SHA or baseline['integrator_sha256'] != c.sha256(canvas.__file__):
        raise ValueError('Integrator or original sampler identity changed')
    chart = views.load_chart(baseline['normalizer'])
    traces = {}
    states = {}
    original_integrate = canvas.integrate

    def capture(model, stage, base, noise, geometry, nfe, callback):
        if nfe != 2:
            raise ValueError('This pilot permits only NFE2 trajectories')
        value, coverage = original_integrate(model, stage, base, noise, geometry, nfe, callback)
        torch.cuda.synchronize()
        digest = hashlib.sha256()
        finite = True
        # Bound host memory: fingerprint one draw at a time.
        for draw in value:
            a = draw.detach().cpu().numpy()
            finite = finite and bool(np.isfinite(a).all())
            digest.update(a.tobytes())
        if not finite:
            raise FloatingPointError('Nonfinite shortened trajectory')
        traces[stage] = dict(shape=list(value.shape), sha256=digest.hexdigest(), finite=finite,
                             coverage_sha256=hashlib.sha256(coverage.cpu().numpy().tobytes()).hexdigest())
        # One predeclared local crop for decoder contract; no truth or scoring.
        states[stage] = value[geometry.crop(0)].detach().clone()
        return value, coverage

    canvas.integrate = capture
    started = time.monotonic()
    try:
        if mode == 'reference17':
            spec = importlib.util.spec_from_file_location('frozen_fullcap_reference', reference_path)
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
            reference.run(input_root, destination, benchmark=True)
        else:
            current.run(input_root, destination, benchmark=True, model_seed=seed)
    finally:
        canvas.integrate = original_integrate
    binding = json.loads((destination/'BINDING.json').read_text())
    if binding['normalizer'] != baseline['normalizer']:
        raise ValueError('Component normalizer differs from baseline')
    if seed == 17 and binding['checkpoints'] != baseline['checkpoints']:
        raise ValueError('Seed17 checkpoint identity differs from baseline')
    if seed == 29 and any(binding['checkpoints'][s] == baseline['checkpoints'][s] for s in ('coarse', 'fine')):
        raise ValueError('Seed29 unexpectedly reuses a baseline checkpoint')
    # Unchanged exact-mass decoder, using the corresponding generated factors.
    z = states['coarse'][:, :, 16:32, 18:30, 18:30].double()
    u = states['fine'].double()
    coarse = torch.exp(z*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
    rho = decode_density(coarse, u*chart['fine_residual']['std'][0])
    mass = float(((block_mean(rho)-coarse)/coarse).abs().max())
    if not torch.isfinite(rho).all() or not (rho > 0).all() or mass >= 2e-6:
        raise ValueError('Pilot decoder contract failed')
    dependency_receipts = dependencies(c)
    if initial_hashes != {str(p):c.sha256(p) for p in live_sources}:
        raise ValueError('Pilot source changed during execution')
    benchmark = json.loads((destination/'BENCHMARK.json').read_text())
    result = dict(phase=phase, cap=cap, mode=mode, model_seed=seed, effective_nfe=2,
        draws=4, traces=traces, decoder_max_mass_error=mass, decoder_positive=True,
        seconds=time.monotonic()-started, peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        benchmark=benchmark, source=c.file_record(__file__, content_hash=True),
        sampler=c.file_record(reference_path if mode=='reference17' else current.__file__, content_hash=True),
        contract=c.file_record(Path(__file__).with_name('e2e_cfm_mixture_contract.py'), content_hash=True),
        binding=c.file_record(destination/'BINDING.json', content_hash=True), dependencies=dependency_receipts,
        input_baseline=c.file_record(baseline_path, content_hash=True), passed=True,
        coverage_improvement_tested=False, full_nfe_parity_tested=False,
        question='Can both fixed trained models be compared with an unchanged full-cap sampler?',
        meaning='Short-trajectory engineering evidence only; no calibrated posterior or coverage repair claim.')
    c.atomic_json(destination/'PILOT.json', result)
    print(json.dumps({k:result[k] for k in ('phase','mode','seconds','peak_host_gib','decoder_max_mass_error','passed')}), flush=True)


def summarize():
    from workflows.sbi import e2e_coupled_contract as c
    c.require_compute()
    rows = []
    for phase in PHASES:
        data = {mode:json.loads((OUT/phase/mode/'PILOT.json').read_text())
                for mode in ('reference17','seed17','seed29')}
        if not all(v['passed'] and v['effective_nfe']==2 for v in data.values()):
            raise ValueError('Incomplete pilot')
        if data['reference17']['traces'] != data['seed17']['traces']:
            raise ValueError('Seed17 shortened trajectory does not exactly replay frozen implementation')
        for stage in ('coarse','fine'):
            if data['seed29']['traces'][stage]['shape'] != data['seed17']['traces'][stage]['shape']:
                raise ValueError('Component geometry differs')
        rows.append(dict(phase=phase, exact_short_trajectory_replay=True,
                         component17=data['seed17'],component29=data['seed29'],reference=data['reference17']))
    result = dict(passed=True, cases=rows, candidate_weights=[.5,.5],
        future_component_assignments={p:[component_seed(p,d) for d in range(32)] for p in PHASES},
        scientific_coverage_result='not tested', full_nfe_parity_tested=False,
        next_decision='Separately authorize matched full-NFE generation only after reviewing throughput and source checks.',
        scope='Two SGC full-footprint NFE2 trajectories per model; no truth, training, property painting or sealed-phase access.')
    c.atomic_json(OUT/'SUMMARY.json', result)
    print(json.dumps(dict(passed=True,scientific_coverage_result='not tested',summary=str(OUT/'SUMMARY.json'))),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--mode', choices=('reference17','seed17','seed29'))
    parser.add_argument('--summarize', action='store_true')
    args=parser.parse_args()
    if args.summarize:
        summarize()
    elif args.phase and args.mode:
        run(args.phase,args.mode)
    else:
        parser.error('Specify --summarize or both --phase and --mode')
