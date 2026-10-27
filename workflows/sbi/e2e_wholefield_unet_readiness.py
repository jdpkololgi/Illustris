"""Audit the existing 13-phase endpoint U-Net without graph products or new fits.

Metadata mode only reads small receipts and file sizes. Compute mode hashes and
checks the actual training/development payloads and runs disposable GPU updates.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_pilot_20261013_v1')
COORD = ROOT.parent / 'coupled_20260918_v1/cartesian_v2'
TRAIN = ('ph000', 'ph002', 'ph003', 'ph007', 'ph008', 'ph009', 'ph010',
         'ph011', 'ph020', 'ph021', 'ph022', 'ph023', 'ph024')
DEV = ('ph012', 'ph013')
SHAPE = (96, 64, 64)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def record_check(record, payload=False):
    path = Path(record['path'])
    require(path.is_file() and path.stat().st_size == record['bytes'], f'Missing/changed size: {path}')
    if payload:
        require(sha(path) == record['sha256'], f'Hash mismatch: {path}')


def metadata(source_revision=None):
    normalizer = COORD / 'normalization/NORMALIZATION_COMPLETE.json'
    norm = read(normalizer)
    require(norm['pass'] and norm['fit_phases'] == list(TRAIN)
            and norm['phase_weight'] == 1 / 13, 'Normalizer train-only contract')
    norm_sha = sha(normalizer)
    domains, bindings, phases = [], {str(normalizer): norm_sha}, {}
    total_bytes = 0
    for phase in TRAIN + DEV:
        path = ROOT / 'data' / phase / 'COMPLETE.json'
        value = read(path)
        require(value['passed'] and tuple(value['shape']) == SHAPE
                and value['normalizer'] == norm_sha
                and value['training'] == (phase in TRAIN), f'Phase contract: {phase}')
        rows = value['pairs']
        expected = 48 if phase in TRAIN else 16
        require(len(rows) == expected and len({r['pair_id'] for r in rows}) == expected,
                f'Address coverage: {phase}')
        require(all(r['phase'] == phase for r in rows), 'Cross-phase address')
        require(all(sum(r['cap'] == cap for r in rows) == expected // 2
                    for cap in ('NGC', 'SGC')), 'Cap balance')
        for source, digest in value['sources'].items():
            require(sha(source) == digest, f'Preparation source changed: {source}')
        record_check(value['geometry'], payload=True)
        bindings[str(path)] = sha(path)
        phases[phase] = {'domains': len(rows), 'role': 'training' if phase in TRAIN else 'development'}
        for row in rows:
            directory = ROOT / 'data' / phase / row['pair_id']
            for kind in ('OBSERVATIONS', 'TARGETS'):
                receipt_path = directory / (kind + '.json')
                receipt = read(receipt_path)
                require(receipt['passed'] and receipt['phase'] == phase
                        and receipt['pair_id'] == row['pair_id'], 'Pair receipt mismatch')
                require(Path(receipt['output']['path']) == directory / (kind + '.npz'),
                        'Payload path differs from actual trainer path')
                record_check(receipt['output'])
                total_bytes += receipt['output']['bytes']
                bindings[str(receipt_path)] = sha(receipt_path)
                for key in (('raw_receipt', 'old_condition') if kind == 'OBSERVATIONS'
                            else ('native_receipt', 'old_target')):
                    record_check(receipt[key], payload=True)
                if kind == 'OBSERVATIONS':
                    require(receipt['contains_matter'] is False
                            and receipt['central_channel_maxabs'] <= 2e-5, 'Observation lineage/parity')
                else:
                    require(max(receipt['central_density_relative'],
                                receipt['coarse_alignment_relative']) <= 2e-6, 'Target parity')
            domains.append((phase, row['pair_id']))
    require(len(domains) == 656, 'Expected 624 training and 32 development domains')
    smoke = read(ROOT / 'smoke/COMPLETE.json')
    require(smoke['passed'], 'Historical smoke failed')
    revision = read(source_revision) if source_revision else {'sources': {}}
    require(set(revision['sources']).issubset(smoke['sources']), 'Unknown source revision')
    current_sources = {}
    for name, digest in smoke['sources'].items():
        if name in revision['sources']:
            binding = revision['sources'][name]
            require(binding['parent_sha256'] == digest, 'Wrong parent source revision')
            digest = binding['sha256']
        require(sha(REPO / 'workflows/sbi' / name) == digest, f'Model source changed: {name}')
        current_sources[name] = digest
    return {'schema': 'field-unet-readiness-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'existing 13-phase 96x64x64 clean-endpoint conditional U-Net',
            'phases': phases, 'domains': domains, 'payload_bytes': total_bytes,
            'normalizer_sha256': norm_sha, 'receipt_bindings': bindings,
            'model_sources': current_sources, 'historical_model_sources': smoke['sources'],
            'source_revision': revision, 'metadata_pass': True,
            'training_ready': False, 'posterior_qualified': False,
            'not_covered': ['25-phase P10 expansion', 'new scientific training run',
                            'posterior calibration', 'fresh held-out evaluation']}


def check_domain(address, chart):
    import numpy as np
    from workflows.sbi import e2e_coupled_operators as op
    from workflows.sbi import e2e_wholefield_pilot_train as train
    phase, pair = address
    directory = ROOT / 'data' / phase / pair
    for name in ('OBSERVATIONS', 'TARGETS'):
        record_check(read(directory / (name + '.json'))['output'], payload=True)
    with np.load(directory / 'OBSERVATIONS.npz') as f:
        require(set(f.files) == {'joint', 'wide'}, 'Observation columns')
    with np.load(directory / 'TARGETS.npz') as f:
        require(set(f.files) == {'rho', 'coarse_wide', 'residual'}, 'Target columns')
        rho, wide, residual = [np.asarray(f[k], dtype='f8') for k in ('rho', 'coarse_wide', 'residual')]
    require(rho.shape == residual.shape == SHAPE and wide.shape == (48, 48, 48), 'Target shapes')
    require(all(np.isfinite(a).all() for a in (rho, wide, residual))
            and rho.min() > 0 and wide.min() > 0, 'Target finite/positive')
    coarse = wide[12:36, 16:32, 16:32]
    mass_error = float(np.max(abs(op.mean_pool(rho) / coarse - 1)))
    reconstruction_error = float(np.max(abs(op.decode(coarse, residual) / rho - 1)))
    require(max(mass_error, reconstruction_error) < 2e-6, 'Density/coarse/residual alignment')
    values = train.arrays(address, chart, training=phase in TRAIN)
    expected = ((1, *SHAPE), (12, *SHAPE), (12, 48, 48, 48), (1, 48, 48, 48))
    require(all(v.shape == shape and np.isfinite(v).all() for v, shape in zip(values, expected)),
            'Actual normalized loader shape/finiteness')
    require(float(np.max(abs(op.mean_pool(values[0][0].astype('f8'))))) < 2e-6,
            'Normalized residual outside block-zero subspace')
    return {'phase': phase, 'pair': pair, 'mass_error': mass_error,
            'reconstruction_error': reconstruction_error, 'pass': True}


def endpoint_smoke(output, chart):
    import copy
    import torch
    from workflows.sbi import e2e_wholefield_pilot_train as train
    from workflows.sbi.e2e_wholefield_pilot_models import FineModel, Condition, sample
    from workflows.sbi.e2e_vdm_context_models import block_mean, decode_density
    train.setup(17)
    ids = train.panel(TRAIN)
    addresses = [ids[0], ids[24]]
    x, cond = train.batch(addresses, chart)
    weight, spectrum = train.spectrum(ids, chart)
    model = FineModel('unet', 'x').cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    first = train.update(model, ema, optimizer, weight, x, cond, 17, 0)
    checkpoint = output / 'DISPOSABLE_ENDPOINT_RESTART.pt'
    train.save(checkpoint, {'model': model.state_dict(), 'ema': ema.state_dict(),
                           'optimizer': optimizer.state_dict(), 'weight': weight,
                           'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()})
    expected_loss = train.update(model, ema, optimizer, weight, x, cond, 17, 1)
    expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    expected_ema = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
    saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
    model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['cpu_rng'].cpu()); torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    actual = train.update(model, ema, optimizer, saved['weight'], x, cond, 17, 1)
    difference = max(float((v.cpu() - expected[k]).abs().max()) for k, v in model.state_dict().items())
    ema_difference = max(float((v.cpu() - expected_ema[k]).abs().max()) for k, v in ema.state_dict().items())
    replay = {'model_maxabs': difference, 'ema_maxabs': ema_difference,
              'loss_abs': abs(actual['loss'] - expected_loss['loss']),
              'expected_update': expected_loss, 'actual_update': actual,
              'pass': max(difference, ema_difference) <= 2e-6
                      and abs(actual['loss'] - expected_loss['loss']) <= 1e-6}
    with (output / 'ENDPOINT_REPLAY.json').open('x') as f:
        json.dump(replay, f, indent=2)
    require(replay['pass'], 'Endpoint restart replay; see ENDPOINT_REPLAY.json')
    development = train.development(ema, chart, weight)
    require(development['available'] and math.isfinite(development['fixed_noise_cfm_loss']),
            'Development loader or loss failed')
    evaluation = Condition(cond.joint[:1], cond.wide[:1], cond.coarse_wide[:1], 'oracle_diagnostic')
    value = sample(ema, evaluation, torch.randn_like(x[:1]), nfe=4)
    cs = chart['coarse_logrho']
    mass = (evaluation.coarse_joint().double() * cs['std'][0] + cs['mean'][0]).exp()
    density = decode_density(block_mean(mass), value.double() * chart['fine_residual']['std'][0])
    error = float(((block_mean(density) / block_mean(mass)) - 1).abs().max())
    require(torch.isfinite(density).all().item() and density.min().item() > 0 and error < 1e-9,
            'Endpoint sampler finite/positive/mass')
    return {'pass': True, 'addresses': addresses, 'first_update': first,
            'replayed_update': actual, 'restart_parameter_maxabs': difference,
            'restart_ema_maxabs': ema_difference, 'mass_relative': error,
            'development': development, 'spectral_fit_addresses': spectrum['ids'],
            'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30,
            'torch_version': torch.__version__, 'scope': 'disposable numerical smoke, not a scientific fit'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('metadata', 'compute'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = metadata(args.source_revision)
    if args.mode == 'compute':
        from workflows.sbi import e2e_coupled_contract as c
        from workflows.sbi import e2e_coupled_views as views
        c.require_compute()
        chart = views.load_chart(report['normalizer_sha256'])
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda a: check_domain(a, chart), report['domains']))
        report['payload_checks'] = results
        with (args.output / 'FIELD_PAYLOADS_VERIFIED.json').open('x') as f:
            f.write(json.dumps({'pass': True, 'domains': results, 'receipt_bindings': report['receipt_bindings']}, indent=2) + '\n')
        print('ALL_656_FIELD_PAYLOADS_PASS', flush=True)
        report['endpoint_smoke'] = endpoint_smoke(args.output, chart)
        final = metadata(args.source_revision)
        require(final['receipt_bindings'] == report['receipt_bindings']
                and final['model_sources'] == report['model_sources'], 'Inputs changed during audit')
        report['training_ready'] = True
    filename = 'FIELD_UNET_READY.json' if report['training_ready'] else 'METADATA_AUDIT.json'
    with (args.output / filename).open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(args.output / filename), 'metadata_pass': True,
                      'training_ready': report['training_ready'], 'payload_bytes': report['payload_bytes']}))


if __name__ == '__main__':
    main()
