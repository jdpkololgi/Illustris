"""Build versioned P12-A low-z indices and raw field/support products.

The bounded first run is ph001/ph006 only. It does not declare training-ready
inputs: spatial folds, full/OOF selection and transforms still follow this stage.
Historical payloads are immutable. No old radial selection is extrapolated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from shared.phase_policy import bound_policy
from workflows.abacus_tweb.p10_phase_assets import load_registry
from workflows.abacus_tweb.p10_redshift_contract import domain_from, require_matching_domains
from workflows.abacus_tweb.p10_training_contract import sha256, atomic_json

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/p12a_lowz_preparation_20261029.json'


def read(path):
    return json.loads(Path(path).read_text())


def load_plan(path=PLAN):
    plan = read(path)
    if plan['schema'] != 'p12a-lowz-preparation-v1':
        raise ValueError('Unsupported low-z preparation plan')
    domain = domain_from(plan)
    if domain['science'] != [.01, .55] or domain['context'] != [0., .60]:
        raise ValueError('Low-z science/context range drift')
    source_path = REPO / plan['source_inputs']
    if sha256(source_path) != plan['source_inputs_sha256']:
        raise ValueError('Bound source input configuration changed')
    source = read(source_path)
    policy = bound_policy(source)
    if plan['smoothing_radius_mpc_h'] != policy['gaussian_smoothing_mpc_h']:
        raise ValueError('R7 policy changed')
    if plan['canary_phases'] != ['ph001', 'ph006'] or plan['model_inputs_ready'] is not False:
        raise ValueError('Preparation scope/readiness changed')
    root = Path(plan['output_root'])
    old = Path(source['source_root'])
    if root.resolve() == old.resolve() or root.resolve().is_relative_to(old.resolve()):
        raise ValueError('New preparation must not write into historical inputs')
    return plan, source, policy


def make_registry(plan, source):
    registry = copy.deepcopy(load_registry(REPO / plan['source_registry']))
    if registry['phase_policy'] != source['phase_policy']:
        raise ValueError('Registry/source phase policy differs')
    registry['reuse_source_root'] = source['source_root']
    registry['path_templates']['phase_output'] = plan['output_root'] + '/{phase}'
    registry['redshift_contract'] = domain_from(plan)
    return registry


def make_schemas(plan):
    domain = domain_from(plan)
    p3 = read(REPO / 'docs/evidence/p3/p3_field_schema_v1.json')
    p4 = read(REPO / 'docs/evidence/p4/p4_spatial_schema_v1.json')
    p3['redshift_contract'] = domain
    p4['redshift_contract'] = domain
    p3['radial_support']['z_context'] = domain['context']
    p3['radial_support']['sentinel_excluded'] = domain['sentinel']
    p3['angular_support']['method'] = 'registered 18-realisation angular random support, cap-separated'
    p3['angular_support']['status'] = 'random footprint; no old radial selection'
    p3['selection'].update(ntilde_path=None, interpolation='deferred training-only fit',
                           expected_counts='zero placeholder until new full/OOF selection fit')
    return p3, p4


def run_command(command, log, deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError('Approved preparation window exhausted')
    with log.open('x') as handle:
        subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                       check=True, timeout=seconds)


def audit_index(phase_root, source_p1, domain):
    import fitsio
    import numpy as np
    p1 = read(phase_root / 'p1_canonical/manifest.json')
    require_matching_domains(p1, {'redshift_contract': domain})
    if p1['parent_sha256'] != source_p1['parent_sha256']:
        raise ValueError('Low-z build changed the corrected catalogue')
    with np.load(phase_root / 'p1_canonical/canonical_index.npz') as index:
        active = index['active']; context = index['context']; valid = index['valid_target']
        ids = index['targetid']
        table = fitsio.read(p1['parent'], columns=['TARGETID', 'Z', 'FILE_NUM', 'BOX_INDEX', 'HALO_INDEX'])
        if not np.array_equal(ids, table['TARGETID']):
            raise ValueError('Index/catalogue row identity failed')
        z = np.asarray(table['Z'], dtype=np.float64)
        expected = (z >= domain['science'][0]) & (z < domain['science'][1]) & valid
        if not np.array_equal(active, expected) or np.any(active & ~context):
            raise ValueError('Low-z supervision/context membership failed')
        counts = np.histogram(z[active], domain['diagnostic_edges'])[0]
        low = active & (z < .1)
        # Audit old host keys, without treating negative legacy indices as a
        # validity cut or silently replacing them with a fictitious halo ID.
        keys = np.rec.fromarrays([table[k][low] for k in ('FILE_NUM', 'BOX_INDEX', 'HALO_INDEX')])
        _, multiplicity = np.unique(keys, return_counts=True)
        return dict(active=int(active.sum()), context=int(context.sum()),
                    active_by_diagnostic_bin=counts.tolist(), diagnostic_edges=domain['diagnostic_edges'],
                    lowz_below_0p1=int(low.sum()),
                    lowz_negative_legacy_box=int((low & (table['BOX_INDEX'] < 0)).sum()),
                    lowz_repeated_legacy_host_groups=int((multiplicity > 1).sum()),
                    lowz_largest_legacy_host_group=int(multiplicity.max(initial=0)),
                    host_key_policy_qualified=False, pass_membership=True)


def prepare_phase(phase, plan, source, output, deadline, *, campaign=False):
    policy = bound_policy(source)
    allowed = (policy['training'] + policy['development_and_calibration']
               if campaign else plan['canary_phases'])
    if phase not in allowed or phase in policy['final_test']:
        raise PermissionError('Phase outside approved preparation scope')
    import numpy as np
    from workflows.sbi.p12a_parent_response import sample_centres
    from workflows.abacus_tweb import p3br_build_random_response as response
    from workflows.abacus_tweb.p6_refit_fullcap_selection import radius_to_redshift_grid
    started = time.monotonic()
    phase_root = output / phase
    phase_root.mkdir()
    logs = phase_root / 'logs'; logs.mkdir()
    old = Path(source['source_root']) / phase
    old_p1 = read(old / 'p1_canonical/manifest.json')
    old_record = read(Path(source['full_contract']) / 'phases' / phase / 'phase_contract.json')
    if sha256(old / 'p1_canonical/manifest.json') != old_record['inputs']['p1_manifest_sha256']:
        raise ValueError('Source P1 binding changed')
    run_command([sys.executable, '-u', '-m', 'workflows.abacus_tweb.p10_build_phase_index',
                 '--registry', str(output / 'registry.json'), '--phase', phase,
                 '--observed', old_p1['parent'], '--out-dir', str(phase_root / 'p1_canonical')],
                logs / 'index.log', deadline)
    membership = audit_index(phase_root, old_p1, domain_from(plan))
    atomic_json(phase_root / 'INDEX_AUDIT.json', membership)
    print(json.dumps(dict(phase=phase, event='index_complete', **membership)), flush=True)
    run_command([sys.executable, '-u', '-m', 'workflows.abacus_tweb.p10_materialize_phase_schemas',
                 '--registry', str(output / 'registry.json'), '--phase', phase,
                 '--phase-root', str(phase_root), '--p3-base', str(output / 'p3_base.json'),
                 '--p4-base', str(output / 'p4_base.json')], logs / 'schemas.log', deadline)
    angular = old / 'p3b_random_response_v1/angular/randoms_n18.npz'
    run_command([sys.executable, '-u', '-m', 'workflows.abacus_tweb.p3a_build_canonical_fields',
                 '--points', str(phase_root / 'p1_canonical/points.npy'),
                 '--canonical-index', str(phase_root / 'p1_canonical/canonical_index.npz'),
                 '--p1-manifest', str(phase_root / 'p1_canonical/manifest.json'),
                 '--schema', str(phase_root / 'contracts/p3_schema_v1.json'),
                 '--angular-random-map', str(angular), '--defer-selection',
                 '--out-dir', str(phase_root / 'p3_fields')], logs / 'fields.log', deadline)
    p3 = read(phase_root / 'p3_fields/field_manifest.json')
    require_matching_domains(p3, plan)
    points = np.load(phase_root / 'p1_canonical/points.npy', mmap_mode='r')
    maps = response.load_map(angular)
    radii, redshifts = radius_to_redshift_grid(0., .60)
    geometry = dict(redshift_contract=domain_from(plan),
                    cosmology=dict(radius_grid_mpc=radii.tolist(), redshift_grid=redshifts.tolist()))
    # Only geometry is required for these two response features. This does not
    # claim that a radial number-density selection has been fitted.
    support_report = {}
    import fitsio
    z = fitsio.read(old_p1['parent'], columns=['Z'])['Z']
    for cap, name in response.CAPS:
        mask = np.asarray(maps['support'], bool) & (maps['domain'] // 2 == cap)
        angle = response.angular_boundary_distance(mask)
        ids = np.flatnonzero((points[:, 3] == cap) & (z >= .01) & (z < .15))
        chosen = ids[np.linspace(0, len(ids)-1, min(2048, len(ids)), dtype=int)]
        distance, support = sample_centres(points[chosen], p3['components'][name], mask, angle, geometry)
        support_report[name] = dict(sample_rows=len(chosen), inside_grid=int(np.isfinite(distance).sum()),
                                    supported=int(support.sum()), selection_fitted=False)
        if not len(chosen) or not np.isfinite(distance).all() or not support.any():
            raise ValueError('Low-z response/grid geometry failed for ' + name)
    record = dict(phase=phase, redshift_contract=domain_from(plan), index_audit=membership,
                  fields=str(phase_root / 'p3_fields/field_manifest.json'),
                  fields_sha256=sha256(phase_root / 'p3_fields/field_manifest.json'),
                  lowz_sample_support=support_report, elapsed_seconds=time.monotonic()-started,
                  model_inputs_ready=False, selection_fitted=False, pass_raw_preparation=True)
    atomic_json(phase_root / 'RAW_PREPARATION_COMPLETE.json', record)
    print(json.dumps(dict(phase=phase, event='raw_preparation_complete',
                         elapsed_seconds=record['elapsed_seconds'])), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--seconds', type=int, default=3000)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('Real preparation requires an approved compute-node allocation')
    if not 0 < args.seconds <= 3000:
        raise ValueError('Preparation must fit the approved one-hour allocation')
    plan, source, _ = load_plan(args.plan)
    output = Path(plan['output_root'])
    output.mkdir(parents=True, exist_ok=False)
    registry = make_registry(plan, source)
    atomic_json(output / 'registry.json', registry)
    for name, schema in zip(('p3', 'p4'), make_schemas(plan)):
        atomic_json(output / f'{name}_base.json', schema)
    sources = [Path(__file__), REPO / 'workflows/abacus_tweb/p10_build_phase_index.py',
               REPO / 'workflows/abacus_tweb/p10_redshift_contract.py',
               REPO / 'workflows/abacus_tweb/p3a_build_canonical_fields.py',
               REPO / 'workflows/sbi/p12a_parent_response.py']
    binding = dict(plan_sha256=sha256(args.plan), source_sha256={str(p):sha256(p) for p in sources},
                   job=os.environ['SLURM_JOB_ID'], host=socket.gethostname())
    atomic_json(output / 'RUN_BINDING.json', binding)
    deadline = time.monotonic() + args.seconds
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(prepare_phase, p, plan, source, output, deadline)
                       for p in plan['canary_phases']]
            results = [future.result() for future in futures]
        if any(sha256(p) != binding['source_sha256'][str(p)] for p in sources):
            raise ValueError('Source code changed during preparation')
        atomic_json(output / 'RAW_PREPARATION_COMPLETE.json',
                    dict(binding=binding, phases=results, model_inputs_ready=False, pass_raw_preparation=True))
    except Exception as exc:
        atomic_json(output / 'PREPARATION_FAILED.json',
                    dict(binding=binding, model_inputs_ready=False, error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
