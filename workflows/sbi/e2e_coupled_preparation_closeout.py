"""Metadata/evidence closeout after all independently qualified preparation.

No particle or field payload processing. Does not turn technical success into
scientific training permission. Archives only bounded, hash-verified JSON evidence.
"""
import argparse
import json
from pathlib import Path
import shutil

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_data_release as release
from workflows.sbi import e2e_coupled_normalization as norm
from workflows.sbi import e2e_coupled_interface_qualify as qa
from workflows.sbi import e2e_coupled_loader_benchmark as loader
from workflows.sbi import e2e_coupled_gpu_benchmark as gpu
from workflows.sbi import e2e_coupled_postprocess_benchmark as post
from workflows.sbi import e2e_coupled_physical_gate as physics
from workflows.sbi import e2e_coupled_resource_estimate as cost
from workflows.sbi.e2e_coupled_prepare_ops import accounting
from workflows.sbi.e2e_coupled_stage_b import used_bytes

DESTINATION = c.REPO / 'docs/evidence/e2e_coupled_20260919'
PROPOSAL = c.REPO / 'docs/e2e_coupled_resource_proposal_20260918.md'


def require_terminal_budget(usage, size, approval):
    active = {'RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED'}
    if any(row['state'] in active for row in usage['allocations']):
        raise RuntimeError('owned preparation allocations must finish before closeout')
    if (usage['cpu_node_hours'] > approval['cpu_node_hours']
            or usage['gpu_hours'] > approval['gpu_hours'] or size > approval['scratch_bytes']):
        raise RuntimeError('original preparation budget exceeded')


def require_no_scientific_authority(data, technical):
    if (data['data_products_qualified'] is not True
            or data['scientific_training_authorized'] is not False
            or data['posterior_calibration_claim'] is not False):
        raise ValueError('data qualification or authority boundary mismatch')
    if (technical['synthetic_only'] is not True or technical['scientific_fit'] is not False
            or technical['cases'] != list(gpu.CASES)):
        raise ValueError('complete synthetic-only technical factor panel required')


def collect():
    data_path = coord.ROOT / 'data_release/DATA_PRODUCTS_QUALIFIED.json'
    # Fail BEFORE directory creation or any large-data access on an unfinished panel.
    if not data_path.exists():
        raise FileNotFoundError('all-panel data qualification not yet complete')
    data = coord.verify_receipt(data_path, payload=False)
    normalizer_path = coord.ROOT / 'normalization/NORMALIZATION_COMPLETE.json'
    normalizer = coord.verify_receipt(normalizer_path, payload=False)
    interface_path = coord.ROOT / 'interface_qualification/INTERFACE_COMPLETE.json'
    interface = coord.verify_receipt(interface_path, payload=False)
    if (normalizer['binding'] != norm.source_binding()
            or interface['binding'] != qa.binding(c.sha256(normalizer_path))
            or data['binding']['publisher_sha256'] != c.sha256(release.__file__)):
        raise ValueError('normalization/interface/publisher implementation drift')
    sources = {data_path, normalizer_path, interface_path}
    phases = {phase: release.phase_evidence(phase) for phase in c.ROLES}
    release.require_panel(phases, normalizer, interface)
    if phases != data['phases'] or data['binding']['audit_dependencies'] != release.audit_dependencies():
        raise ValueError('data release/source audit evidence drift')
    for phase, row in phases.items():
        sources.update(Path(row[key]['path']) for key in
                       ('phase_audit', 'audit_pointer', 'normalized_interface'))
        for base, name in ((c.ROOT/'matter', 'DENSITY_COMPLETE.json'),
                           (c.ROOT/'matter', 'PARTICLES_VERIFIED.json'),
                           (c.ROOT/'observations', 'PARENT_COMPLETE.json'),
                           (c.ROOT/'observations', 'OBSERVED_COMPLETE.json'),
                           (c.ROOT/'observations', 'angular/ANGULAR_COMPLETE.json'),
                           (coord.ROOT/'geometry', 'GEOMETRY_COMPLETE.json'),
                           (coord.ROOT/'observations', 'NGC_COMPLETE.json'),
                           (coord.ROOT/'observations', 'SGC_COMPLETE.json'),
                           (coord.ROOT/'conditions', 'CONDITIONS_COMPLETE.json'),
                           (coord.ROOT/'targets', 'TARGETS_COMPLETE.json')):
            sources.add(base/phase/name)
    for item in data['sources']:
        path = Path(item['path'])
        if c.sha256(path) != item['sha256']:
            raise ValueError('data qualification source receipt changed')
        sources.add(path)
    for phase in c.config()['approval']['hpss_b_restore_phases']:
        path = c.ROOT / 'particle_b' / phase / 'TRANSFER_COMPLETE.json'
        transferred = json.loads(path.read_text())
        if (transferred['phase'] != phase or transferred['config_sha256'] != c.sha256(c.CONFIG)
                or transferred['payload_bytes'] <= 0):
            raise ValueError('B restore provenance mismatch')
        # Transfer receipts deliberately do NOT claim CRC/scientific QA. Those
        # subsequent checks are bound by each independent phase audit above.
        sources.add(path)
    for item in normalizer['sources']:
        path = Path(item['path'])
        if c.sha256(path) != item['sha256']:
            raise ValueError('normalization source binding changed')
        sources.add(path)
    reports = {}
    for phase in c.TRAIN:
        marker = normalizer_path.parent / f'{phase}_MOMENTS.json'
        if not marker.is_file():
            raise FileNotFoundError('cached moments required; closeout never rebuilds payload statistics')
        reports[phase], _ = norm.phase_statistics(phase, normalizer_path.parent, norm.source_binding())
    if norm.equal_phase_statistics(reports) != normalizer['normalization']:
        raise ValueError('cached equal-phase statistics do not reproduce normalizer')
    physical_path = coord.ROOT / 'physical_gate/PHYSICAL_GATE_COMPLETE.json'
    physical = coord.verify_receipt(physical_path, payload=False)
    expected_physics = dict(config_sha256=c.sha256(physics.CONFIG),
        layout_sha256=c.sha256(physics.op.LAYOUT), operator_sha256=c.sha256(physics.op.__file__),
        builder_sha256=c.sha256(physics.__file__))
    if physical['binding'] != expected_physics:
        raise ValueError('physical reference implementation drift')
    for item in physical['sources']:
        if c.sha256(item['path']) != item['sha256']:
            raise ValueError('physical reference input receipt drift')
    cfg = json.loads(physics.CONFIG.read_text())
    release.require_physical_panel(physical, cfg)
    if physics.decide(physical['cases'], cfg) != physical['decision'] or not physical['decision']['pass']:
        raise ValueError('physical representation decision mismatch')
    sources.add(physical_path)
    technical_path = coord.ROOT/'technical_gpu'/c.digest(gpu.binding())[:16]/'GPU_BENCHMARK_COMPLETE.json'
    technical = coord.verify_receipt(technical_path)
    require_no_scientific_authority(data, technical)
    if technical['binding'] != gpu.binding():
        raise ValueError('technical GPU implementation changed')
    sources.add(technical_path)
    for item in technical['outputs']:
        case_path = Path(item['path']); case = coord.verify_receipt(case_path)
        sources.add(case_path)
        sources.update(Path(row['path']) for row in case['outputs'])
    loader_path = coord.ROOT/'technical_cpu/normalized_loader/LOADER_BENCHMARK_COMPLETE.json'
    loaded = coord.verify_receipt(loader_path, payload=False)
    if loaded['binding'] != loader.binding(normalizer_path):
        raise ValueError('actual loader measurement changed')
    for item in loaded['sources']:
        if c.sha256(item['path']) != item['sha256']:
            raise ValueError('loader input receipt changed')
    sources.add(loader_path)
    post_path = coord.ROOT/'technical_cpu/postprocess'/c.digest(post.binding())[:16]/'POSTPROCESS_BENCHMARK_COMPLETE.json'
    sources.add(post_path)
    projected = cost.run([16,32,64], [128,256], 1664)
    sources.add(Path(projected['path']))
    usage = accounting(); size = used_bytes(c.ROOT)
    require_terminal_budget(usage, size, c.config()['approval'])
    proposal = PROPOSAL.read_text()
    if 'NOT approval-ready' in proposal or '260 allocated GPU-hours' not in proposal:
        raise ValueError('final measured proposal must be reviewed and ready for separate approval')
    result = dict(**coord.provenance(), preparation_requirements_pass=True,
        data_products_qualified=True, scientific_training_authorized=False,
        posterior_calibration_claim=False, proposal_ready_for_user_review=True,
        git_commit_verification_is_separate=True, phases=21, training_pairs=1664,
        development_pairs=32, confirmation_pairs=96, physical_cases=224,
        normalized_offset_cases=11776, technical_factor_cases=7,
        normalization_recomputed_from_cached_train_moments=True,
        receipt_bindings_and_audited_file_metadata_rechecked=True, payload_crc_rescanned=False,
        accounting=usage, new_scratch_bytes=size, approval=c.config()['approval'],
        proposal=c.file_record(PROPOSAL, content_hash=True),
        cost_projection=c.file_record(projected['path'], content_hash=True),
        claim_boundaries=data['claim_boundaries'], **{'pass': True})
    return result, sorted(sources)


def archive(result, sources, destination=DESTINATION):
    if destination.exists():
        raise FileExistsError('preserve the existing closeout evidence; do not overwrite it')
    if any(path.suffix != '.json' or not path.resolve().is_relative_to(c.ROOT) for path in sources):
        raise PermissionError('only bounded preparation JSON receipts may be archived')
    if sum(path.stat().st_size for path in sources) > 32*1024**2:
        raise ValueError('small evidence archive exceeds32MiB; review instead of copying bulk data')
    records = []
    for source in sources:
        relative = source.relative_to(c.ROOT)
        target = destination/'receipts'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = c.file_record(source, content_hash=True)
        shutil.copyfile(source, target)
        if c.sha256(target) != expected['sha256']:
            raise ValueError('archival copy mismatch')
        records.append(dict(source=expected, relative=str(target.relative_to(destination))))
    c.atomic_json(destination/'CLOSEOUT.json', result)
    c.atomic_json(destination/'MANIFEST.json', dict(files=records,
        closeout_sha256=c.sha256(destination/'CLOSEOUT.json'), bulk_payloads_copied=False,
        scientific_training_authorized=False))
    return dict(directory=str(destination), receipts=len(records), **{'pass': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    result, sources = collect()
    print(json.dumps(archive(result, sources) if args.archive else result, indent=2))
