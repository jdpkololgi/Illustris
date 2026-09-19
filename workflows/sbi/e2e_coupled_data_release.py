"""All-panel data qualification index, never scientific training authority.

Requires the existing independent phase audits and the completed normalized
interface test. This indexes their evidence; it does not replace a payload audit
with an existence check, rerun particle CRCs, or certify posterior calibration.
The technical GPU/resource-proposal closeout remains a separate requirement.
"""
import json
from pathlib import Path

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_audit_worker as queue
from workflows.sbi import e2e_coupled_product_audit as audit
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_coupled_normalization as norm
from workflows.sbi import e2e_coupled_interface_qualify as interface
from workflows.sbi import e2e_coupled_physical_gate as physical


def require_panel(phases, normalizer, qualified_interface):
    if set(phases) != set(c.ROLES):
        raise ValueError('all21 authorized phases are required; no partial data release')
    for phase, row in phases.items():
        expected = 128 if phase in c.TRAIN else 16
        if row['role'] != c.ROLES[phase] or row['pairs'] != expected or row['pass'] is not True:
            raise ValueError('phase role, pair quota or qualification failure')
    if (normalizer['fit_phases'] != list(c.TRAIN) or normalizer['phase_weight'] != 1/13
            or normalizer['pass'] is not True):
        raise ValueError('complete equal-phase train-only normalization required')
    if (qualified_interface['phase_roles'] != c.ROLES or qualified_interface['pairs'] != 1792
            or qualified_interface['offset_cases'] != 11776
            or qualified_interface['pass'] is not True
            or qualified_interface['science_scores_evaluated'] is not False):
        raise ValueError('complete numerical normalized interface qualification required')


def audit_dependencies():
    return {Path(module.__file__).name: c.sha256(module.__file__) for module in (
        c, coord, audit.reader, audit.geometry, audit.matter, audit.op, audit.targets,
        audit.conditions_builder, audit.observations_builder, views)}


def require_physical_panel(record, cfg):
    expected = {(phase, cap, shell, kind, tuple(offset))
                for phase in cfg['phases'] for cap in ('NGC', 'SGC') for shell in range(4)
                for kind in ('interior', 'boundary') for offset in audit.op.layout()['context_offsets_raw']}
    actual = [(row['phase'], row['cap'], row['shell'], row['support_stratum'], tuple(row['offset_raw']))
              for row in record['cases']]
    if (set(actual) != expected or len(actual) != len(expected)
            or record['no_posterior_or_predictive_scoring'] is not True):
        raise ValueError('physical-reference primary/translation panel is incomplete or duplicated')


def phase_evidence(phase):
    c.phase_guard(phase)
    if not queue.qualified(phase):
        raise FileNotFoundError('independent full-payload audit missing: ' + phase)
    pointer_path = coord.ROOT / 'product_audit' / phase / 'LATEST_AUDIT.json'
    pointer = coord.verify_receipt(pointer_path, payload=False)
    report_path = Path(pointer['audit']['path'])
    report = coord.verify_receipt(report_path, payload=False)
    if (report['audit_dependencies'] != audit_dependencies()
            or report['science_scores_evaluated'] is not False
            or report['phase'] != phase or report['role'] != c.ROLES[phase]):
        raise ValueError('phase audit dependency/identity/role drift')
    # The independent audit already read these actual SHA256s. Check their
    # path/size/mtime again at publication; do not mislabel this as a new hash
    # or particle-CRC scan. The normalized interface separately rehashes every
    # condition/target shard. No source or committed payload is rewritten here.
    for item in report['actual_sha256_files']:
        path = c.guarded(item['path'], phase)
        current = c.file_record(path)
        if any(current[key] != item[key] for key in ('path', 'bytes', 'mtime_ns')):
            raise ValueError('audited source/payload path or metadata changed: ' + str(path))
    marker = coord.ROOT / 'interface_qualification' / phase / 'INTERFACE_PHASE_COMPLETE.json'
    checked = coord.verify_receipt(marker, payload=False)
    if checked['phase'] != phase or checked['role'] != c.ROLES[phase]:
        raise ValueError('normalized phase interface identity mismatch')
    return dict(role=c.ROLES[phase], pairs=len(report['pairs']), **{'pass': True},
                phase_audit=c.file_record(report_path, content_hash=True),
                audit_pointer=c.file_record(pointer_path, content_hash=True),
                normalized_interface=c.file_record(marker, content_hash=True),
                files_checked_for_metadata_drift=len(report['actual_sha256_files']))


def run():
    c.require_compute(); c.config(); coord.require_host_checks()
    # Check prerequisites before invoking the idempotent verifiers. This
    # publisher must not silently substitute itself for an unfinished build.
    required = [coord.ROOT / 'normalization/NORMALIZATION_COMPLETE.json',
                coord.ROOT / 'interface_qualification/INTERFACE_COMPLETE.json',
                coord.ROOT / 'physical_gate/PHYSICAL_GATE_COMPLETE.json']
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('data qualification prerequisites incomplete: ' + ', '.join(missing))
    normalizer = norm.fit()
    interfaces = interface.run()
    physics = physical.run()
    cfg = json.loads(physical.CONFIG.read_text())
    require_physical_panel(physics, cfg)
    if (physics['pass'] is not True or physical.decide(physics['cases'], cfg) != physics['decision']
            or not physics['decision']['pass'] or len(physics['cases']) != 224):
        raise ValueError('registered physical-reference gate is incomplete or failed')
    phases = {phase: phase_evidence(phase) for phase in c.ROLES}
    require_panel(phases, normalizer, interfaces)
    binding = dict(publisher_sha256=c.sha256(__file__), phase_roles=c.ROLES,
                   layout_sha256=c.sha256(audit.op.LAYOUT), audit_dependencies=audit_dependencies(),
                   normalization_sha256=c.sha256(required[0]), interface_sha256=c.sha256(required[1]),
                   physical_gate_sha256=c.sha256(required[2]))
    sources = [c.file_record(path, content_hash=True) for path in required]
    sources.extend(c.file_record(coord.ROOT / 'coordinate_audit' / phase / 'CORRECTED_COMPLETE.json',
                                 content_hash=True) for phase in coord.config()['host_check_phases'])
    result = dict(**coord.provenance(), binding=binding, phases=phases, sources=sources,
        training_pairs=1664, development_pairs=32, confirmation_pairs=96,
        data_products_qualified=True, scientific_training_authorized=False,
        posterior_calibration_claim=False, full_preparation_complete=False,
        technical_gpu_and_resource_proposal_checked=False,
        source_verification='Independent phase payload hashes plus unchanged source metadata at publication; normalized interface rehashes all pair payloads',
        claim_boundaries=['Fixed-epoch z0.2 c000 matter, not an evolving matter lightcone',
            'Identified nominal paired mock-release branch, not recovered numerical HOD parameters',
            'No HOD marginalization or demonstrated real-DESI likelihood validity',
            'Adjacent-pair products do not establish global coherence across separately generated domains'],
        outputs=[], **{'pass': True})
    directory = coord.ROOT / 'data_release'
    with c.single_writer(directory):
        marker = directory / 'DATA_PRODUCTS_QUALIFIED.json'
        if marker.exists():
            previous = coord.verify_receipt(marker, payload=False)
            if any(previous[key] != result[key] for key in ('binding', 'phases', 'sources')):
                raise ValueError('qualified data release changed')
            return previous
        c.atomic_json(marker, result)
    return result


if __name__ == '__main__':
    result = run()
    print(json.dumps(dict(data_products_qualified=result['data_products_qualified'],
                         scientific_training_authorized=False, full_preparation_complete=False,
                         phases=len(result['phases']))), flush=True)
