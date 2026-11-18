"""Verify terminal low-z preparation receipts without promoting training readiness."""
from __future__ import annotations

import argparse
from pathlib import Path

from workflows.sbi import prepare_p12a_lowz_inputs as p12
from workflows.sbi import cfm_lowz_selection_response as cfm
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def summarize(output, metadata_only=False):
    if not metadata_only:
        require_compute()
    pplan, _, policy = p12.load_plan()
    cplan, cpolicy, _ = cfm.load_plan()
    if policy != cpolicy:
        raise ValueError('Cross-project phase policy mismatch')
    phases = policy['training'] + policy['development_and_calibration']
    proot, croot = Path(pplan['output_root']), Path(cplan['output_root'])
    raw = p12.read(proot/'CAMPAIGN_RAW_COMPLETE.json')
    geometry = p12.read(proot/'GEOMETRY_RESPONSES_COMPLETE.json')
    if raw['phases'] != phases or geometry['phases'] != phases:
        raise ValueError('P12 inventory differs')
    supervised = lowz = parent_rows = support_rows = 0
    for phase, row in zip(phases, geometry['records']):
        if row['phase'] != phase or not row['geometry_features_ready'] or row['model_inputs_ready']:
            raise ValueError('P12 geometry status differs')
        if not metadata_only:
            for key, path in row['arrays'].items():
                if sha256(path) != row['array_sha256'][key]:
                    raise ValueError('P12 geometry payload hash changed')
        phase_raw = p12.read(proot/phase/'RAW_PREPARATION_COMPLETE.json')
        supervised += phase_raw['index_audit']['active']
        lowz += phase_raw['index_audit']['lowz_below_0p1']
        parent_rows += row['rows']
        support_rows += row['support_rows']
    complete_path = croot/'run_60477149/PREPARATION_COMPLETE.json'
    complete = p12.read(complete_path)
    if complete['binding']['phases'] != phases:
        raise ValueError('CFM inventory differs')
    for path, expected in complete['outputs'].items():
        if sha256(path) != expected:
            raise ValueError('CFM terminal receipt binding changed')
    for omitted in [None]+policy['training']:
        fit = p12.read(croot/('full' if omitted is None else 'oof_'+omitted)/'SELECTION.json')
        if fit['fit_phases'] != [p for p in policy['training'] if p != omitted]:
            raise ValueError('CFM fit phase leak')
    payload_path = croot/'PAYLOAD_AUDIT.json'
    payload = (p12.read(payload_path) if payload_path.exists() else
               dict(numerical_identity_pass=False, all_payload_hashes_verified=False,
                    status='incomplete: allocation shell auto-logout interrupted audit'))
    closure = p12.read(croot/'INDEPENDENT_CLOSURE_DIAGNOSTICS.json')
    if not metadata_only and (set(payload.get('phases', {})) != set(phases) or not payload['numerical_identity_pass']):
        raise ValueError('CFM payload audit incomplete')
    volumes = {phase:p12.read(croot/('LOWZ_VOLUME_AUDIT_'+phase+'.json')) for phase in ('ph001','ph006')}
    closure_extrema = {}
    for role in ('train_oof', 'development'):
        entries = [(abs(row['expected_over_observed_minus_one']), phase, cap, row)
                   for phase, data in closure['records'].items() if data['role'] == role
                   for cap, rows in data['caps'].items() for row in rows
                   if row['expected_over_observed_minus_one'] is not None]
        value, phase, cap, row = max(entries, key=lambda item:item[0])
        closure_extrema[role] = dict(max_abs_fractional_residual=value, phase=phase, cap=cap, bin=row)
    receipts = [proot/'CAMPAIGN_RAW_COMPLETE.json', proot/'GEOMETRY_RESPONSES_COMPLETE.json',
                complete_path, croot/'INDEPENDENT_CLOSURE_DIAGNOSTICS.json']
    if payload_path.exists():
        receipts.append(payload_path)
    receipts += [croot/('LOWZ_VOLUME_AUDIT_'+p+'.json') for p in volumes]
    save_json(output, dict(phases=phases, scope=[.01,.55], model_inputs_ready=False,
        summary_metadata_only=metadata_only,
        p12=dict(raw_phases=len(phases), geometry_response_phases=len(geometry['records']),
                 supervised_rows=supervised, supervised_z_lt_0p1_rows=lowz,
                 parent_rows=parent_rows, supported_parent_rows=support_rows,
                 geometry_payload_hashes_verified=not metadata_only, radial_selection_fitted=False,
                 host_fold_policy_qualified=False),
        cfm=dict(statistics_phases=len(phases), full_selection_fits=1, oof_selection_fits=20,
                 response_phases=len(phases), response_hdf5_files=2*len(phases),
                 payload_audit=payload, lowz_volume_screens=volumes,
                 independent_closure_extrema=closure_extrema),
        receipts={str(p):sha256(p) for p in receipts},
        remaining=['P12 radial selection/full-OOF transforms and qualified fold/loader integration',
                   'CFM selection/volume qualification, domain allocation, matched targets and normalizers/loaders']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--metadata-only', action='store_true', help='Read bounded JSON receipts only; no payload verification claim')
    a = p.parse_args()
    summarize(a.output, a.metadata_only)
