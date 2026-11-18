"""Versioned low-z campaign adapters and support-only final pair geometry.

Preserve native matter, coordinate checks and observation identities with fresh
payload hashes and explicit lineage. Never adopt old geometry or fitted charts.
Matched responses receive a new, explicit schema; no legacy arithmetic claim.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import h5py
import numpy as np

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi import prepare_cfm_lowz_geometry_candidates as candidates


def authority():
    cfg = c.config()
    if not c.SHARED_POLICY or not cfg.get('lowz'):
        raise ValueError('Explicit low-z campaign configuration required')
    link = cfg['lowz']
    source = c.REPO/link['source_config']
    if c.sha256(source) != link['source_config_sha256']:
        raise ValueError('Source campaign changed')
    old = json.loads(source.read_text())
    if (old['target'] != cfg['target'] or old['geometry'] != cfg['geometry']
            or old['phase_roles'] != cfg['phase_roles']
            or old['root'] != cfg['reuse_source_root']):
        raise ValueError('Unapproved target, geometry or phase-role change')
    if (cfg['observation']['science_shell_edges'] != [.01,.25,.35,.45,.55]
            or cfg['observation']['context_redshift'] != [0.,.6]):
        raise ValueError('Wrong low-z domain')
    return cfg, old


def checked_source(path, phase, source_hash, payload=True):
    record = json.loads(Path(path).read_text())
    if record.get('config_sha256') != source_hash or not record.get('pass') or record.get('phase') != phase:
        raise ValueError('Source authority/identity failure: '+str(path))
    if payload:
        for item in record.get('outputs', []):
            p = c.guarded(item['path'], phase)
            if p.stat().st_size != item['bytes'] or c.sha256(p) != item['sha256']:
                raise ValueError('Source payload changed: '+str(p))
    return record


def adopt_one(relative, phase):
    cfg, old = authority()
    source = Path(old['root'])/relative
    record = checked_source(source, phase, cfg['lowz']['source_config_sha256'])
    value = copy.deepcopy(record)
    value.update(c.provenance())
    if 'role' in value:
        value['role'] = c.ROLES[phase]
    value['lowz_adoption'] = dict(source_receipt=c.file_record(source, content_hash=True),
        source_config_sha256=cfg['lowz']['source_config_sha256'],
        payloads_rehashed=True, raw_source_validation_inherited=True,
        adopter_sha256=c.sha256(__file__))
    if 'parent_receipt_sha256' in value:
        old_parent = Path(old['root'])/'observations'/phase/'PARENT_COMPLETE.json'
        if value['parent_receipt_sha256'] != c.sha256(old_parent):
            raise ValueError('Old observed-parent lineage changed')
        value['parent_receipt_sha256'] = c.sha256(c.ROOT/'observations'/phase/'PARENT_COMPLETE.json')
    dest = c.ROOT/relative
    if dest.exists():
        existing = c.verify_receipt(dest)
        if existing['lowz_adoption'] != value['lowz_adoption']:
            raise ValueError('Conflicting low-z adoption')
    else:
        c.atomic_json(dest, value)
    return c.file_record(dest, content_hash=True)


def adopt_checks():
    c.require_compute(); authority(); coord.bind()
    for phase in coord.config()['host_check_phases']:
        c.phase_guard(phase)
        adopt_one(f'cartesian_v2/coordinate_audit/{phase}/CORRECTED_COMPLETE.json', phase)
    coord.require_host_checks()


def adopt_base(phase):
    c.require_compute(); c.phase_guard(phase); authority(); coord.require_host_checks()
    directory = c.ROOT/'input_adoption'/phase
    with c.single_writer(directory):
        receipts = [adopt_one(relative, phase) for relative in (
            f'matter/{phase}/PARTICLES_VERIFIED.json', f'matter/{phase}/DENSITY_COMPLETE.json',
            f'observations/{phase}/PARENT_COMPLETE.json', f'observations/{phase}/OBSERVED_COMPLETE.json',
            f'observations/{phase}/angular/ANGULAR_COMPLETE.json')]
        marker = directory/'BASE_INPUTS_COMPLETE.json'
        if not marker.exists():
            c.atomic_json(marker, dict(**c.provenance(), phase=phase, receipts=receipts,
                outputs=[], geometry_and_targets_adopted=False, **{'pass':True}))


def adopt_responses(phase):
    c.require_compute(); c.phase_guard(phase); cfg,_ = authority()
    root = Path(cfg['lowz']['response_root'])
    selection = Path(cfg['observation']['fixed_selection_manifest'])
    selected = json.loads(selection.read_text())
    if selected['fit_phases'] != list(c.TRAIN) or selected['omitted_phase'] is not None:
        raise PermissionError('Wrong selection fit phases')
    for cap in lowz.CAPS:
        source = root/cfg['lowz']['matched_response_version']/phase/(cap+'_COMPLETE.json')
        record = json.loads(source.read_text())
        if (record['phase'] != phase or record['cap'] != cap or not record['pass']
                or record['contains_matter'] or record['binding']['selection_sha256'] != c.sha256(selection)):
            raise ValueError('Invalid matched response')
        for item in (record['output'], record['external_source']):
            if c.sha256(item['path']) != item['sha256']:
                raise ValueError('Matched response or external geometry changed')
        grid = dict(record['grid'], distance_unit='Mpc/h', padding_mpc_h=0.,
                    coordinate_sha256=c.sha256(coord.CONFIG))
        coord.validate_grid(grid)
        value = dict(**coord.provenance(), schema='cfm-lowz-adopted-matched-response-v1',
            phase=phase, role=c.ROLES[phase], cap=cap, grid=grid,
            outputs=[c.file_record(record['output']['path'], content_hash=True)],
            matched_source=c.file_record(source, content_hash=True), external_source=record['external_source'],
            source_code_sha256=c.sha256(__file__), selection=dict(c.file_record(selection, content_hash=True),
                fit_phases=list(c.TRAIN)), gates=dict(source_point_cic_conservation=True,
                same_count_expectation_operator=True), contains_matter=False, **{'pass':True})
        dest = coord.ROOT/'observations'/phase/(cap+'_COMPLETE.json')
        if dest.exists():
            prior = coord.verify_receipt(dest)
            if prior['matched_source'] != value['matched_source']:
                raise ValueError('Conflicting response adoption')
        else:
            c.atomic_json(dest, value)


def select_panel(phase, buckets):
    """Retain all original quotas and exact periodic ownership constraints."""
    quota = 8 if c.ROLES[phase] == 'train' else 1
    cursors = Counter(); selected=[]; mids=[]; cores=[]
    for ordinal in range(quota):
        for key, bucket in buckets.items():
            accepted = None
            while cursors[key] < len(bucket):
                row = bucket[cursors[key]]; cursors[key] += 1
                midpoint = np.asarray(row['source_midpoint_mpc_h'])
                pair = np.asarray(row['source_owned_core_centers_mpc_h'])
                if geometry.admissible(midpoint,pair,mids,cores):
                    accepted = dict(row,pair_id=f'{phase}_{key[0]}_s{key[1]}_{key[2]}_{ordinal:02d}')
                    mids.append(midpoint); cores.extend(pair); break
            if accepted is None:
                raise ValueError(f'Insufficient nonoverlapping low-z panel: {phase} {key}')
            selected.append(accepted)
    return selected


def verify_lowz_rows(rows, phase):
    qa = geometry.verify_rows(rows, phase)
    for row in rows:
        z = row['redshift']
        shell = int(np.searchsorted([.01,.25,.35,.45,.55],z,side='right')-1)
        if shell != row['shell'] or not 0 <= shell < 4:
            raise ValueError('Low-z row outside declared shell')
    for cap in lowz.CAPS:
        if not any(r['cap']==cap and r['redshift']<.15 for r in rows):
            raise ValueError('Low-z coverage missing from '+cap)
    return dict(qa, lowz_midpoints_in_both_caps=True)


def build_geometry(phase):
    c.require_compute(); c.phase_guard(phase); authority(); coord.require_host_checks()
    directory = coord.ROOT/'geometry'/phase
    with c.single_writer(directory):
        marker = directory/'GEOMETRY_COMPLETE.json'
        if marker.exists():
            result = coord.verify_receipt(marker, payload=False)
            verify_lowz_rows(result['pairs'], phase)
            return result
        angular = c.verify_receipt(c.ROOT/'observations'/phase/'angular/ANGULAR_COMPLETE.json')
        with np.load(angular['outputs'][0]['path'],allow_pickle=False) as saved:
            angular_data = {k:saved[k] for k in ('support','domain')}
        buckets={}; sources={}; diagnostics={}
        for cap in ('NGC','SGC'):
            path = coord.ROOT/'observations'/phase/(cap+'_COMPLETE.json')
            record = coord.verify_receipt(path)
            regular, standard_qa = geometry.candidate_buckets(phase,cap,record)
            extra, lowz_qa = candidates.candidates(phase,cap,record['outputs'][0],angular_data)
            for kind in ('interior','boundary'):
                low = sorted((dict(r,role=c.ROLES[phase],shell=0) for r in extra
                              if r['support_stratum']==kind), key=lambda r:(r['redshift'],tuple(r['center'])))
                regular[0,kind] = low + regular[0,kind]
            buckets.update({(cap,*key):value for key,value in regular.items()})
            diagnostics[cap] = dict(standard=standard_qa,lowz=lowz_qa)
            sources[cap] = dict(receipt=str(path),sha256=c.sha256(path))
        selected = select_panel(phase,buckets)
        qa = verify_lowz_rows(selected,phase)
        result = dict(**coord.provenance(),phase=phase,role=c.ROLES[phase],sources=sources,
            source_code_sha256=c.sha256(__file__),candidate_code_sha256=c.sha256(candidates.__file__),
            pairs=selected,diagnostics=diagnostics,qa=qa,counts_or_targets_read=False,
            shell_edges=[.01,.25,.35,.45,.55],model_inputs_ready=False,**{'pass':True})
        c.atomic_json(marker,result)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('checks','base','responses','geometry'))
    parser.add_argument('--phases',nargs='+',default=[])
    args=parser.parse_args()
    if args.action=='checks': adopt_checks()
    else:
        for phase in args.phases:
            {'base':adopt_base,'responses':adopt_responses,'geometry':build_geometry}[args.action](phase)
