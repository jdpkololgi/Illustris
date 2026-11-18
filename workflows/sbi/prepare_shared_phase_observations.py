"""Revalidate existing target-free source matches and fixed coordinates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import fitsio
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_pairing as pairing
from workflows.sbi.e2e_coupled_coordinate_verify import verify as verify_coordinates


def campaign():
    c.require_compute()
    if not c.SHARED_POLICY:
        raise ValueError('Explicit shared campaign required')
    c.bind_run()
    return json.loads((c.REPO/'configs/phase_migration_20261027.json').read_text())


def coordinate_evidence():
    cfg=campaign(); old=Path(cfg['old_cfm_root'])
    for phase in coord.config()['host_check_phases']:
        if phase not in c.TRAIN:
            raise ValueError('Coordinate references must remain training phases')
        original=old/'coordinate_audit'/phase/'AUDIT.json'
        corrected=old/'cartesian_v2/coordinate_audit'/phase/'CORRECTED_COMPLETE.json'
        report=json.loads(corrected.read_text())
        if (report.get('pass') is not True or report.get('phase')!=phase
                or report['coordinate_sha256']!=c.sha256(coord.CONFIG)
                or report['old_audit_sha256']!=c.sha256(original)):
            raise ValueError('Historical coordinate evidence changed')
        for host in report['host_sources']:
            item=host['source']; stat=Path(item['path']).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(item['bytes'],item['mtime_ns']):
                raise ValueError('Coordinate reference source changed')
        for item in report['outputs']:
            if c.sha256(item['path'])!=item['sha256']:
                raise ValueError('Linked coordinate samples changed')
        destination=c.ROOT/'coordinate_audit'/phase/'AUDIT.json'
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if c.sha256(destination)!=c.sha256(original):raise ValueError('Existing audit differs')
        else:
            shutil.copy2(original,destination)
        # Recompute all residuals and wrong-unit controls under the new authority.
        value=verify_coordinates(phase)
        print(json.dumps({'phase':phase,'coordinate_pass':value['pass'],
                          'corrected':value['variants_mpc_h']['corrected']}),flush=True)


def validate_parent_record(record, phase, sources, columns):
    if (record.get('phase')!=phase or record.get('target_truth_present') is not False
            or record.get('parity',{}).get('row_identity_verified') is not True):
        raise ValueError('Unqualified or target-bearing parent')
    if any(name=='CWEB' or name.startswith(('LAMBDA','TWEB','HESSIAN')) for name in columns):
        raise ValueError('Tidal columns in target-free parent')
    for name in ('cutsky','forfa'):
        if Path(record[name]['path']).resolve()!=Path(sources[name]).resolve():
            raise ValueError('Parent source differs from current registered phase')


def observations(phase):
    cfg=campaign(); c.phase_guard(phase)
    if phase not in cfg['cfm_missing_phases']:
        raise ValueError('This entrypoint prepares missing phase observations only')
    if phase=='ph001':
        folder=Path(cfg['new_p12a_root'])/phase/'catalogues'
    else:
        p1=json.loads((Path(cfg['old_p12a_root'])/phase/'p1_canonical/manifest.json').read_text())
        folder=Path(p1['parent']).parent
    source_marker=folder/f'{phase}_bright_parent_metadata_v2.fits.complete.json'
    record=json.loads(source_marker.read_text()); source=Path(record['output']['path'])
    sources=c.paths(phase)
    with fitsio.FITS(source) as f: columns=f[1].get_colnames(); rows=f[1].get_nrows()
    validate_parent_record(record,phase,sources,columns)
    if rows!=record['output']['rows'] or c.sha256(source)!=record['output']['sha256']:
        raise ValueError('Existing target-free parent payload differs')
    parity=pairing.parent.validate_parent_against_forfa(source,sources['forfa'],rows,500_000)
    directory=c.ROOT/'observations'/phase
    with c.single_writer(directory):
        marker=directory/'PARENT_COMPLETE.json'
        if not marker.exists():
            value=dict(**c.provenance(),schema='e2e-coupled-parent-reuse-v1',phase=phase,role=c.ROLES[phase],
                       outputs=[c.file_record(source,content_hash=True)],parity=parity,
                       original_parent_receipt=c.file_record(source_marker,content_hash=True),
                       target_columns_present=False,reuse_scope='P10 target-free parent; full current forFA parity revalidated',
                       **{'pass':True})
            c.atomic_json(marker,value)
    result=pairing.prepare_phase(phase)
    print(json.dumps({'phase':phase,'observed_rows':result['observed_rows'],'pass':result['pass']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['coordinates','observations'])
    p.add_argument('--phases',nargs='+',default=['ph001','ph004','ph005','ph006'])
    args=p.parse_args()
    if args.action=='coordinates':coordinate_evidence()
    else:
        for phase in args.phases:observations(phase)

if __name__=='__main__':main()
