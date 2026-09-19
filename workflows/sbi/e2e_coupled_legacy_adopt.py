"""Adoption-only recovery: reuse strictly checked CRC receipts, never redeposit."""
import argparse
import json
from pathlib import Path
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_matter as matter
from workflows.sbi import e2e_coupled_legacy_native as legacy


def checked_particles(phase,directory):
    if phase not in ('ph002','ph003'): raise PermissionError('counted legacy phases only')
    record=c.verify_receipt(directory/'PARTICLES_VERIFIED.json',payload=False)
    a,b,manifest=legacy.legacy_paths(phase)
    original=c.ROOT/'source_snapshots/legacy_native_v1/workflows/sbi/e2e_coupled_legacy_native.py'
    expected_binding=dict(builder_sha256=c.sha256(original),verifier_sha256=c.sha256(matter.__file__),
                          legacy_manifest_sha256=c.sha256(manifest),a_root=str(a),b_root=str(b))
    if record['phase']!=phase or record['binding']!=expected_binding or len(record['files'])!=136:
        raise ValueError('not the verified original legacy-particle receipt')
    expected={}
    for sample,root in (('A',a),('B',b)):
        for kind in ('field','halo'):
            folder=root/f'{kind}_rv_{sample}'; crc=folder/'checksums.crc32'
            source=next(v for v in record['crc_manifests'] if Path(v['path'])==crc)
            if c.sha256(crc)!=source['sha256']: raise ValueError('particle CRC manifest drift')
            for name,(checksum,size) in matter.crc_manifest(crc).items():
                expected[str((folder/name).resolve())]=(checksum,size,sample)
    if {v['path'] for v in record['files']}!=set(expected): raise ValueError('particle inventory changed')
    for item in record['files']:
        if c.file_record(item['path'])!={k:item[k] for k in ('path','bytes','mtime_ns')}:
            raise ValueError('CRC-verified particle bytes/mtime changed')
        if (item['crc'],item['bytes'],item['sample'])!=expected[item['path']]:
            raise ValueError('particle CRC receipt disagrees with source manifest')
        if item['header']['SimName']!=f'AbacusSummit_base_c000_{phase}' or item['particle_count']<0:
            raise ValueError('particle phase/count mismatch')
    if record['particle_count']!=sum(v['particle_count'] for v in record['files']):
        raise ValueError('particle receipt count disagreement')
    return record


def run(phase):
    c.require_compute(); c.phase_guard(phase)
    directory=c.ROOT/'matter'/phase
    with c.single_writer(directory):
        final=directory/'DENSITY_COMPLETE.json'
        if final.exists(): return c.verify_receipt(final)
        c.verify_receipt(c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json',payload=False)
        particles=checked_particles(phase,directory)
        return legacy.adopt(phase,directory,particles)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phases',nargs='+',default=['ph002','ph003'],choices=['ph002','ph003'])
    a=p.parse_args()
    for phase in a.phases:
        result=run(phase)
        print(json.dumps({k:v for k,v in result.items() if k not in ('binding','outputs')}),flush=True)
