"""Complete particle provenance for reused native fields; never redeposit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_matter as matter
from workflows.sbi.prepare_shared_phase_sources import validate_native_manifest


def prepare(phase):
    c.require_compute();c.phase_guard(phase)
    if not c.SHARED_POLICY:raise ValueError('New campaign required')
    directory=c.ROOT/'matter'/phase
    with c.single_writer(directory):
        final=directory/'PARTICLES_VERIFIED.json'
        if final.exists():return c.verify_receipt(final,payload=False)
        native=c.verify_receipt(directory/'DENSITY_COMPLETE.json',payload=False)
        start=time.monotonic()
        if 'reuse' in native:
            prior=Path(c.config()['reuse_source_root'])/'matter'/phase/'PARTICLES_VERIFIED.json'
            value=json.loads(prior.read_text())
            if value['phase']!=phase or not value['pass'] or value['config_sha256']!=c.sha256(c.REPO/'configs/e2e_coupled_data_v1.json'):
                raise ValueError('Historical particle inventory authority mismatch')
            for item in value['crc_manifests']:
                if c.sha256(item['path'])!=item['sha256']:raise ValueError('CRC manifest changed')
            for item in value['files']:
                if c.file_record(item['path'])!={k:item[k] for k in ('path','bytes','mtime_ns')}:raise ValueError('Particle metadata changed')
            result=dict(value,**{})
            result.update(c.provenance())
            result['reuse']=dict(source_receipt=c.file_record(prior,content_hash=True),scope='Prior complete CRC/header verification, current manifest hashes and all particle metadata checks')
        else:
            item=native['source_manifest'];path=Path(item['path'])
            if c.sha256(path)!=item['sha256']:raise ValueError('P10 density source manifest drift')
            value=json.loads(path.read_text());build=validate_native_manifest(value,phase)
            files={str(Path(p).resolve()) for p in build['processed_files']}
            jobs=[];manifests=[]
            for name,info in value['inputs']['directories'].items():
                kind,sample=name.split('_');manifest=Path(info['checksum_manifest'])
                expected=matter.crc_manifest(manifest)
                names={f'{kind}_rv_{sample}_{slab:03d}.asdf' for slab in range(34)}
                if set(expected)!=names or len(info['files'])!=34:raise ValueError('Particle inventory incomplete')
                manifests.append(c.file_record(manifest,content_hash=True))
                for filename in info['files']:
                    p=Path(filename)
                    if str(p.resolve()) not in files or p.name not in names:raise ValueError('Input/build particle list differs')
                    jobs.append((p,phase,sample,expected[p.name]))
            if len(jobs)!=136 or len({str(j[0].resolve()) for j in jobs})!=136 or len(manifests)!=4:
                raise ValueError('A+B particle completeness failure')
            print(json.dumps(dict(phase=phase,crc_files=136,bytes=sum(j[3][1] for j in jobs))),flush=True)
            with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(matter.verify_particle_file,jobs))
            result=dict(**c.provenance(),phase=phase,files=records,crc_manifests=manifests,
                source_code_sha256=c.sha256(matter.__file__),source_manifest=item,
                elapsed_seconds=time.monotonic()-start,fresh_crc_scan=True,**{'pass':True})
        c.atomic_json(final,result)
    print(json.dumps(dict(phase=phase,particle_inventory_complete=True,seconds=time.monotonic()-start)),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phases',nargs='+',required=True);a=p.parse_args()
    for phase in a.phases:prepare(phase)
