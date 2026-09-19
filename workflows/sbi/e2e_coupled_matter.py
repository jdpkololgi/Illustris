"""CRC/header-qualified A+B native density with two-slot durable checkpoints."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

import asdf
import numpy as np

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_coupled_stage_b import used_bytes


def crc_manifest(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        crc,size,name = line.split()
        if name in rows or Path(name).name != name:
            raise ValueError('duplicate or nonlocal CRC filename')
        rows[name] = (int(crc),int(size))
    return rows


def verify_particle_file(args):
    path,phase,sample,expected = args
    before = c.file_record(path)
    output = subprocess.run(['/usr/bin/cksum',str(path)],check=True,text=True,capture_output=True).stdout
    crc,size,_ = output.strip().split(maxsplit=2)
    if (int(crc),int(size)) != expected:
        raise ValueError(f'particle CRC mismatch: {path}')
    with asdf.open(path,lazy_load=True) as f:
        header = f.tree['header']
        checks = {'Redshift':.2, 'BoxSizeHMpc':2000.,
                  'ParticleSubsampleA':.03, 'ParticleSubsampleB':.07}
        if header.get('SimName') != f'AbacusSummit_base_c000_{phase}':
            raise ValueError('particle phase mismatch')
        for key,value in checks.items():
            if abs(float(header.get(key,float('nan')))-value) > 1e-8 or not np.isfinite(float(header.get(key,float('nan')))):
                raise ValueError(f'particle header mismatch: {key}')
        native = {key:str(header.get(key)) for key in ('SimName','Redshift','BoxSizeHMpc','H0','Omega_M')}
    if c.file_record(path) != before:
        raise ValueError('particle source changed during verification')
    return dict(**before,crc=int(crc),sample=sample,header=native)


def particle_inventory(phase, directory):
    c.phase_guard(phase)
    marker = directory/'PARTICLES_VERIFIED.json'
    if marker.exists():
        result = c.verify_receipt(marker,payload=False)
        for record in result['files']:
            if c.file_record(record['path']) != {k:record[k] for k in ('path','bytes','mtime_ns')}:
                raise ValueError('verified particle source changed')
        return result
    if phase in ('ph000','ph002','ph003'):
        raise ValueError('historical native products require explicit provenance adoption, not automatic rebuilding')
    sources = c.paths(phase)
    a_root = sources['snapshot_root']
    b_root = a_root if phase == 'ph007' else c.ROOT/'particle_b'/phase/'halos/z0.200'
    if phase != 'ph007':
        transfer = json.loads((c.ROOT/'particle_b'/phase/'TRANSFER_COMPLETE.json').read_text())
        if transfer['config_sha256'] != c.sha256(c.CONFIG):
            raise ValueError('B transfer authority mismatch')
    jobs,manifests = [],[]
    for sample,root in (('A',a_root),('B',b_root)):
        for kind in ('field','halo'):
            folder = root/f'{kind}_rv_{sample}'
            manifest = folder/'checksums.crc32'
            expected = crc_manifest(manifest)
            names = [f'{kind}_rv_{sample}_{i:03d}.asdf' for i in range(34)]
            if set(expected) != set(names) or {p.name for p in folder.glob('*.asdf')} != set(names):
                raise ValueError('incomplete or extra particle slabs')
            manifests.append(c.file_record(manifest,content_hash=True))
            jobs.extend((folder/name,phase,sample,expected[name]) for name in names)
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(verify_particle_file,jobs))
    result = dict(**c.provenance(),phase=phase,files=records,crc_manifests=manifests,
                  source_code_sha256=c.sha256(__file__),elapsed_seconds=time.monotonic()-started,
                  **{'pass':True})
    c.atomic_json(marker,result)
    return result


def load_checkpoint(directory,binding,shape):
    pointer = directory/'LATEST.json'
    if not pointer.exists():
        return np.zeros(shape,dtype=np.float32),dict(next_file=0,particles=0,slot=1)
    state = json.loads(pointer.read_text())
    if state['binding'] != binding or state['slot'] not in (0,1):
        raise ValueError('checkpoint binding/slot mismatch')
    payload = directory/f"counts_slot{state['slot']}.npy"
    if c.sha256(payload) != state['sha256']:
        raise ValueError('committed native checkpoint corrupted; no silent fallback')
    grid = np.load(payload,allow_pickle=False)
    if grid.shape != shape or grid.dtype != np.float32:
        raise ValueError('native checkpoint shape/dtype mismatch')
    return grid,state


def save_checkpoint(directory,grid,state,binding):
    slot = 1-int(state['slot'])
    payload = directory/f'counts_slot{slot}.npy'
    temporary = directory/f'counts_attempt_{time.time_ns()}.npy'
    np.save(temporary,grid,allow_pickle=False)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    checksum = c.sha256(temporary)
    # Only the inactive generated checkpoint slot is replaced. LATEST remains
    # recoverable throughout the write; no source or user artifact is deleted.
    os.replace(temporary,payload)
    result = dict(state,slot=slot,binding=binding,sha256=checksum)
    c.atomic_json(directory/'LATEST.json',result,replace=True)
    return result


def build(phase,stop_after_seconds,threads=32):
    c.require_compute(); c.phase_guard(phase); c.bind_run()
    directory = c.ROOT/'matter'/phase
    with c.single_writer(directory):
        marker = directory/'DENSITY_COMPLETE.json'
        if marker.exists():
            return c.verify_receipt(marker)
        c.verify_receipt(c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json',payload=False)
        inputs = particle_inventory(phase,directory)
        binding = c.digest(dict(config=c.sha256(c.CONFIG),inputs=c.sha256(directory/'PARTICLES_VERIFIED.json'),
                                density_code=c.sha256(__file__)))
        n=c.config()['target']['native_grid']
        grid,state=load_checkpoint(directory,binding,(n,n,n))
        if used_bytes(c.ROOT)+3*grid.nbytes > c.config()['approval']['scratch_bytes']:
            raise RuntimeError('insufficient approved Scratch budget for transactional density slots')
        from abacusnbody.data.read_abacus import read_asdf
        from abacusnbody.analysis.tsc import tsc_parallel
        started=time.monotonic()
        for i,record in enumerate(inputs['files'][state['next_file']:],start=state['next_file']):
            path=Path(record['path'])
            data=read_asdf(path,load=['pos'],verbose=False)
            positions=np.asarray(data['pos'],dtype=np.float32)
            positions=np.mod(positions,2000.)
            if not np.isfinite(positions).all():
                raise ValueError('nonfinite particle positions')
            tsc_parallel(positions,grid,2000.,nthread=threads)
            state['particles']+=len(positions)
            state['next_file']=i+1
            del data,positions
            paused=time.monotonic()-started >= stop_after_seconds
            if (i+1)%16==0 or i+1==len(inputs['files']) or paused:
                state=save_checkpoint(directory,grid,state,binding)
            print(json.dumps(dict(phase=phase,files=i+1,particles=state['particles'],
                                  elapsed_seconds=time.monotonic()-started)),flush=True)
            if paused and i+1<len(inputs['files']):
                return {'paused':True,'phase':phase,'next_file':i+1}
        count=float(grid.sum(dtype=np.float64))
        error=abs(count-state['particles'])/state['particles']
        if error>2e-6 or not np.isfinite(count) or np.min(grid)<0:
            raise ValueError('native density positivity/count conservation failed')
        output=directory/'counts_tsc_2048.npy'
        source=directory/f"counts_slot{state['slot']}.npy"
        if output.exists():
            raise ValueError('unreceipted final density output')
        os.link(source,output)
        result=dict(**c.provenance(),phase=phase,role=c.ROLES[phase],binding=binding,
                    source_code_sha256=c.sha256(__file__),particle_count=state['particles'],
                    deposited_count=count,relative_count_error=error,processed_files=state['next_file'],
                    ngrid=n,box_mpc_h=2000.,dtype='float32',subsample_fraction=.10,
                    outputs=[dict(**c.file_record(output),sha256=state['sha256'])],
                    elapsed_seconds_this_segment=time.monotonic()-started,**{'pass':True})
        c.atomic_json(marker,result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True)
    p.add_argument('--stop-after-seconds',type=float,default=12000)
    p.add_argument('--threads',type=int,default=32)
    args=p.parse_args()
    result=build(args.phase,args.stop_after_seconds,args.threads)
    print(json.dumps(result),flush=True)
    raise SystemExit(75 if result.get('paused') else 0)
