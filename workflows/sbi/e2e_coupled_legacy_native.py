"""Explicit qualification of the existing 000/002/003 native-matter sources.

002/003: verify all original A+B CRCs/headers/counts and adopt the byte-identical
native arrays through receipts, without copying them. 000: its old manifest has
no exact run counts, so rebuild from the already-restored particles. No restores.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import asdf
import numpy as np

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_matter as matter
from workflows.sbi.e2e_coupled_stage_b import used_bytes

LEGACY=('ph000','ph002','ph003')
NATIVE_INDEX=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v1/data_20260908/native_truth/NATIVE_TRUTH_COMPLETE.json')


def legacy_paths(phase):
    c.phase_guard(phase)
    if phase not in LEGACY: raise PermissionError('only registered legacy native phases')
    registry=json.loads((c.REPO/c.config()['source_registry']).read_text())
    a=c.paths(phase)['snapshot_root']
    b=(Path(registry['phases'][phase]['particle_b']['root']) if phase=='ph000' else
       Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase/particle_b')/
       f'AbacusSummit_base_c000_{phase}'/'halos/z0.200')
    manifest=Path(f'/pscratch/sd/d/dkololgi/abacus/p10_multiphase/{phase}/targets/density/'
                  f'AbacusSummit_base_c000_{phase}_z0.200_ngrid2048_ab10_tsc_counts.manifest.json')
    return c.guarded(a,phase),c.guarded(b,phase),c.guarded(manifest,phase)


def qualified_file(args):
    record=matter.verify_particle_file(args)
    with asdf.open(record['path'],lazy_load=True) as f:
        data=f.tree['data']['rvint']
        shape=tuple(data.shape)
        if len(shape)!=2 or shape[1]!=3 or shape[0]<0:
            raise ValueError('invalid Abacus packed position shape')
        record['particle_count']=int(shape[0])
    return record


def inventory(phase,directory):
    c.require_compute()
    a,b,manifest=legacy_paths(phase)
    binding=dict(builder_sha256=c.sha256(__file__),verifier_sha256=c.sha256(matter.__file__),
                 legacy_manifest_sha256=c.sha256(manifest),a_root=str(a),b_root=str(b))
    marker=directory/'PARTICLES_VERIFIED.json'
    if marker.exists():
        record=c.verify_receipt(marker,payload=False)
        if record['binding']!=binding: raise ValueError('legacy particle verifier/source drift')
        for file in record['files']:
            if c.file_record(file['path'])!={k:file[k] for k in ('path','bytes','mtime_ns')}:
                raise ValueError('legacy particle source changed')
        return record
    jobs=[]; manifests=[]
    for sample,root in (('A',a),('B',b)):
        for kind in ('field','halo'):
            folder=root/f'{kind}_rv_{sample}'; crc_path=folder/'checksums.crc32'
            expected=matter.crc_manifest(crc_path)
            names=[f'{kind}_rv_{sample}_{i:03d}.asdf' for i in range(34)]
            if set(expected)!=set(names) or {p.name for p in folder.glob('*.asdf')}!=set(names):
                raise ValueError('legacy native source has missing/extra particle slabs')
            manifests.append(c.file_record(crc_path,content_hash=True))
            jobs.extend((folder/name,phase,sample,expected[name]) for name in names)
    with ThreadPoolExecutor(max_workers=8) as executor:
        files=list(executor.map(qualified_file,jobs))
    record=dict(**c.provenance(),phase=phase,binding=binding,files=files,
                particle_count=sum(v['particle_count'] for v in files),crc_manifests=manifests,
                outputs=[],**{'pass':True})
    c.atomic_json(marker,record)
    print(json.dumps(dict(phase=phase,particle_crc_headers_verified=len(files),
                          particles=record['particle_count'])),flush=True)
    return record


def validate_legacy_record(phase,manifest,reference,verified_particles):
    if phase not in ('ph002','ph003'):
        raise PermissionError('ph000 requires a fresh accountable build')
    build=manifest['build']; target=manifest['target_contract']
    if (manifest['phase']!=phase or reference['phase']!=phase
            or build['processed_file_count']!=136 or build['ngrid']!=2048
            or build['boxsize_mpc_h']!=2000. or target['mass_assignment']!='TSC'
            or build['particle_count']!=verified_particles
            or target['redshift']!=.2 or target['cosmology']!='c000'
            or target['particle_subsamples']['total_fraction']!=.10):
        raise ValueError('legacy density contract/count mismatch')
    return build


def adopt(phase,directory,particles):
    _,_,manifest_path=legacy_paths(phase)
    manifest=json.loads(manifest_path.read_text())
    reference=next(v for v in json.loads(NATIVE_INDEX.read_text())['phases'] if v['phase']==phase)
    if c.sha256(manifest_path)!=reference['density_manifest_sha256']:
        raise ValueError('legacy density manifest differs from verified native index')
    build=validate_legacy_record(phase,manifest,reference,particles['particle_count'])
    path=c.guarded(reference['density_path'],phase)
    if path!=Path(build['output']).resolve() or c.sha256(path)!=reference['density_sha256']:
        raise ValueError('legacy density payload/path hash mismatch')
    grid=np.load(path,mmap_mode='r',allow_pickle=False)
    if grid.shape!=(2048,)*3 or grid.dtype!=np.float32:
        raise ValueError('legacy native array shape/dtype mismatch')
    total=0.
    for start in range(0,2048,16):
        slab=grid[start:start+16]
        if not np.isfinite(slab).all() or np.min(slab)<0: raise ValueError('invalid legacy density values')
        total+=float(slab.sum(dtype=np.float64))
    chunk_total=total
    # Reproduce the original whole-array reduction for comparison with its
    # recorded value. For 8.6 billion float32 cells, changing the summation tree
    # shifts float64 totals by about .0016 particles despite identical bytes.
    total=float(grid.sum(dtype=np.float64))
    error=abs(total-particles['particle_count'])/particles['particle_count']
    if error>2e-6 or abs(total-build['deposited_count'])>64*np.spacing(total):
        raise ValueError('legacy particle/grid conservation failure')
    record=dict(**c.provenance(),phase=phase,role=c.ROLES[phase],
        binding=dict(builder_sha256=c.sha256(__file__),particle_receipt_sha256=c.sha256(directory/'PARTICLES_VERIFIED.json'),
                     legacy_manifest_sha256=c.sha256(manifest_path),native_index_sha256=c.sha256(NATIVE_INDEX)),
        adoption='Existing byte-identical native array; original manifest and every A+B input requalified',
        particle_count=particles['particle_count'],deposited_count=total,relative_count_error=error,
        chunked_minus_original_reduction=chunk_total-total,
        processed_files=136,ngrid=2048,box_mpc_h=2000.,dtype='float32',subsample_fraction=.10,
        outputs=[dict(**c.file_record(path),sha256=reference['density_sha256'])],**{'pass':True})
    c.atomic_json(directory/'DENSITY_COMPLETE.json',record)
    return record


def rebuild_reference(directory,particles,seconds,threads):
    from abacusnbody.data.read_abacus import read_asdf
    from abacusnbody.analysis.tsc import tsc_parallel
    phase='ph000'; started=time.monotonic()
    binding=c.digest(dict(config=c.sha256(c.CONFIG),builder=c.sha256(__file__),
        checkpoint_code=c.sha256(matter.__file__),particles=c.sha256(directory/'PARTICLES_VERIFIED.json')))
    grid,state=matter.load_checkpoint(directory,binding,(2048,)*3)
    if used_bytes(c.ROOT)+3*grid.nbytes>c.config()['approval']['scratch_bytes']:
        raise RuntimeError('insufficient approved Scratch headroom for legacy rebuild')
    for index,item in enumerate(particles['files'][state['next_file']:],start=state['next_file']):
        data=read_asdf(item['path'],load=['pos'],verbose=False)
        positions=np.mod(np.asarray(data['pos'],dtype=np.float32),2000.)
        if not np.isfinite(positions).all() or len(positions)!=item['particle_count']:
            raise ValueError('packed/source particle count mismatch')
        tsc_parallel(positions,grid,2000.,nthread=threads)
        state['particles']+=len(positions); state['next_file']=index+1
        del data,positions
        paused=time.monotonic()-started>=seconds
        if (index+1)%16==0 or index==135 or paused:
            state=matter.save_checkpoint(directory,grid,state,binding)
        print(json.dumps(dict(phase=phase,files=index+1,particles=state['particles'])),flush=True)
        if paused and index<135: return dict(phase=phase,paused=True,next_file=index+1)
    total=float(grid.sum(dtype=np.float64)); expected=particles['particle_count']
    error=abs(total-expected)/expected
    if state['particles']!=expected or error>2e-6 or np.min(grid)<0 or not np.isfinite(total):
        raise ValueError('rebuilt legacy native density conservation failure')
    output=directory/'counts_tsc_2048.npy'; source=directory/f"counts_slot{state['slot']}.npy"
    if output.exists():
        if not os.path.samefile(source,output): raise ValueError('unreceipted density is not the final checkpoint')
    else:
        os.link(source,output)
    record=dict(**c.provenance(),phase=phase,role='train',binding=binding,
        adoption='Fresh A+B replay; older uncounted legacy array preserved but not used',
        particle_count=expected,deposited_count=total,relative_count_error=error,processed_files=136,
        ngrid=2048,box_mpc_h=2000.,dtype='float32',subsample_fraction=.10,
        outputs=[dict(**c.file_record(output),sha256=state['sha256'])],**{'pass':True})
    c.atomic_json(directory/'DENSITY_COMPLETE.json',record)
    return record


def build(phase,seconds=10000,threads=16):
    c.require_compute(); c.phase_guard(phase); legacy_paths(phase); c.bind_run()
    directory=c.ROOT/'matter'/phase
    with c.single_writer(directory):
        final=directory/'DENSITY_COMPLETE.json'
        if final.exists(): return c.verify_receipt(final)
        particles=inventory(phase,directory)
        # Particle verification may precede the catalogue join. Adoption/build
        # completion may not: retain the same pairing gate as new native phases.
        pair=c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json'
        if not pair.exists(): return dict(phase=phase,paused=True,waiting_for_pairing=True)
        c.verify_receipt(pair,payload=False)
        return (rebuild_reference(directory,particles,seconds,threads) if phase=='ph000'
                else adopt(phase,directory,particles))


def run(seconds):
    c.require_compute(); started=time.monotonic()
    snapshot=c.REPO/'SOURCE.json'
    for record in json.loads(snapshot.read_text())['files']:
        if c.sha256(c.REPO/record['relative'])!=record['sha256']:
            raise ValueError('legacy qualification snapshot changed')
    remaining=list(LEGACY)
    while remaining and time.monotonic()-started<seconds-900:
        did_work=False
        for phase in tuple(remaining):
            result=build(phase,max(60,seconds-(time.monotonic()-started)-600))
            print(json.dumps(result),flush=True)
            if not result.get('paused'):
                remaining.remove(phase); did_work=True
            elif not result.get('waiting_for_pairing'):
                return 75
        if remaining and not did_work: time.sleep(60)
    return 75 if remaining else 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seconds',type=float,default=10500)
    a=p.parse_args(); raise SystemExit(run(a.seconds))
