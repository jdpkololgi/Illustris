"""Independent all-payload phase audit, not a global training-readiness claim.

Re-read SHA256s and physical arrays, verify receipt links and targetless reads,
and recompute mass/trace/owned-condition identities. Particle CRC qualification
is checked against official manifests and unchanged input metadata; this audit
does not unnecessarily repeat the entire multi-terabyte CRC scan.
"""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_conditions as reader
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi import e2e_coupled_matter as matter
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi import e2e_coupled_condition_products as conditions_builder
from workflows.sbi import e2e_coupled_observations as observations_builder


class Audit:
    def __init__(self,phase):
        c.phase_guard(phase); self.phase=phase; self.hashed={}

    def file(self,item,hash_payload=True):
        path=c.guarded(item['path'],self.phase)
        before=c.file_record(path)
        if any(before[k]!=item[k] for k in ('bytes','mtime_ns') if k in item):
            raise ValueError('source/payload metadata changed: '+str(path))
        if hash_payload:
            # Lustre timestamps can have coarse resolution. An equal-size
            # rewrite within that resolution must not reuse a cached hash.
            actual=c.sha256(path)
            if actual!=item['sha256']: raise ValueError('source/payload SHA256 changed: '+str(path))
            if c.file_record(path)!=before: raise ValueError('source changed during hash')
            self.hashed[str(path)]=dict(**before,sha256=actual)
        return path

    def receipt(self,path,cartesian=False):
        item=c.file_record(path,content_hash=True); self.file(item)
        record=(coord if cartesian else c).verify_receipt(path,payload=False)
        if record.get('phase')!=self.phase: raise ValueError('receipt phase mismatch')
        if 'role' in record and record['role']!=c.ROLES[self.phase]: raise ValueError('receipt role mismatch')
        for item in record.get('outputs',[]): self.file(item)
        return record


def pair_index(record,expected,directory):
    rows=record['pair_receipts']; index={Path(v['path']).stem:v for v in rows}
    if len(index)!=len(rows) or set(index)!=set(expected):
        raise ValueError('missing, duplicate or unexpected phase pair IDs')
    if any(Path(v['path']).resolve().parent!=directory.resolve() for v in rows):
        raise PermissionError('pair receipt outside its declared observation/target directory')
    return index


def read_target(path,phase,pair_id):
    with h5py.File(path,'r') as saved:
        expected=dict(schema=targets.SCHEMA,phase=phase,pair_id=pair_id,role=c.ROLES[phase],
                      distance_unit='Mpc/h',coordinate_sha256=c.sha256(coord.CONFIG),
                      smoothing_mpc_h=7.,smoothing_count=1)
        if any(saved.attrs.get(k)!=v for k,v in expected.items()):
            raise ValueError('target metadata/units/smoothing mismatch')
        if bool(saved.attrs.get('contains_observations',True)):
            raise ValueError('target shard mixes observation inputs')
        if set(saved)!={'rho_joint','coarse_rho_extended','fullbox_tensor_cores'}:
            raise ValueError('target payload schema mismatch')
        for key in saved:
            if not isinstance(saved.get(key,getlink=True),h5py.HardLink):
                raise PermissionError('external target links prohibited')
            if saved[key].is_virtual or saved[key].external:
                raise PermissionError('external/virtual target storage prohibited')
        return {key:saved[key][:] for key in saved}


def particles(audit,record):
    phase=audit.phase
    expected={f'{kind}_rv_{sample}_{slab:03d}.asdf' for kind in ('field','halo')
              for sample in ('A','B') for slab in range(34)}
    files=record['files']
    if len(files)!=136 or {Path(v['path']).name for v in files}!=expected:
        raise ValueError('particle sample/kind/slab completeness failure')
    manifests={}
    for item in record['crc_manifests']:
        path=audit.file(item); manifests[path.parent]=matter.crc_manifest(path)
    if len(manifests)!=4: raise ValueError('particle CRC manifest completeness failure')
    for item in files:
        path=audit.file(item,hash_payload=False)
        if manifests[path.parent][path.name]!=(item['crc'],item['bytes']):
            raise ValueError('qualified particle CRC differs from official manifest')
        header=item['header']
        if (header['SimName']!=f'AbacusSummit_base_c000_{phase}'
                or float(header['Redshift'])!=.2 or float(header['BoxSizeHMpc'])!=2000.
                or abs(float(header['H0'])-67.36)>1e-6
                or abs(float(header['Omega_M'])-.315192)>1e-7):
            raise ValueError('particle phase/epoch/cosmology mismatch')
    verifier=record.get('source_code_sha256',record.get('binding',{}).get('verifier_sha256'))
    if verifier!=c.sha256(matter.__file__): raise ValueError('particle verifier code unrecognized')
    return dict(files=136,crc_manifest_binding=True,source_metadata_unchanged=True,
                prior_crc_payload_verification_retained=True,crc_rescanned=False)


def run(phase):
    c.require_compute(); c.phase_guard(phase); coord.require_host_checks()
    started=time.monotonic(); audit=Audit(phase)
    directory=coord.ROOT/'product_audit'/phase
    with c.single_writer(directory):
        # Publish unique audit generations. Never reuse yesterday's green flag
        # instead of reading today's actual payloads.
        source=c.ROOT/'observations'/phase
        parent=audit.receipt(source/'PARENT_COMPLETE.json')
        observed=audit.receipt(source/'OBSERVED_COMPLETE.json')
        if (observed['parent_receipt_sha256']!=c.sha256(source/'PARENT_COMPLETE.json')
                or not observed['exact_id_sky_rsd_join'] or observed['target_columns_present']
                or parent['target_columns_present'] or not parent['parity']['row_identity_verified']):
            raise ValueError('paired catalogue identity/targetless gate failed')
        for item in (*parent['sources'].values(),observed['source']): audit.file(item)
        angular=audit.receipt(source/'angular/ANGULAR_COMPLETE.json')
        if angular['random_ids']!=list(range(18)) or len(angular['sources'])!=18:
            raise ValueError('angular response random panel incomplete')
        for ordinal,item in enumerate(angular['sources']):
            path=Path(item['receipt'])
            if path!=source/'angular'/f'random_{ordinal:02d}.json': raise ValueError('random stream mismatch')
            if c.sha256(path)!=item['sha256']: raise ValueError('random receipt hash mismatch')
            record=audit.receipt(path); audit.file(record['source'])
        p=audit.receipt(c.ROOT/'matter'/phase/'PARTICLES_VERIFIED.json')
        particle_qa=particles(audit,p)
        native=audit.receipt(c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json')
        if (native['processed_files']!=136 or native['ngrid']!=2048 or native['box_mpc_h']!=2000.
                or native['subsample_fraction']!=.10): raise ValueError('native contract failure')
        counts=np.load(native['outputs'][0]['path'],mmap_mode='r',allow_pickle=False)
        if counts.shape!=(2048,)*3 or counts.dtype!=np.float32: raise ValueError('native array mismatch')
        for start in range(0,2048,8):
            block=counts[start:start+8]
            if not np.isfinite(block).all() or np.min(block)<0: raise ValueError('invalid native counts')
        deposited=float(counts.sum(dtype=np.float64)); del counts
        count_error=abs(deposited-native['particle_count'])/native['particle_count']
        if count_error>2e-6 or abs(deposited-native['deposited_count'])>64*np.spacing(deposited):
            raise ValueError('independent native count conservation failure')
        geo_path=coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
        geo=audit.receipt(geo_path,True); geometry_qa=geometry.verify_rows(geo['pairs'],phase)
        if geo['counts_or_targets_read'] is not False: raise ValueError('geometry is not support-only')
        raw_sources={}; response_hashes={}; coarse_hashes={}
        for cap in ('NGC','SGC'):
            path=coord.ROOT/'observations'/phase/f'{cap}_COMPLETE.json'
            if c.sha256(path)!=geo['sources'][cap]['sha256']: raise ValueError('geometry response source drift')
            raw=audit.receipt(path,True); coord.validate_grid(raw['grid']); audit.file(raw['selection'])
            if raw['source_code_sha256']!=c.sha256(observations_builder.__file__):
                raise ValueError('unrecognized observation builder')
            if not all(raw['gates'].values()) or set(raw['selection']['fit_phases'])!={'ph000','ph002','ph003','ph004','ph005'}:
                raise ValueError('response/selection qualification drift')
            raw_sources[cap]=raw
            response_hashes[cap]=c.sha256(path)
        condition_dir=coord.ROOT/'conditions'/phase; target_dir=coord.ROOT/'targets'/phase
        conditions=audit.receipt(condition_dir/'CONDITIONS_COMPLETE.json',True)
        target=audit.receipt(target_dir/'TARGETS_COMPLETE.json',True)
        for cap in ('NGC','SGC'):
            path=condition_dir/f'{cap}_FACTOR4.json'; record=audit.receipt(path,True)
            if record['binding']!=dict(**conditions['binding'],response_receipt_sha256=response_hashes[cap]):
                raise ValueError('coarsened response source binding mismatch')
            coarse_hashes[cap]=c.sha256(path)
        if target['science_scores_evaluated'] is not False: raise ValueError('preparation opened predictive scoring')
        expected=[r['pair_id'] for r in geo['pairs']]
        ci=pair_index(conditions,expected,condition_dir); ti=pair_index(target,expected,target_dir)
        if (target['binding']['native_receipt_sha256']!=c.sha256(c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json')
                or target['binding']['geometry_receipt_sha256']!=c.sha256(geo_path)
                or conditions['binding']['geometry_receipt_sha256']!=c.sha256(geo_path)):
            raise ValueError('product native/geometry binding mismatch')
        for item in target['stage_receipts']:
            audit.file(item); audit.receipt(item['path'],True)
        checks=[]
        for row in geo['pairs']:
            pair_id=row['pair_id']
            for item in (ci[pair_id],ti[pair_id]):
                audit.file(item); audit.receipt(item['path'],True)
            condition_record=coord.verify_receipt(ci[pair_id]['path'],payload=False)
            expected_binding=dict(**conditions['binding'],response_receipt_sha256=response_hashes[row['cap']],
                coarse_receipt_sha256=coarse_hashes[row['cap']],row_sha256=c.digest(row))
            if condition_record['binding']!=expected_binding:
                raise ValueError('pair observation source/geometry binding mismatch')
            arrays=reader.load_pair(phase,pair_id)
            with h5py.File(raw_sources[row['cap']]['outputs'][0]['path'],'r') as raw:
                for side,crop in enumerate(op.layout()['owned_core_crops_in_joint']):
                    center=np.array(row['center'])+[32*side,0,0]
                    region=tuple(slice(v-16,v+16) for v in center)
                    expected_count=np.log1p(raw['counts'][region]).reshape(16,2,16,2,16,2).mean((1,3,5))
                    owned=tuple(slice(*s) for s in crop)
                    if not np.array_equal(arrays['joint'][0][owned],expected_count):
                        raise ValueError('independent owned-core count extraction failed')
                    if float(arrays['support'][owned].mean(dtype=np.float64))!=row['science_core_support_fractions'][side]:
                        raise ValueError('independent owned-core support failed')
            record=coord.verify_receipt(ti[pair_id]['path'],payload=False)
            if record['binding']!=target['binding']: raise ValueError('pair target source binding mismatch')
            payload=record['outputs'][0]
            if Path(payload['path']).resolve().parent!=target_dir.resolve(): raise PermissionError('target outside target directory')
            values=read_target(payload['path'],phase,pair_id)
            qa=targets.density_qa(values['rho_joint'],values['coarse_rho_extended'])
            qa.update(targets.fullbox_qa(values['rho_joint'],values['fullbox_tensor_cores']))
            from workflows.sbi import e2e_coupled_views as views
            parents=views.independent_parents(values['rho_joint'])
            if not np.array_equal(views.assemble_independent(*parents),values['rho_joint']):
                raise ValueError('I/J common-domain truth assembly is not exact')
            checks.append(dict(pair_id=pair_id,owned_condition_count_exact=True,support_exact=True,
                               independent_joint_truth_domain_exact=True,**qa))
        result=dict(**coord.provenance(),phase=phase,role=c.ROLES[phase],source_code_sha256=c.sha256(__file__),
            audit_dependencies={Path(module.__file__).name:c.sha256(module.__file__) for module in
                (c,coord,reader,geometry,matter,op,targets,conditions_builder,observations_builder,views)},
            geometry=geometry_qa,particles=particle_qa,pairs=checks,
            actual_sha256_files=list(audit.hashed.values()),native_count_error=count_error,
            elapsed_seconds=time.monotonic()-started,outputs=[],science_scores_evaluated=False,
            global_training_readiness=None,**{'pass':True})
        path=directory/f'PRODUCT_AUDIT_{time.time_ns()}.json'; c.atomic_json(path,result)
        return dict(path=str(path),sha256=c.sha256(path),phase=phase,pairs=len(checks),
                    actual_hashes=len(audit.hashed),elapsed_seconds=result['elapsed_seconds'],**{'pass':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',required=True)
    a=p.parse_args(); print(json.dumps(run(a.phase)),flush=True)
