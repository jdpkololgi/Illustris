"""Audit explicit reuse provenance without disguising historical raw-file hashes.

Derived map payloads are freshly hashed and normalized maps are recomputed from
integer counts. Raw random hashes remain inherited with current metadata checks.
Particle checks still use the ordinary full CFM particle qualification gate.
"""
import json
from pathlib import Path
import numpy as np
import fitsio
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_pairing as pairing
from workflows.abacus_tweb import p3br_build_random_response as response
from workflows.sbi.prepare_shared_phase_observations import validate_parent_record


def parent_sources(audit,parent,observed):
    if not c.SHARED_POLICY:raise ValueError('Shared-campaign reuse only')
    audit.file(observed['source'])
    if 'sources' in parent:
        for item in parent['sources'].values():audit.file(item)
        return dict(parent_raw_sources_rehashed=True)
    if parent.get('schema')!='e2e-coupled-parent-reuse-v1':raise ValueError('Unknown parent provenance')
    original=audit.file(parent['original_parent_receipt']);record=json.loads(original.read_text())
    payload=parent['outputs'][0]
    if payload['sha256']!=record['output']['sha256']:raise ValueError('Parent proof/payload mismatch')
    sources=c.paths(audit.phase)
    with fitsio.FITS(payload['path']) as f:columns=f[1].get_colnames();rows=f[1].get_nrows()
    validate_parent_record(record,audit.phase,sources,columns)
    if rows!=record['output']['rows']:raise ValueError('Parent row count mismatch')
    parity=pairing.parent.validate_parent_against_forfa(payload['path'],sources['forfa'],rows,500_000)
    if not parity['row_identity_verified']:raise ValueError('Fresh parent/forFA identity mismatch')
    return dict(parent_current_forfa_parity=parity,parent_raw_cutsky_rehashed=False,
                original_parent_receipt_sha256=c.sha256(original))


def validate_counts(counts):
    if counts.shape!=(4,12*256**2) or not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0):
        raise ValueError('Invalid integer random counts')
    return counts


def angular_sources(audit,angular,*,source_root=None,source_config_sha256=None,reuse_source_root=None):
    if not c.SHARED_POLICY or angular['random_ids']!=list(range(18)):
        raise ValueError('Invalid shared random panel')
    phase=audit.phase;paths=c.paths(phase)['randoms'];sources=angular['sources']
    if len(sources)==1 and 'source' in angular.get('reuse',{}):
        item=sources[0];marker=Path(item['receipt'])
        if c.sha256(marker)!=item['sha256']:raise ValueError('Legacy random-map receipt drift')
        value=json.loads(marker.read_text())
        if (value['phase']!=phase or value['random_ids']!=list(range(18)) or value['nside']!=256 or value['ordering']!='RING'):
            raise ValueError('Legacy map authority mismatch')
        registry=Path(value['source_registry'])
        if c.sha256(registry)!=value['source_registry_sha256'] or value['source_registry_sha256']!=angular['reuse']['registry_sha256']:
            raise ValueError('Legacy map registry drift')
        registered=json.loads(registry.read_text())['mock_phases'][phase]['full_random']
        lookup={r['path']:r for r in registered}
        if [str(p) for p in paths]!=[r['path'] for r in value['sources']]:raise ValueError('Legacy random order differs')
        for raw in value['sources']:
            expected=lookup[raw['path']]
            if raw['sha256']!=expected['sha256']:raise ValueError('Legacy raw hash lineage mismatch')
            audit.file(expected,hash_payload=False)
        source=angular['reuse']['source'];path=audit.file(source)
        if source['sha256']!=value['sha256']:raise ValueError('Legacy map content mismatch')
        with np.load(path,allow_pickle=False) as saved:counts=validate_counts(saved['raw_counts_by_domain'])
    elif len(sources)==18:
        counts=np.zeros((4,12*256**2),dtype=np.int64)
        old_config=c.sha256(c.REPO/'configs/e2e_coupled_data_v1.json')
        root=Path(source_root or c.ROOT)/'observations'/phase/'angular'
        allowed_hash=source_config_sha256 or c.sha256(c.CONFIG)
        if 'source_receipt' in angular.get('reuse',{}):
            proof=Path(angular['reuse']['source_receipt'])
            expected=Path(reuse_source_root or c.config()['reuse_source_root'])/'observations'/phase/'angular/ANGULAR_COMPLETE.json'
            if proof!=expected or c.sha256(proof)!=angular['reuse']['source_receipt_sha256']:
                raise ValueError('Historical angular provenance drift')
            prior=json.loads(proof.read_text())
            if prior['sources']!=sources or prior['config_sha256']!=old_config or not prior['pass']:
                raise ValueError('Historical angular authority mismatch')
            root=proof.parent;allowed_hash=old_config
        for ordinal,item in enumerate(sources):
            marker=Path(item['receipt'])
            if marker!=root/f'random_{ordinal:02d}.json' or c.sha256(marker)!=item['sha256']:
                raise ValueError('Random stream order/hash mismatch')
            value=json.loads(marker.read_text())
            if value['config_sha256']!=allowed_hash or value['phase']!=phase or not value['pass']:
                raise ValueError('Random stream authority mismatch')
            if Path(value['source']['path']).resolve()!=paths[ordinal].resolve():raise ValueError('Raw random source differs')
            audit.file(value['source'],hash_payload=False)
            if len(value['outputs'])!=1:raise ValueError('Ambiguous random output')
            output=audit.file(value['outputs'][0])
            with np.load(output,allow_pickle=False) as saved:counts+=validate_counts(saved['counts'])
    else:raise ValueError('Unknown angular reuse provenance')
    expected=response.normalized_map(counts)
    with np.load(angular['outputs'][0]['path'],allow_pickle=False) as saved:
        if set(saved.files)!={k for k in expected if k!='metadata'}:raise ValueError('Angular map key mismatch')
        for key in saved.files:
            if not np.array_equal(saved[key],expected[key]):raise ValueError('Angular recomposition mismatch: '+key)
    return dict(random_files=18,raw_random_payloads_rehashed=False,
                raw_random_hashes_inherited=True,current_raw_metadata_checked=True,
                derived_payloads_rehashed=True,integer_counts_recomposition_exact=True,
                historical_algorithm_sha256=angular['legacy_algorithm_sha256'],
                current_algorithm_sha256=c.sha256(response.__file__))
