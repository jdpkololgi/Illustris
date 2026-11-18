"""Adopt immutable target-free random counts and recompute current normalization.

Map and registry payloads are freshly hashed; historical random-file hashes are
inherited, with current path/size checks, not represented as a fresh CFS scan.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_coupled_observations import publish_npz
from workflows.abacus_tweb import p3br_build_random_response as response
from shared.phase_policy import REPO


def adopt(phase):
    c.require_compute();c.phase_guard(phase)
    if not c.config().get('phase_policy'):raise ValueError('Versioned campaign required')
    source=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase')/phase/'p3b_random_response_v1/angular/randoms_n18.npz'
    metadata=source.with_suffix('.json');r=json.loads(metadata.read_text())
    if r['phase']!=phase or r['random_ids']!=list(range(18)) or r['nside']!=256 or r['ordering']!='RING':
        raise ValueError('Random-map identity mismatch')
    if c.sha256(source)!=r['sha256'] or c.sha256(r['source_registry'])!=r['source_registry_sha256']:
        raise ValueError('Historical angular payload or registry changed')
    registry=json.loads(Path(r['source_registry']).read_text())
    records=registry['mock_phases'][phase]['full_random']
    lookup={a['path']:a for a in records}
    paths=c.paths(phase)['randoms']
    if [str(x) for x in paths]!=[a['path'] for a in r['sources']]:raise ValueError('Random sources differ')
    for a in r['sources']:
        b=lookup[a['path']]
        if a['sha256']!=b['sha256'] or Path(a['path']).stat().st_size!=b['bytes']:
            raise ValueError('Random source fingerprint differs')
    directory=c.ROOT/'observations'/phase/'angular'
    with c.single_writer(directory):
        marker=directory/'ANGULAR_COMPLETE.json'
        if marker.exists():raise FileExistsError(marker)
        with np.load(source) as data:counts=data['raw_counts_by_domain']
        if counts.shape!=(4,12*256**2) or not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0):
            raise ValueError('Invalid raw random counts')
        result=response.normalized_map(counts)
        output=publish_npz(directory,'angular',**{k:v for k,v in result.items() if k!='metadata'})
        receipt=dict(**c.provenance(),phase=phase,random_ids=list(range(18)),
            sources=[dict(receipt=str(metadata),sha256=c.sha256(metadata))],
            reuse=dict(source=c.file_record(source,content_hash=True),registry_sha256=r['source_registry_sha256'],
                scope='Fresh map and registry hashes; current random path/size checks; inherited historical random content hashes'),
            metadata=result['metadata'],legacy_algorithm_sha256=c.sha256(response.__file__),
            outputs=[c.file_record(output,content_hash=True)],**{'pass':True})
        c.atomic_json(marker,receipt)
    print(json.dumps(dict(phase=phase,angular_reused=True)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True);a=p.parse_args();adopt(a.phase)
