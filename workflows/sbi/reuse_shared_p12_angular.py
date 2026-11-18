"""Qualify the shared campaign's target-free maps for canonical P12 response IO."""
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_coupled_product_audit import Audit
from workflows.sbi.audit_shared_phase_sources import angular_sources
from workflows.abacus_tweb import p3br_build_random_response as response


def source_path(phase,ready):
    c.phase_guard(phase)
    if not c.SHARED_POLICY or c.config()['phase_policy']!=ready['phase_policy']:
        raise ValueError('P12 and CFM shared map policies differ')
    return c.ROOT/'observations'/phase/'angular/ANGULAR_COMPLETE.json'


def prepare(phase,ready,directory):
    c.require_compute();source=source_path(phase,ready);audit=Audit(phase)
    record=audit.receipt(source);proof=angular_sources(audit,record)
    if len(record['outputs'])!=1:raise ValueError('Ambiguous angular map')
    payload=record['outputs'][0]
    with np.load(payload['path'],allow_pickle=False) as saved:
        result=response.normalized_map(saved['raw_counts_by_domain'])
        for key in saved.files:
            if not np.array_equal(saved[key],result[key]):raise ValueError('P12/CFM angular arithmetic mismatch')
    path=Path(directory)/'randoms_n18.npz';metadata=path.with_suffix('.json')
    binding=dict(source_receipt_sha256=c.sha256(source),source_payload_sha256=payload['sha256'],
                 source_helper_sha256=c.sha256(__file__),audit_helper_sha256=c.sha256(c.REPO/'workflows/sbi/audit_shared_phase_sources.py'))
    if metadata.exists():
        prior=json.loads(metadata.read_text())
        if prior.get('shared_binding')!=binding or prior['phase']!=phase or c.sha256(path)!=prior['sha256']:
            raise ValueError('Existing shared P12 angular binding differs')
    else:
        if path.exists():raise FileExistsError('Unreceipted angular output: '+str(path))
        response.save_map(path,result,dict(schema_version='p3br-shared-angular-random-map-v1',phase=phase,
            nside=256,ordering='RING',random_ids=list(range(18)),shared_binding=binding,
            source_receipt=str(source),source_qualification=proof,phase_policy=ready['phase_policy']))
    return dict(reused_shared_map=str(source),shared_binding=binding,qualification=proof,
                raw_random_hashes_inherited=True,new_raw_random_scan=False)
