"""Strict P4 input binding for separate galaxy identity and native host groups."""
import json
from pathlib import Path
import numpy as np
from workflows.abacus_tweb.product_lifecycle import digest


def load(identity_path,host_path,p1_path,targetid,active):
    identity_path,host_path,p1_path=map(Path,(identity_path,host_path,p1_path))
    identity=json.loads(identity_path.read_text());host=json.loads(host_path.read_text())
    p1_hash=digest(p1_path)
    if (identity.get('schema')!='p12a-galaxy-identity-v1' or not identity.get('identity_pass')
            or identity.get('p1_sha256')!=p1_hash or host.get('p1_sha256')!=p1_hash
            or host.get('phase')!=identity.get('phase')
            or not host.get('cross_branch_host_mapping_pass')
            or not host.get('all_selected_masses_and_centrals_checked')):
        raise ValueError('Unqualified or mismatched identity/host sidecars')
    if identity.get('phase_roles_changed') or identity.get('low_branch_rows_removed')!=0:
        raise ValueError('Identity repair changed roles or discarded low-z galaxies')
    if digest(Path(identity['output']))!=identity['output_sha256']:
        raise ValueError('Identity payload changed')
    if digest(Path(host['output']['path']))!=host['output']['sha256']:
        raise ValueError('Host payload changed')
    active=np.asarray(active,bool);targetid=np.asarray(targetid)
    with np.load(identity['output'],allow_pickle=False) as saved:
        if not np.array_equal(saved['parent_node_id'],np.arange(len(active))):
            raise ValueError('Identity parent alignment failed')
        eligible=saved['supervised_eligible'].astype(bool)
        duplicate=saved['periodic_duplicate_image'].astype(bool)
    if (eligible.shape!=active.shape or np.any(eligible&~active)
            or not np.array_equal(duplicate,active&~eligible)
            or int(eligible.sum())!=identity['eligible_rows']):
        raise ValueError('Invalid galaxy eligibility partition')
    groups=np.full(len(active),-1,dtype=np.int64)
    with np.load(host['output']['path'],allow_pickle=False) as saved:
        ids=saved['parent_node_id']
        if (not np.array_equal(ids,np.flatnonzero(active))
                or not np.array_equal(saved['targetid'],targetid[ids])
                or np.any(saved['host_group']<0)):
            raise ValueError('Host map does not cover exactly the active rows')
        groups[ids]=saved['host_group']
    provenance=dict(galaxy_identity=str(identity_path),galaxy_identity_sha256=digest(identity_path),
        host_mapping=str(host_path),host_mapping_sha256=digest(host_path),
        distinct_galaxies_preserved=True,cross_branch_hosts_qualified=True,
        identity_audit=identity)
    return eligible,duplicate,groups,provenance
