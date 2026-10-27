"""Exact target-cell equivalence never establishes a unique particle identity."""
import numpy as np
from galaxy_position_mapping import periodic_delta, field_indices

RULE_VERSION = 'all-candidate-cell-and-label-equivalence-v1'


def validate_equivalence(record, tolerance):
    ids=np.asarray(record['candidate_source_rows'])
    positions=np.asarray(record['candidate_positions'],dtype=float)
    query=np.asarray(record['query_position'],dtype=float)
    labels=np.asarray(record['candidate_eigenvalues'],dtype=float)
    target=np.asarray(record['query_eigenvalues'],dtype=float)
    n=len(ids)
    if (n<2 or ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer)
        or np.any(ids<0) or len(np.unique(ids))!=n or positions.shape!=(n,3)
        or query.shape!=(3,) or labels.shape!=(n,3) or target.shape!=(3,)):
        raise ValueError('Incomplete or duplicate candidate evidence')
    if record.get('candidate_search_exhaustive') is not True:
        raise ValueError('All candidates must be enumerated')
    if not all(np.isfinite(v).all() for v in (positions,query,labels,target)):
        raise ValueError('Nonfinite candidate evidence')
    if np.any(np.linalg.norm(periodic_delta(positions,query),axis=1)>=tolerance):
        raise ValueError('Candidate outside frozen coordinate tolerance')
    cells=field_indices(positions);query_cell=field_indices(query[None,:])[0]
    if not np.all(cells==query_cell):
        raise ValueError('Candidate cells differ')
    if not np.all(labels==target) or np.any(np.diff(target)<0):
        raise ValueError('Candidate labels differ or are unordered')
    return True
