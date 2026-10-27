"""Keep field-label availability separate from per-galaxy independent evidence."""
from pathlib import Path
import hashlib
import numpy as np

LABEL_POLICY_VERSION='galaxy-realspace-all-populations-v2'
INDEPENDENT_POSITION_CODES={0:'not_individually_cross_checked',1:'original_cubic_galaxy_xyz',
                            2:'native_halo_central_xyz_and_mass',3:'native_field_particle_central_xyz'}
REFERENCE_KIND_CODES={'cubic_galaxy':1,'resolved_central':2,'unresolved_central':3}

def load_independent_index(record):
    p=Path(record['path'])
    if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Independent position evidence hash mismatch')
    with np.load(p,allow_pickle=False) as a:
        rows=a['source_row'].copy();codes=a['independent_position_code'].copy()
    if (rows.ndim!=1 or codes.shape!=rows.shape or not np.issubdtype(rows.dtype,np.integer)
        or np.any(rows<0) or np.any(np.diff(rows)<=0) or not np.isin(codes,[1,2,3]).all()):
        raise ValueError('Invalid independent-position index')
    return rows,codes

def independent_status(source_rows,index):
    """Only exact source-row matches get evidence; never promote a whole population."""
    query=np.asarray(source_rows,dtype=np.int64);out=np.zeros(query.shape,dtype=np.uint8)
    if index is None:return out
    rows,codes=index
    if len(rows)==0:return out
    loc=np.searchsorted(rows,query);valid=loc<len(rows)
    match=np.zeros(query.shape,dtype=bool);match[valid]=rows[loc[valid]]==query[valid]
    out[match]=codes[loc[match]]
    return out


def load_ambiguity_rows(record):
    p=Path(record['path'])
    if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('Independent evidence hash mismatch')
    with np.load(p,allow_pickle=False) as a:
        rows=a['ambiguous_source_row'].copy() if 'ambiguous_source_row' in a else np.array([],dtype=np.int64)
    if rows.ndim!=1 or not np.issubdtype(rows.dtype,np.integer) or np.any(rows<0) or np.any(np.diff(rows)<=0):
        raise ValueError('Invalid ambiguous identity index')
    return rows


def ambiguity_status(source_rows, ambiguous_rows):
    rows=np.asarray(source_rows,dtype=np.int64)
    if ambiguous_rows is None:return np.zeros(rows.shape,dtype=np.uint8)
    return independent_status(rows,(ambiguous_rows,np.ones(len(ambiguous_rows),dtype=np.uint8)))
