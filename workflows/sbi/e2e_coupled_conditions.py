"""Targetless condition reader: phase IDs are routing metadata, never inputs."""
import json
from pathlib import Path
import re

import h5py
import numpy as np

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_condition_products as products
from workflows.sbi import e2e_coupled_operators as op


def condition_directory(phase):
    c.phase_guard(phase)
    return coord.ROOT/'conditions'/phase


def load_pair(phase,pair_id):
    """Read only committed condition payloads, including sealed-to-scoring roles.

    This function never opens a target path and returns no phase/role feature.
    Confirmation conditions may be packaged/checked without predictive scoring.
    """
    directory=condition_directory(phase)
    if not re.fullmatch(re.escape(phase)+r'_(NGC|SGC)_s[0-3]_(interior|boundary)_\d{2}',pair_id):
        raise ValueError('invalid pair ID')
    record=coord.verify_receipt(directory/f'{pair_id}.json',payload=False)
    if (record.get('phase')!=phase or record.get('pair_id')!=pair_id
            or record.get('contains_matter') is not False or len(record['outputs'])!=1):
        raise ValueError('not a target-free condition receipt')
    file=record['outputs'][0]
    path=c.guarded(file['path'],phase)
    if path.parent!=directory.resolve() or not path.name.startswith(pair_id+'_generation_'):
        raise PermissionError('condition reader cannot follow a non-condition payload')
    if path.stat().st_size!=file['bytes'] or c.sha256(path)!=file['sha256']:
        raise ValueError('condition payload checksum mismatch')
    with h5py.File(path,'r') as saved:
        expected=dict(schema=products.SCHEMA,phase=phase,pair_id=pair_id,
                      coordinate_sha256=c.sha256(coord.CONFIG),distance_unit='Mpc/h')
        if (any(saved.attrs.get(k)!=v for k,v in expected.items())
                or bool(saved.attrs.get('contains_matter',True))
                or set(saved.keys())!={'joint','wide_extended','support'}
                or json.loads(saved.attrs['local_channels'])!=list(products.LOCAL_CHANNELS)
                or json.loads(saved.attrs['wide_channels'])!=list(products.WIDE_CHANNELS)):
            raise ValueError('condition payload contract mismatch')
        for name in saved:
            # Inspect links before resolving them: even opening an external
            # target dataset is forbidden to this condition-only reader.
            if not isinstance(saved.get(name,getlink=True),h5py.HardLink):
                raise PermissionError('external/virtual condition data prohibited')
            dataset=saved[name]
            if dataset.is_virtual or dataset.external:
                raise PermissionError('external/virtual condition data prohibited')
        arrays={name:saved[name][:] for name in saved}
    products.validate_arrays(arrays)
    return arrays


def crop_context(arrays,phase,offset=(0,0,0)):
    """All future arms share this input; offset augmentation is training-only."""
    c.phase_guard(phase); products.validate_arrays(arrays)
    cfg=op.layout(); offset=list(offset)
    if offset not in cfg['context_offsets_raw']:
        raise ValueError('unregistered context translation')
    if c.ROLES[phase]!='train' and offset!=[0,0,0]:
        raise PermissionError('held-out context translations are not registered science cases')
    start=np.asarray(cfg['wide_crop_base_start'])+np.asarray(offset)//8
    stop=start+cfg['wide_crop_shape']
    slices=tuple(slice(int(a),int(b)) for a,b in zip(start,stop))
    wide=arrays['wide_extended'][(slice(None),*slices)]
    if wide.shape!=(len(products.WIDE_CHANNELS),48,48,48):
        raise ValueError('context crop outside stored extended grid')
    # Metadata needed for physical reconstruction is deterministic from offset;
    # do not smuggle phase IDs, source names, or targets into model channels.
    return dict(joint=arrays['joint'],wide=wide,support=arrays['support'])
