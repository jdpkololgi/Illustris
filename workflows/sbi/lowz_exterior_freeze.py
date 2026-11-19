"""Fit and bind the October spatial completion law to the corrected campaign."""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_spatial_exterior as spatial

DATA_SHA = '158fce06f64528f7d6f2f901934d5ce249388e531a6a08c203bd745717caaf31'
OLD_SHA = '6bd49651231448443fe171ad7699032fee7151dcf34a33caf7952484b0745232'


def inputs():
    if c.sha256(c.CONFIG) != DATA_SHA or len(set(c.TRAIN)) != 20:
        raise ValueError('Wrong input campaign')
    historical = c.ROOT.parent/'coherent_exterior_20261002_v1/SELECTION_FREEZE.json'
    if c.sha256(historical) != OLD_SHA:
        raise ValueError('Historical selection changed')
    selected = json.loads(historical.read_text())
    if (selected['method'], selected['fullcap_padding']) != ('learned', 32):
        raise ValueError('Wrong exterior prescription')
    sources = {}
    for phase in c.TRAIN:
        if c.ROLES[phase] != 'train':
            raise PermissionError(phase)
        g = json.loads((coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json').read_text())
        ids = [r['pair_id'] for r in g['pairs']]
        if not g['pass'] or g['phase'] != phase or len(ids) != 128 or len(set(ids)) != 128:
            raise ValueError('Invalid training geometry: '+phase)
        rows = []
        for ident in ids:
            path = c.guarded(coord.ROOT/'targets'/phase/(ident+'.json'), phase)
            rec = json.loads(path.read_text())
            if not rec['pass'] or rec['phase'] != phase or rec['pair_id'] != ident or len(rec['outputs']) != 1:
                raise ValueError('Invalid training target: '+ident)
            item = rec['outputs'][0]
            target = c.guarded(item['path'], phase)
            if target.parent != path.parent or target.stat().st_size != item['bytes']:
                raise ValueError('Training target lineage: '+ident)
            rows.append((str(target),item['sha256']))
        sources[phase] = rows
    return sources


def fit():
    c.require_compute()
    sources = inputs()
    out = c.ROOT/'exterior_learned_pad32_v1'
    out.mkdir(exist_ok=False)
    moments = []
    for rows in sources.values():
        phase_moments = []
        for path, digest in rows:
            if c.sha256(path) != digest:
                raise ValueError('Training density changed: '+path)
            with h5py.File(path) as f:
                rho = f['rho_joint'][:]
            if rho.shape != (64,48,48) or not np.all(np.isfinite(rho) & (rho>0)):
                raise ValueError('Invalid training density')
            phase_moments.append(spatial.training_moments(rho))
        moments.append(phase_moments)
    prior = spatial.fit_prior(moments)
    prior.update(training_phases=list(sources), pair_counts=list(map(len,moments)),
                 sources=sources, data_sha256=DATA_SHA, spatial_source=c.sha256(spatial.__file__))
    c.atomic_json(out/'PRIOR.json', prior)
    c.atomic_json(out/'FREEZE.json', dict(method='learned', padding=32,
        historical_selection=OLD_SHA, prior_sha256=c.sha256(out/'PRIOR.json'),
        training_phases=list(sources), caps=['NGC','SGC'], data_sha256=DATA_SHA,
        runner_sha256=c.sha256(__file__), job=os.environ['SLURM_JOB_ID']))
    for name in ('PRIOR.json','FREEZE.json'):
        (out/name).chmod(0o444)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', action='store_true')
    if parser.parse_args().fit:
        fit()
    else:
        print({p:len(rows) for p,rows in inputs().items()})
