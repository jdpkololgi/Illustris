"""Materialize corrected galaxy eligibility without changing phase/fold roles."""
import argparse
import os
from pathlib import Path

import numpy as np

from workflows.sbi import prepare_p12a_lowz_inputs as p12
from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi.p12a_galaxy_identity import identity_arrays
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def run(phase):
    require_compute()
    plan, _, policy = p12.load_plan()
    lowz.phase_guard(phase, policy)
    import fitsio
    from workflows.abacus_tweb.galaxy_position_mapping import comoving_distance
    root = Path(plan['output_root'])/phase
    marker = root/'galaxy_identity_v1/IDENTITY_COMPLETE.json'
    if marker.parent.exists():
        raise FileExistsError(marker.parent)
    p1_path = root/'p1_canonical/manifest.json'
    p1 = p12.read(p1_path)
    for path, expected in ((p1['parent'], p1['parent_sha256']), (p1['index'], p1['index_sha256'])):
        if sha256(path) != expected:
            raise ValueError('Identity source changed')
    table = fitsio.read(p1['parent'], columns=['TARGETID','SOURCE_ROW','RA','DEC','Z_COSMO',
                        'FILE_NUM','BOX_INDEX','HALO_INDEX','RES','CEN','HALO_MASS'])
    with np.load(p1['index'], allow_pickle=False) as index:
        active = index['active'].astype(bool) & index['valid_target'].astype(bool)
        np.testing.assert_array_equal(index['targetid'], table['TARGETID'])
    ra, dec = np.deg2rad(table['RA']), np.deg2rad(table['DEC'])
    unit = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
    xyz = unit*comoving_distance(table['Z_COSMO'])[:,None]
    arrays, audit = identity_arrays(table, active, xyz)
    marker.parent.mkdir(parents=True, exist_ok=False)
    output = marker.parent/'identity.npz'
    with output.open('xb') as stream:
        np.savez_compressed(stream, parent_node_id=np.arange(len(table),dtype=np.int64),
                            source_row=table['SOURCE_ROW'], **arrays)
    save_json(marker, dict(schema='p12a-galaxy-identity-v1', phase=phase, job=os.environ['SLURM_JOB_ID'],
        p1_manifest=str(p1_path), p1_sha256=sha256(p1_path), output=str(output), output_sha256=sha256(output),
        builder_sha256=sha256(__file__), identity_source_sha256=sha256(Path(__file__).with_name('p12a_galaxy_identity.py')),
        phase_roles_changed=False, model_inputs_ready=False, **audit))
    print('GALAXY_IDENTITY_PASS', phase, audit, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phases', nargs='+', required=True)
    for phase in p.parse_args().phases:
        run(phase)
