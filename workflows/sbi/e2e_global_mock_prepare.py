"""Registered ph016/ph017 overlapping-window calibration inputs; no fitting."""
import argparse
import json
import os
from pathlib import Path
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_conditions as conditions
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi import e2e_loa_field_conditions as adapter
from workflows.sbi.e2e_cfm_field_gallery import compare_channels
from workflows.sbi.e2e_coupled_product_audit import read_target
from workflows.abacus_tweb import p3br_build_random_response as response_ops

PHASES = ('ph016', 'ph017')


def guard(phase):
    if phase not in PHASES:
        raise PermissionError('Only already-exposed ph016/ph017 are registered here')


def read_truth(phase, pair_id):
    guard(phase)  # Before target I/O, including receipts.
    directory = coord.ROOT / 'targets' / phase
    receipt_path = directory / f'{pair_id}.json'
    receipt = coord.verify_receipt(receipt_path, payload=False)
    if receipt['phase'] != phase or receipt['pair_id'] != pair_id or len(receipt['outputs']) != 1:
        raise ValueError('Target identity mismatch')
    item = receipt['outputs'][0]
    path = c.guarded(item['path'], phase)
    if path.parent != directory.resolve() or c.sha256(path) != item['sha256']:
        raise ValueError('Target payload mismatch')
    return read_target(path, phase, pair_id), c.file_record(receipt_path, content_hash=True)


def prepare(out):
    c.require_compute()
    out.mkdir(parents=True, exist_ok=False)
    source_records, cases = [], []
    selection_path = Path(c.config()['observation']['fixed_selection_manifest'])
    selection = json.loads(selection_path.read_text())
    source_records.append(c.file_record(selection_path, content_hash=True))
    for phase in PHASES:
        guard(phase)
        gp = coord.ROOT / 'geometry' / phase / 'GEOMETRY_COMPLETE.json'
        geo = coord.verify_receipt(gp, payload=False)
        geometry.verify_rows(geo['pairs'], phase)
        source_records.append(c.file_record(gp, content_hash=True))
        ap = c.ROOT / 'observations' / phase / 'angular' / 'ANGULAR_COMPLETE.json'
        ar = c.verify_receipt(ap)
        source_records.append(c.file_record(ap, content_hash=True))
        with np.load(ar['outputs'][0]['path']) as f:
            angular = {k: f[k] for k in f.files}
        for cap in ('NGC', 'SGC'):
            rp = coord.ROOT / 'observations' / phase / f'{cap}_COMPLETE.json'
            wp = coord.ROOT / 'conditions' / phase / f'{cap}_FACTOR4.json'
            raw = coord.verify_receipt(rp)
            coarse = coord.verify_receipt(wp)
            if coarse['binding']['response_receipt_sha256'] != c.sha256(rp):
                raise ValueError('Coarse/raw observation provenance mismatch')
            if raw['selection']['sha256'] != c.sha256(selection_path):
                raise ValueError('Selection changed')
            source_records.extend(c.file_record(p, content_hash=True) for p in (rp, wp))
            support = angular['support'].astype(bool) & (angular['domain']//2 == (cap == 'NGC'))
            response = (support, angular['angular_response'],
                        response_ops.angular_boundary_distance(support), selection)
            with h5py.File(raw['outputs'][0]['path'], 'r') as r, h5py.File(coarse['outputs'][0]['path'], 'r') as w:
                for row in (v for v in geo['pairs'] if v['cap'] == cap):
                    pair_id = row['pair_id']
                    mid = np.asarray(row['center']) + [16, 0, 0]
                    directory = out / pair_id
                    directory.mkdir()
                    arrays = [adapter.build(row['grid'], mid + [64*i, 0, 0], cap,
                                            r['counts'], w, response) for i in range(2)]
                    original = conditions.crop_context(conditions.load_pair(phase, pair_id), phase)
                    replay = compare_channels(arrays[0], original)
                    if not all(v['passed'] for v in replay):
                        raise ValueError('Left condition replay failed')
                    # Exact same physical observations on all overlapping grid cells.
                    for name, shift in [('joint', 32), ('wide', 8)]:
                        if not np.allclose(arrays[0][name][:, shift:], arrays[1][name][:, :-shift],
                                           rtol=2e-5, atol=2e-5):
                            raise ValueError('Observation overlap mismatch: ' + name)
                    files = []
                    for i, values in enumerate(arrays):
                        path = directory / f'conditions_{i}.npz'
                        np.savez_compressed(path, **values)
                        files.append(c.file_record(path, content_hash=True))
                    truth, source = read_truth(phase, pair_id)
                    path = directory / 'truth.npz'
                    np.savez_compressed(path, **truth)
                    case = dict(phase=phase, pair_id=pair_id, cap=cap, shell=row['shell'],
                                support_stratum=row['support_stratum'], midpoint=mid.tolist(),
                                conditions=files, truth=c.file_record(path, content_hash=True),
                                truth_source=source, replay=replay,
                                seam_support=[float(arrays[0]['support'][x:x+8,16:32,16:32].mean())
                                              for x in (40,48)])
                    c.atomic_json(directory / 'PREPARED.json', case)
                    cases.append(case)
                    print('prepared', pair_id, flush=True)
    if len(cases) != 32:
        raise ValueError('Incomplete registered panel')
    record = dict(cases=cases, sources=source_records, phases=list(PHASES),
                  confirmation_018_019_access=False, job=os.environ['SLURM_JOB_ID'],
                  script_sha256=c.sha256(__file__), passed=True)
    c.atomic_json(out / 'PREPARED.json', record)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    prepare(p.parse_args().out)
