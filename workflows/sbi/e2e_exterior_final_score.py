"""Frozen full-cap evaluation, with phase authority before any truth access."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi import e2e_footprint_tidal_operator as tidal
from workflows.sbi import e2e_exterior_campaign as campaign


@lru_cache(maxsize=16)
def frozen(phase, path):
    # Reject the sealed reserve even if somebody edits a manifest to include it.
    if phase not in ('ph004', 'ph005', 'ph006', 'ph012', 'ph013', 'ph014', 'ph015', 'ph016', 'ph017', 'ph018', 'ph019'):
        raise PermissionError('Not an authorized final evaluation phase')
    record = json.loads(path.read_text())
    if record.get('schema') != 'e2e-exterior-selection-freeze-v1' or phase not in record['final_phases']:
        raise PermissionError('No frozen evaluation authority')
    for name, sha in record['sources'].items():
        if c.sha256(c.REPO/name) != sha:
            raise ValueError('Frozen evaluation code changed')
    if c.sha256(record['prior']['path']) != record['prior']['sha256']:
        raise ValueError('Frozen prior changed')
    return record


def truth(phase, pair_id, freeze):
    frozen(phase, freeze)
    if phase in ('ph004', 'ph005', 'ph006'):
        from workflows.sbi import e2e_exterior_legacy as legacy
        receipt_path = legacy.ROOT/phase/'TARGETS.json'
        receipt = legacy.receipt(receipt_path, phase, payload=False)
        item = next(x for x in receipt['outputs'] if x['pair_id'] == pair_id)
        source = legacy.checked_path(item['path'], phase)
        if c.sha256(source) != item['sha256']:
            raise ValueError('Legacy truth changed')
        with np.load(source) as f:
            values = {key: f[key] for key in f.files}
    else:
        from workflows.sbi.e2e_coupled_product_audit import read_target
        receipt_path = coord.ROOT/'targets'/phase/(pair_id+'.json')
        receipt = coord.verify_receipt(receipt_path, payload=False)
        if receipt['phase'] != phase or receipt['pair_id'] != pair_id or len(receipt['outputs']) != 1:
            raise ValueError('Wrong final target identity')
        item = receipt['outputs'][0]; source = c.guarded(item['path'], phase)
        if source.parent != receipt_path.parent.resolve() or c.sha256(source) != item['sha256']:
            raise ValueError('Final truth lineage changed')
        values = read_target(source, phase, pair_id)
    return values, c.file_record(receipt_path, content_hash=True)


def geometry(phase, freeze):
    frozen(phase, freeze)
    if phase in ('ph004', 'ph005', 'ph006'):
        from workflows.sbi import e2e_exterior_legacy as legacy
        path = legacy.ROOT/phase/'GEOMETRY.json'
        return legacy.receipt(path, phase, payload=False)['pairs']
    return coord.verify_receipt(coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json', payload=False)['pairs']


def score(phase, sample_root, freeze, out):
    c.require_compute(); selected = frozen(phase, freeze)
    from workflows.sbi.e2e_global_mock_evaluate import metric_summary, density_statistics, controls
    from workflows.sbi.e2e_exterior_metrics import joint_rank, distance_masks
    from workflows.sbi.e2e_vdm_context_metrics import fair_energy, variogram_score
    rows = geometry(phase, freeze)
    cases, input_receipts, cap_features = [], [], {}
    for cap in ('NGC', 'SGC'):
        mass_vectors, mass_targets, tidal_vectors, tidal_targets = [], [], [], []
        records = []
        for first in range(0, selected['draws'], 4):
            path = sample_root/phase/cap/f'd{first:03d}'/'COMPLETE.json'
            record = json.loads(path.read_text())
            if (record['phase'] != phase or record['cap'] != cap or record['first'] != first
                    or record['draws'] != 4 or record['method'] != selected['method'] or not record['passed']):
                raise ValueError('Incomplete or mismatched full-cap posterior ensemble')
            binding_path = path.parent/'BINDING.json'
            if c.sha256(binding_path) != record['binding']:
                raise ValueError('Changed full-cap binding')
            binding = json.loads(binding_path.read_text())
            if binding['freeze_sha256'] != c.sha256(freeze):
                raise ValueError('Prediction was not made under the final freeze')
            if c.sha256(record['output']['path']) != record['output']['sha256']:
                raise ValueError('Changed full-cap output')
            records.append(record); input_receipts.append(c.file_record(path, content_hash=True))
        for row in (v for v in rows if v['cap'] == cap):
            densities, eigenvalues = [], []
            support = None
            for record in records:
                with h5py.File(record['output']['path']) as f:
                    origin = np.asarray(f.attrs['origin_mpc_h'])
                    cell = float(f.attrs['cell_mpc_h'])
                    padding = int(f.attrs['padding_cells'])
                    raw_start = np.asarray(row['center'])-16
                    # Origins are lower raw-cell edges in the observation
                    # convention. Both grids use the identical two-cell average.
                    position = np.asarray(row['grid']['origin_mpc_h'])+raw_start*3.383
                    start_float = (position-origin)/cell
                    start = np.rint(start_float).astype(int)
                    if cell != 6.766 or np.max(abs(start_float-start)) > 1e-8:
                        raise ValueError('Full-cap and reference voxel grids do not align')
                    shape = np.array([32, 16, 16]); stop = start+shape
                    if np.any(start < 0) or np.any(stop > np.array(f['support'].shape)):
                        raise ValueError('Registered science core missing from full cap')
                    cut = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
                    widecut = tuple(slice(int(a+padding), int(b+padding)) for a, b in zip(start, stop))
                    local_support = f['support'][cut].reshape(2, 16, 16, 16)
                    if support is not None and not np.array_equal(support, local_support):
                        raise ValueError('Full-cap support changed between draws')
                    support = local_support
                    densities.append(f['rho_completed'][(slice(None), *widecut)].reshape(4, 2, 16, 16, 16)-1)
                    eigenvalues.append(f['eigenvalues'][(slice(None), *cut, slice(None))].reshape(4, 2, 16, 16, 16, 3))
            draws = np.concatenate(densities); eig = np.concatenate(eigenvalues)
            values, source = truth(phase, row['pair_id'], freeze)
            density = targets.owned_core_arrays(values['rho_joint'])-1
            eigen = tidal.eigensystem(values['fullbox_tensor_cores'])
            mass, mass_truth = draws.mean((2, 3, 4)), density.mean((1, 2, 3))
            regional, regional_truth = eig.mean((2, 3, 4)).reshape(len(eig), 6), eigen.mean((1, 2, 3)).reshape(6)
            mass_vectors.append(mass); mass_targets.append(mass_truth)
            tidal_vectors.append(regional); tidal_targets.append(regional_truth)
            metrics = dict(density=metric_summary(draws, density),
                           tidal=[metric_summary(eig[..., j], eigen[..., j]) for j in range(3)],
                           core_mass=metric_summary(mass, mass_truth),
                           joint_core_mass=joint_rank(mass, mass_truth),
                           joint_tidal_regional=joint_rank(regional, regional_truth),
                           core_mass_energy=float(fair_energy(mass, mass_truth)),
                           core_mass_variogram=variogram_score(mass, mass_truth))
            # Boundary distances need the complete local support neighbourhood,
            # not an artificial edge at each cropped science core. Read a joint
            # context surrounding both cores from the same full-cap support.
            with h5py.File(records[0]['output']['path']) as f:
                joint_start = start-np.array([16, 16, 16])
                joint_stop = joint_start+np.array([64, 48, 48])
                if np.any(joint_start < 0) or np.any(joint_stop > f['support'].shape):
                    raise ValueError('Boundary context missing from full cap')
                context = f['support'][tuple(slice(int(a), int(b)) for a, b in zip(joint_start, joint_stop))]
            masks = distance_masks(context)
            metrics['distance_splits'] = {name: dict(cells=int(mask.sum()),
                density=metric_summary(draws[:, mask], density[mask]),
                tidal=[metric_summary(eig[..., j][:, mask], eigen[..., j][mask]) for j in range(3)])
                for name, mask in masks.items() if mask.any()}
            for threshold in (0., .2):
                labels = (eig > threshold).sum(-1)
                probabilities = np.stack([(labels == j).mean(0) for j in range(4)], axis=-1)
                actual = (eigen > threshold).sum(-1)
                metrics[f'brier_{threshold}'] = float(np.mean(np.sum((probabilities-np.eye(4)[actual])**2, axis=-1)))
            cases.append(dict(pair_id=row['pair_id'], cap=cap, shell=row['shell'],
                              support_stratum=row['support_stratum'], metrics=metrics, truth_source=source))
            print(json.dumps(dict(phase=phase, scored=row['pair_id'],
                                  tidal90=[v['coverage']['0.9'] for v in metrics['tidal']])), flush=True)
        cap_features[cap] = dict(mass=np.concatenate(mass_vectors, axis=1),
                                 mass_truth=np.concatenate(mass_targets),
                                 tidal=np.concatenate(tidal_vectors, axis=1),
                                 tidal_truth=np.concatenate(tidal_targets))
    joint = {}
    for name in ('NGC', 'SGC', 'both_caps_independence_approximation'):
        selected_caps = ('NGC', 'SGC') if name.startswith('both_') else (name,)
        joint[name] = {}
        for feature in ('mass', 'tidal'):
            draws = np.concatenate([cap_features[v][feature] for v in selected_caps], axis=1)
            actual = np.concatenate([cap_features[v][feature+'_truth'] for v in selected_caps])
            shuffled = draws.copy()
            # Independently permute each registered pair's regional vector,
            # retaining its one-region marginal but removing inter-region links.
            block = 2 if feature == 'mass' else 6
            rng = np.random.default_rng(741)
            for lo in range(block, draws.shape[1], block):
                shuffled[:, lo:lo+block] = draws[rng.permutation(len(draws)), lo:lo+block]
            joint[name][feature] = dict(rank=joint_rank(draws, actual),
                energy=float(fair_energy(draws, actual)), variogram=variogram_score(draws, actual),
                shuffled_energy=float(fair_energy(shuffled, actual)),
                shuffled_variogram=variogram_score(shuffled, actual))
    out.mkdir(parents=True, exist_ok=False)
    c.atomic_json(out/'SCORES.json', dict(phase=phase, cases=cases, inputs=input_receipts,
                  freeze=c.file_record(freeze, content_hash=True), controls=controls(selected['draws']),
                  joint_fullcap_summaries=joint,
                  posterior_qualified=False, panel_complete=len(cases) == 16,
                  joint_scope='capwise modular posterior; no joint two-cap qualification'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', required=True)
    p.add_argument('--samples', type=Path, required=True); p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args(); score(a.phase, a.samples, a.freeze, a.out)
