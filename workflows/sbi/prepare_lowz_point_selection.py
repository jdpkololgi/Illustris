"""Versioned point-count/analytic-volume selection CANDIDATES for P12 and CFM.

No dependence on host IDs, truth validity, spatial folds or old fitted curves.
P12 uses Planck18 Mpc; CFM uses pinned DESI/Abacus Mpc/h. These candidates are
not interchangeable and must not feed the old Gaussian-apodized expectation
writer. A matching CIC forward response is a separate prerequisite.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as cfm
from workflows.sbi import prepare_p12a_lowz_inputs as p12
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def shell_volumes(solid_angle, radii):
    radii = np.asarray(radii, dtype=float)
    if (radii.ndim != 1 or len(radii) < 2 or not np.isfinite(radii).all()
            or radii[0] < 0 or np.any(np.diff(radii) <= 0)
            or not np.isfinite(solid_angle) or solid_angle <= 0):
        raise ValueError('Invalid spherical-shell volume')
    return solid_angle*np.diff(radii**3)/3.


def fit(records, training, edges, unit, omitted=None):
    if unit not in ('Mpc^-3', '(Mpc/h)^-3') or len(training) != len(set(training)):
        raise ValueError('Invalid units or repeated training phases')
    if omitted is not None and omitted not in training:
        raise ValueError('Only a training phase can be omitted')
    phases = [p for p in training if p != omitted]
    if set(records) != set(phases) or not phases:
        raise ValueError('Exactly the full/OOF training phases are required')
    count = np.zeros((2, len(edges)-1))
    volume = np.zeros_like(count)
    for phase in phases:
        record = records[phase]
        if (record['phase'] != phase or record['density_unit'] != unit or not record['label_free']
                or not np.array_equal(record['bin_edges'], edges)):
            raise ValueError('Point-statistics identity or unit mismatch')
        n, v = np.asarray(record['counts'], float), np.asarray(record['volume'], float)
        if (n.shape != count.shape or v.shape != count.shape or not np.isfinite(n).all()
                or not np.isfinite(v).all() or (n < 0).any() or (v < 0).any()):
            raise ValueError('Invalid counts/volumes')
        count += n
        volume += v
    valid = (volume > 0) & (count >= 20)
    edges = np.asarray(edges)
    science = (edges[:-1] >= .01) & (edges[1:] <= .55)
    if not valid[:, science].all():
        raise ValueError('Unpopulated science bin; no extrapolation')
    density = np.divide(count, volume, out=np.zeros_like(count), where=valid)
    return dict(schema='lowz-point-selection-candidate-v1', fit_phases=phases,
        omitted_phase=omitted, bin_edges=edges.tolist(), density_unit=unit,
        counts=count.tolist(), volume=volume.tolist(), density=density.tolist(), valid=valid.tolist(),
        model_inputs_ready=False, matched_cic_response_required=True,
        estimator='supported observed point counts / analytic angular-response-weighted shell volume')


def statistics(pipeline, phase, plan, policy, source, destination):
    import fitsio
    import healpy as hp
    from workflows.abacus_tweb.p3br_build_random_response import galactic_cap
    cfm.phase_guard(phase, policy)
    edges = np.round(np.linspace(0., .6, 121), 12)
    if pipeline == 'p12':
        from astropy.cosmology import Planck18
        root = Path(plan['output_root'])/phase
        raw = p12.read(root/'RAW_PREPARATION_COMPLETE.json')
        fields_path = root/'p3_fields/field_manifest.json'
        if raw['phase'] != phase or not raw['pass_raw_preparation'] or sha256(fields_path) != raw['fields_sha256']:
            raise ValueError('Unqualified P12 raw fields')
        fields = p12.read(fields_path)
        p1_path = root/'p1_canonical/manifest.json'
        if sha256(p1_path) != fields['p1_manifest_sha256']:
            raise ValueError('P12 parent manifest changed')
        parent = p12.read(p1_path)
        p12.require_matching_domains(plan, fields, parent)
        path = Path(parent['parent'])
        angular_path = Path(fields['angular_support']['source'])
        if sha256(path) != parent['parent_sha256'] or sha256(angular_path) != fields['angular_support']['sha256']:
            raise ValueError('P12 parent/random payload changed')
        bindings = dict(parent_manifest=str(p1_path), parent_manifest_sha256=sha256(p1_path),
                        parent=str(path), parent_sha256=parent['parent_sha256'],
                        angular=str(angular_path), angular_sha256=fields['angular_support']['sha256'])
        radii = Planck18.comoving_distance(edges).value
        unit = 'Mpc^-3'
    elif pipeline == 'cfm':
        from workflows.sbi import e2e_coupled_coordinates as coord
        path, observed = cfm.source_payload(Path(source['root']), phase, 'OBSERVED_COMPLETE.json',
                                           plan['source_campaign']['sha256'], True)
        angular_path, angular = cfm.source_payload(Path(source['root']), phase, 'angular/ANGULAR_COMPLETE.json',
                                                  plan['source_campaign']['sha256'])
        bindings = dict(observed=observed, angular=angular)
        radii = coord.radius_mpc_h(edges)
        unit = '(Mpc/h)^-3'
    else:
        raise ValueError('Unknown pipeline')
    with np.load(angular_path, allow_pickle=False) as arrays:
        angular = {k: arrays[k] for k in ('support', 'domain', 'angular_response')}
    if not np.isfinite(angular['angular_response']).all() or (angular['angular_response'] < 0).any():
        raise ValueError('Invalid angular response')
    # P12 parent files also contain labels, but they are neither read nor used.
    rows = fitsio.read(path, columns=['RA', 'DEC', 'Z'])
    if not all(np.isfinite(rows[k]).all() for k in ('RA', 'DEC', 'Z')):
        raise ValueError('Nonfinite observed coordinate')
    radial_plan = dict(context_redshift=[0., .6], excluded_sentinel_redshift=[.585, .595])
    context = cfm.radial_mask(rows['Z'], radial_plan)
    rows = rows[context]
    caps = galactic_cap(rows['RA'], rows['DEC'])
    pixels = hp.ang2pix(256, rows['RA'], rows['DEC'], lonlat=True)
    counts, volumes, excluded = [], [], []
    valid_bins = cfm.radial_mask((edges[:-1]+edges[1:])/2, radial_plan)
    for cap in range(2):
        support = angular['support'].astype(bool) & (angular['domain']//2 == cap)
        selected = caps == cap
        inside = support[pixels]
        counts.append(np.histogram(rows['Z'][selected & inside], edges)[0].tolist())
        excluded.append(np.histogram(rows['Z'][selected & ~inside], edges)[0].tolist())
        angle = float(angular['angular_response'][support].astype(float).sum()*hp.nside2pixarea(256))
        volumes.append(np.where(valid_bins, shell_volumes(angle, radii), 0.).tolist())
    record = dict(phase=phase, pipeline=pipeline, density_unit=unit, label_free=True,
        bin_edges=edges.tolist(), radius_edges=radii.tolist(), counts=counts, volume=volumes, unsupported_point_counts=excluded,
        context_rows=len(rows), excluded_context_rows=int((~context).sum()),
        sources=bindings, coordinates='Planck18 observer Mpc' if pipeline == 'p12' else 'pinned DESI/Abacus Mpc/h',
        model_inputs_ready=False)
    save_json(destination/(phase + '.json'), record)
    print(pipeline, phase, 'point statistics complete', flush=True)
    return record


def run(pipeline, workers=2):
    require_compute()
    if pipeline not in ('p12', 'cfm') or workers not in (1, 2):
        raise ValueError('Unregistered point-selection resources')
    if pipeline == 'p12':
        plan, source, policy = p12.load_plan()
    else:
        plan, policy, source = cfm.load_plan()
    destination = Path(plan['output_root'])/'point_selection_candidates_v1'
    destination.mkdir(exist_ok=False)
    phases = policy['training']+policy['development_and_calibration']
    implementation = sha256(__file__)
    bindings = cfm.code_bindings()
    bindings['workflows/sbi/prepare_p12a_lowz_inputs.py'] = sha256(p12.__file__)
    plan_path = p12.PLAN if pipeline == 'p12' else cfm.DEFAULT_PLAN
    plan_hash = sha256(plan_path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(statistics, pipeline, p, plan, policy, source, destination) for p in phases]
        records = {row['phase']: row for job in jobs for row in [job.result()]}
    edges = np.round(np.linspace(0., .6, 121), 12)
    unit = 'Mpc^-3' if pipeline == 'p12' else '(Mpc/h)^-3'
    fits = {}
    for omitted in [None]+policy['training']:
        selected = {p: records[p] for p in policy['training'] if p != omitted}
        result = fit(selected, policy['training'], edges, unit, omitted)
        result.update(pipeline=pipeline, statistics_sha256={p: sha256(destination/(p+'.json')) for p in selected},
                      source_sha256=implementation)
        name = 'full' if omitted is None else 'oof_'+omitted
        path = destination/(name+'.json')
        save_json(path, result)
        fits[name] = sha256(path)
    if (sha256(__file__) != implementation or sha256(plan_path) != plan_hash
            or any(sha256(p12.REPO/path) != expected for path, expected in bindings.items())):
        raise ValueError('Point-selection implementation changed')
    save_json(destination/'CANDIDATES_COMPLETE.json', dict(pipeline=pipeline, job=os.environ['SLURM_JOB_ID'],
        phases=phases, statistics={p: sha256(destination/(p+'.json')) for p in phases},
        fits=fits, source_sha256=implementation, source_plan=str(plan_path), source_plan_sha256=plan_hash,
        code_bindings=bindings, policy=policy, density_unit=unit,
        development_used_for_fitting=False, model_inputs_ready=False,
        qualification='candidate point-selection fits; matched CIC response and sensitivity checks still required'))
    print('POINT_SELECTION_CANDIDATES_COMPLETE', pipeline, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', choices=('p12', 'cfm'), required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run(args.pipeline, args.workers)
