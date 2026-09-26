"""Read-only ph000 import closure and small P12 encoder lineage verification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from workflows.abacus_tweb import p12a_coordinate_sample_audit as a

BASE = Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase')
TRAIN = ['ph000', 'ph002', 'ph003', 'ph004', 'ph005']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def lineage():
    records = []
    for phase in TRAIN + ['ph006']:
        summary, sr = a.read_json(BASE / f'p12_oof_summaries/{phase}/OOF_SUMMARY_COMPLETE.json')
        checkpoint = Path(summary['checkpoint'])
        run, rr = a.read_json(checkpoint.parent / 'run_manifest.json')
        final, fr = a.read_json(checkpoint.parent / 'arm_a_summary.json')
        contract = Path(summary['contract_root'])
        paths = {'checkpoint': checkpoint, 'target_scaler': Path(run['target_scaler']),
                 'field_transform': contract/'transforms/field/field_transform.json',
                 'training_ready_marker': Path(run['training_ready_marker'])}
        expected_train = [p for p in TRAIN if p != phase]
        checks = dict(training_membership=summary['training_phases'] == expected_train,
                      run_training_membership=run['training_phases'] == expected_train,
                      omitted_phase_absent=phase not in run['training_phases'],
                      selected_epoch_twenty=final['best_epoch'] == 20,
                      summary_marked_complete=summary['pass'] is True)
        files = {}
        for key, path in paths.items():
            files[key] = a.record(path, small=True)
            checks[key+'_hash'] = files[key]['sha256'] == (summary if key=='checkpoint' else run)[key+'_sha256']
        ready, _ = a.read_json(paths['training_ready_marker'])
        # Cross-fit loader markers record roles but omit an adapter hash. Bind
        # their current adapter bytes to the separately pinned full-fit inventory.
        crossfit = 'adapter_inventory' not in ready
        authority = ready
        if crossfit:
            authority, ar = a.read_json(BASE/'training_contract/TRAINING_LOADER_READY.json')
            fullrun, _ = a.read_json(BASE/'arm_a_training/arm_a_r0_v1/unet/seed_42/run_manifest.json')
            checks['fullfit_inventory_authority'] = ar['sha256'] == fullrun['training_ready_marker_sha256']
        invpath = Path(authority['adapter_inventory'])
        inv, ir = a.read_json(invpath)
        checks['adapter_inventory_binding'] = ir['sha256'] == authority['adapter_inventory_sha256']
        adapters = {}
        for p in expected_train + [phase]:
            row = inv['phases'][p]
            current_path = contract/f'adapters/{p}/field/adapter_manifest.json'
            manifest, mr = a.read_json(current_path)
            checks[p+'_adapter_hash'] = mr['sha256'] == row['field_manifest_sha256']
            field, field_rec = a.read_json(manifest['p3_manifest'])
            checks[p+'_p3_hash'] = field_rec['sha256'] == manifest['p3_manifest_sha256']
            adapters[p] = dict(adapter=mr, p3=field_rec, recorded_points=manifest['points'])
        records.append(dict(phase=phase, summary=sr, run=rr, final=fr, files=files,
                            adapter_inventory=ir, adapters=adapters, checks=checks,
                            pass_checks=all(checks.values()),
                            crossfit_adapter_binding=('current bytes match pinned fullfit inventory; '
                              'crossfit loader does not itself hash adapters' if crossfit else 'direct'),
                            selection_caveat='omitted phase monitored; selected epoch is fixed terminal epoch20',
                            summary_arrays_rehashed=False, field_payloads_rehashed=False))
    return dict(schema='p12a-small-encoder-lineage-v1', records=records,
                pass_checks=all(r['pass_checks'] for r in records),
                ready_for_desi_canary=False,
                scope='live small checkpoint/transform/receipt bindings, not full response qualification')


def ph000():
    a.require_compute()
    source_before = a.record(__file__,small=True)
    helper_before = a.record(a.__file__,small=True)
    import fitsio
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import Planck18
    root = BASE / 'ph000'
    manifest, mr = a.read_json(root/'p1_canonical/manifest.json')
    if manifest['phase'] != 'ph000' or manifest['counts']['total'] != 9538254:
        raise ValueError('unexpected legacy manifest')
    original = Path(manifest['parent'])
    imported = root/'catalogues/observed/ph000_bgs_bright_full_observed_with_tweb.fits'
    points_source = Path(manifest['points'])
    points_imported = root/'p1_canonical/points.npy'
    paths = [original, imported, points_source, points_imported]
    before = [a.record(p) for p in paths]
    hashes = {str(p): sha(p) for p in paths}
    checks = dict(original_catalogue_hash=hashes[str(original)] == manifest['parent_sha256'],
                  imported_catalogue_hash=hashes[str(imported)] == manifest['parent_sha256'],
                  imported_points_hash=hashes[str(points_imported)] == hashes[str(points_source)])
    names = ['TARGETID','RA','DEC','Z','FILE_NUM','HALO_INDEX','BOX_INDEX','LAMBDA1','LAMBDA2','LAMBDA3','CWEB']
    rows = np.unique(np.linspace(0, manifest['counts']['total']-1, 16384, dtype=np.int64))
    with fitsio.FITS(imported) as f:
        if f[1].get_nrows() != manifest['counts']['total']:
            raise ValueError('imported catalogue row count changed')
        data = f[1].read(rows=rows, columns=names)
    sky = SkyCoord(ra=data['RA']*u.deg, dec=data['DEC']*u.deg,
                   distance=Planck18.comoving_distance(data['Z']), frame='icrs')
    expected = sky.cartesian.xyz.to_value(u.Mpc).T
    caps = (sky.galactic.b.deg > 0).astype(int)
    points = np.load(points_imported, mmap_mode='r', allow_pickle=False)
    if points.shape != (len(points),4) or len(points) != manifest['counts']['total']:
        raise ValueError('imported point shape mismatch')
    sampled = np.asarray(points[rows])
    error = np.linalg.norm(sampled[:,:3]-expected,axis=1)
    checks['direct_planck18_replay'] = float(error.max()) <= 1e-9
    checks['cap_identity'] = bool(np.array_equal(caps,sampled[:,3]))
    selected, strata, slabs = a.choose_native_rows(data,caps)
    registry, registry_rec = a.read_json(a.ROOT/'configs/p10_phase_registry_v1.json')
    inv = dict(phase='ph000',snapshot_root=registry['path_templates']['snapshot_root'].format(phase='ph000'),
               tweb_dir=str(root/'targets/tweb/backend_optimized_ngrid_2048_rsmooth_7'))
    native = a.native_labels(inv,data,selected)
    checks.update(all_cap_shells_sampled=all(n>0 for n in strata.values()),
                  native_labels=native['labels_equal_at_float32'],native_classes=native['cweb_equal'])
    if before != [a.record(p) for p in paths]:
        raise ValueError('input identity changed during audit')
    if source_before != a.record(__file__,small=True) or helper_before != a.record(a.__file__,small=True):
        raise ValueError('audit source changed during execution')
    return dict(schema='p12a-ph000-import-closure-v1',manifest=mr,registry=registry_rec,
                source=source_before,helper_source=helper_before,
                inputs=before,live_sha256=hashes,checks=checks,pass_checks=all(checks.values()),
                sampled_rows=rows.tolist(),sampled_targetids=data['TARGETID'].tolist(),
                sampled_data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                max_replay_error_mpc=float(error.max()),tolerance_mpc=1e-9,
                strata=strata,slabs=slabs,native=native,ready_for_desi_canary=False,
                parent_join='not assumed: original/imported catalogue contents hash-verified')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['ph000','lineage'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    report=ph000() if args.mode=='ph000' else lineage()
    with args.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(args.output),pass_checks=report['pass_checks'])))
    raise SystemExit(0 if report['pass_checks'] else 1)
