"""Separately authorized legacy004/005/006 rebuild; old guards stay unchanged.

Common numerical helpers operate on supplied arrays. This module owns its own
phase validation, receipts, output root and target access. Never impersonates
an old coupled-data receipt or accesses the sealed ph001 reserve.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import re
import time
import fitsio
import h5py
import numpy as np
from scipy import fft
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_observations as obs
from workflows.sbi import e2e_coupled_condition_products as conditions
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi import e2e_coupled_pairing as pairing
from workflows.abacus_tweb import p3a_build_canonical_fields as grids
from workflows.abacus_tweb import p3br_build_random_response as responses

CONFIG = c.REPO/'configs/e2e_exterior_legacy_inputs_20261002.json'
PHASES = ('ph004', 'ph005', 'ph006')
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1/legacy_inputs')
OLD = Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase')


def guard(phase):
    if phase not in PHASES:
        raise PermissionError('Only explicitly authorized non-blind legacy replication phases')
    config = json.loads(CONFIG.read_text())
    if (config['phases'] != list(PHASES) or config['forbidden_phases'] != ['ph001']
            or config['root'] != str(ROOT) or config['model_training_allowed'] or config['model_selection_allowed']):
        raise PermissionError('Legacy rebuild authority changed')
    return config


def checked_path(path, phase, output=False):
    guard(phase)
    p = Path(path).resolve()
    found = set(re.findall(r'ph\d{3}', str(p)))
    if found and found != {phase}:
        raise PermissionError('Legacy source has a different or sealed phase')
    if output and ROOT/phase not in p.parents:
        raise PermissionError('Legacy writes must remain inside new phase root')
    return p


def record_base(phase):
    guard(phase)
    return dict(schema='e2e-exterior-legacy-receipt-v1', phase=phase,
                role='legacy_nonblind_replication', authority_sha256=c.sha256(CONFIG),
                coordinate_sha256=c.sha256(coord.CONFIG), builder_sha256=c.sha256(__file__),
                job=os.environ.get('SLURM_JOB_ID'), predictive_scores_evaluated=False)


def receipt(path, phase, payload=True):
    path = checked_path(path, phase)
    record = json.loads(path.read_text())
    if (record['schema'] != 'e2e-exterior-legacy-receipt-v1' or record['phase'] != phase
            or record['authority_sha256'] != c.sha256(CONFIG) or record.get('passed') is not True):
        raise ValueError('Unqualified legacy receipt')
    if payload:
        for item in record.get('outputs', []):
            source = checked_path(item['path'], phase)
            if c.sha256(source) != item['sha256']:
                raise ValueError('Legacy payload changed')
    return record


def observed(phase):
    c.require_compute(); guard(phase)
    directory = ROOT/phase; directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'OBSERVED.json'
    if marker.exists():
        return receipt(marker, phase)
    from workflows.abacus_tweb import p10_build_bright_parent as parent
    registry = json.loads((c.REPO/'configs/p10_phase_registry_v1.json').read_text())
    source = {key: checked_path(value.format(phase=phase, mock=int(phase[2:])), phase)
              for key, value in registry['path_templates'].items() if key in ('forfa', 'lss', 'cutsky')}
    pp = OLD/phase/'catalogues/bright_parent'/f'{phase}_bgs_bright_parent_linkage.fits.complete.json'
    old = json.loads(pp.read_text())
    if old['phase'] != phase or old['target_truth_present'] or not old['parity']['row_identity_verified']:
        raise ValueError('Legacy parent is not target-free and verified')
    parent_path = checked_path(old['output']['path'], phase)
    if c.sha256(parent_path) != old['output']['sha256']:
        raise ValueError('Legacy parent bytes changed')
    parity = parent.validate_parent_against_forfa(parent_path, source['forfa'], old['output']['rows'], 500000)
    linkage = fitsio.read(parent_path, columns=['TARGETID', 'RA', 'DEC', 'Z'])
    if not np.array_equal(linkage['TARGETID'], np.arange(1, len(linkage)+1)):
        raise ValueError('Parent IDs are not a complete sequential index')
    full = source['lss']/'BGS_BRIGHT_full_HPmapcut.dat.fits'
    output = directory/'observed_geometry.fits'
    if output.exists():
        raise ValueError('Unreceipted observed output; no overwrite')
    seen = np.zeros(len(linkage), bool); first = True; total = 0
    with fitsio.FITS(full) as src, fitsio.FITS(output, 'rw') as dst:
        n = int(src[1].get_nrows())
        for start in range(0, n, 500000):
            block = src[1].read(rows=np.arange(start, min(start+500000, n)), columns=list(pairing.REQUIRED['full']))
            block = block[pairing.successful(block)]
            if len(block) == 0:
                continue
            ids = block['TARGETID'].astype(np.int64)
            if ids.min() < 1 or ids.max() > len(linkage) or len(np.unique(ids)) != len(ids) or seen[ids-1].any():
                raise ValueError('Duplicate or invalid observed ID')
            pairing.validate_successful_join(block, linkage[ids-1]); seen[ids-1] = True
            compact = np.empty(len(block), dtype=[('TARGETID', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8')])
            for key in ('TARGETID', 'RA', 'DEC'):
                compact[key] = block[key]
            compact['Z'] = block['Z_not4clus']
            if first:
                dst.write(compact, extname='OBSERVED'); first = False
            else:
                dst[-1].append(compact)
            total += len(block)
    if first:
        raise ValueError('Empty successful catalogue')
    record = dict(record_base(phase), outputs=[c.file_record(output, content_hash=True)], rows=total,
                  sources=[c.file_record(pp, content_hash=True), c.file_record(full, content_hash=True),
                           c.file_record(source['forfa'], content_hash=True)],
                  parent_sha256=old['output']['sha256'], parity=parity,
                  exact_successful_id_sky_rsd_join=True, target_columns_present=False, passed=True)
    c.atomic_json(marker, record); print('legacy observed qualified', phase, total, flush=True)
    return record


def angular(phase):
    c.require_compute(); guard(phase)
    directory = ROOT/phase; marker = directory/'ANGULAR.json'
    if marker.exists():
        return receipt(marker, phase)
    old_path = OLD/phase/'p3b_random_response_v1/angular/randoms_n18.json'
    old = json.loads(old_path.read_text())
    if old['phase'] != phase or old['random_ids'] != list(range(18)) or old['nside'] != 256 or len(old['sources']) != 18:
        raise ValueError('Wrong legacy random panel')
    path = checked_path(old['path'], phase)
    if c.sha256(path) != old['sha256']:
        raise ValueError('Legacy angular raw-count bytes changed')
    verified = []
    for item in old['sources']:
        source = checked_path(item['path'], phase)
        current = c.file_record(source, content_hash=True)
        if current['sha256'] != item['sha256']:
            raise ValueError('Legacy random source changed')
        verified.append(current)
    with np.load(path) as f:
        counts = f['raw_counts_by_domain']
    if counts.shape != (4, 12*256**2) or counts.dtype.kind not in 'iu' or np.any(counts < 0):
        raise ValueError('Invalid legacy angular counts')
    if int(counts.sum()) != sum(item['accepted_rows'] for item in old['sources']):
        raise ValueError('Random row accounting mismatch')
    result = responses.normalized_map(counts)
    output = directory/'angular.npz'
    np.savez_compressed(output, **{k: v for k, v in result.items() if k != 'metadata'})
    record = dict(record_base(phase), outputs=[c.file_record(output, content_hash=True)],
                  sources=verified, inherited_counts=c.file_record(old_path, content_hash=True),
                  random_ids=list(range(18)), normalization=result['metadata'], passed=True)
    c.atomic_json(marker, record); return record


def build_cap(phase, cap, xyz, angular_values):
    """Current physical response recipe, with a separate owning receipt."""
    guard(phase); directory = ROOT/phase; marker = directory/(cap+'_RESPONSE.json')
    if marker.exists():
        return receipt(marker, phase)
    selection_path = Path(c.config()['observation']['fixed_selection_manifest'])
    selection = json.loads(selection_path.read_text())
    spec = grids.grid_from_xyz(xyz, 3.383, coord.config()['padding_mpc_h'])
    counts, audit = grids.cic_deposit(xyz, spec)
    if audit['lost_weight'] != 0 or abs(float(counts.sum(dtype=np.float64))-len(xyz)) > 2e-6*len(xyz):
        raise ValueError('Legacy observed CIC conservation failed')
    support = angular_values['support'].astype(bool) & (angular_values['domain']//2 == (cap == 'NGC'))
    boundary = responses.angular_boundary_distance(support)
    shape = tuple(min(64, n) for n in spec.shape); path = directory/(cap+'_response.h5')
    totals = {str(i): dict(observed=0., expected=0., support_voxels=0) for i in range(4)}
    with h5py.File(path, 'x') as f:
        f.attrs.update(schema='e2e-exterior-legacy-observations-v1', phase=phase, cap=cap,
                       origin_mpc_h=spec.origin, shape=spec.shape, cell_mpc_h=3.383,
                       coordinate_sha256=c.sha256(coord.CONFIG), distance_unit='Mpc/h')
        fields = {}
        for i, sl in enumerate(grids.iter_chunks(spec.shape, shape)):
            values, z = obs.response_chunk(spec, sl, support, angular_values['angular_response'], boundary, selection, cap)
            values['counts'] = counts[sl]
            values['log_count_ratio_random'] = grids.log_count_ratio(values['counts'], values['expected_counts_random'],
                                                                    values['exposure_apodized_random'], 1e-3, 1e-4)
            if not fields:
                fields = {k: f.create_dataset(k, shape=spec.shape, dtype=v.dtype, chunks=shape,
                                              compression='lzf', shuffle=True, fillvalue=0) for k, v in values.items()}
            for key, value in values.items():
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite legacy observation')
                if np.any(value): fields[key][sl] = value
            for j, (low, high) in enumerate(zip((.15, .25, .35, .45), (.25, .35, .45, .55))):
                mask = (z >= low) & (z < high)
                totals[str(j)]['observed'] += float(values['counts'][mask].sum(dtype=np.float64))
                totals[str(j)]['expected'] += float(values['expected_counts_random'][mask].sum(dtype=np.float64))
                totals[str(j)]['support_voxels'] += int(values['support_random'][mask].sum())
            if i % 100 == 0:
                print(json.dumps(dict(phase=phase, cap=cap, response_chunk=i)), flush=True)
    ratios = {key: v['observed']/v['expected'] if v['expected'] > 0 else None for key, v in totals.items()}
    if not all(v is not None and .75 <= v <= 1.25 for v in ratios.values()):
        raise ValueError('Legacy selection closure failed: '+str(ratios))
    record = dict(record_base(phase), cap=cap, grid=coord.grid_record(spec), cic=audit,
                  selection=c.file_record(selection_path, content_hash=True), shell_totals=totals,
                  observed_expected_ratio=ratios, outputs=[c.file_record(path, content_hash=True)], passed=True)
    c.atomic_json(marker, record)
    return record


def observations(phase):
    c.require_compute(); guard(phase)
    source = observed(phase); angles = angular(phase)
    xyz, cap = obs.observed_points(source['outputs'][0]['path'])
    with np.load(angles['outputs'][0]['path']) as f:
        values = {k: f[k] for k in f.files}
    for name, index in (('NGC', 1), ('SGC', 0)):
        raw = build_cap(phase, name, xyz[cap == index], values)
        cp = ROOT/phase/(name+'_FACTOR4.json')
        if cp.exists():
            receipt(cp, phase)
        else:
            output = ROOT/phase/(name+'_factor4.h5')
            conservation = conditions.coarsen_response(raw['outputs'][0]['path'], output, raw['grid'])
            c.atomic_json(cp, dict(record_base(phase), cap=name, conservation=conservation,
                raw_receipt_sha256=c.sha256(ROOT/phase/(name+'_RESPONSE.json')),
                outputs=[c.file_record(output, content_hash=True)], passed=True))


def make_geometry(phase):
    c.require_compute(); guard(phase)
    marker = ROOT/phase/'GEOMETRY.json'
    if marker.exists():
        return receipt(marker, phase, False)
    buckets = {}; sources = []
    for cap in ('NGC', 'SGC'):
        path = ROOT/phase/(cap+'_RESPONSE.json'); raw = receipt(path, phase)
        sources.append(c.file_record(path, content_hash=True)); grid = raw['grid']
        rng = np.random.default_rng(geometry.address(60918, phase, cap, 'pair-geometry'))
        centers = np.column_stack([rng.integers(2, (side-margin)//8+1, 65536)*8
                                   for side, margin in zip(grid['shape'], (96, 64, 64))])
        with h5py.File(raw['outputs'][0]['path']) as f:
            support = f['support_random'][:].astype(bool)
        seen = set()
        for center in centers:
            key = tuple(map(int, center))
            if key in seen: continue
            seen.add(key)
            xyz = np.asarray(grid['origin_mpc_h'])+(center+[16, 0, 0])*3.383
            redshift = float(coord.redshift(np.linalg.norm(xyz)))
            shell = int(np.searchsorted((.15, .25, .35, .45, .55), redshift, side='right')-1)
            if not 0 <= shell < 4: continue
            fractions = [float(support[tuple(slice(int(v)-16, int(v)+16) for v in center+[32*side, 0, 0])].mean())
                         for side in (0, 1)]
            if min(fractions) < .25: continue
            kind = 'interior' if min(fractions) >= .95 else 'boundary'
            mid, cores = geometry.domain_positions(center, grid)
            row = dict(phase=phase, role='legacy_nonblind_replication', cap=cap, center=list(key), grid=grid,
                       shell=shell, redshift=redshift, support_stratum=kind, science_core_support_fractions=fractions,
                       source_midpoint_mpc_h=mid.tolist(), source_owned_core_centers_mpc_h=cores.tolist())
            buckets.setdefault((cap, shell, kind), []).append(row)
    selected, mids, cores = [], [], []
    for cap in ('NGC', 'SGC'):
        for shell in range(4):
            for kind in ('interior', 'boundary'):
                accepted = None
                for row in buckets.get((cap, shell, kind), []):
                    mid, pair = np.asarray(row['source_midpoint_mpc_h']), np.asarray(row['source_owned_core_centers_mpc_h'])
                    if geometry.admissible(mid, pair, mids, cores):
                        accepted = dict(row, pair_id=f'{phase}_{cap}_s{shell}_{kind}_00')
                        mids.append(mid); cores.extend(pair); break
                if accepted is None: raise ValueError('No admissible legacy support-only core')
                selected.append(accepted)
    record = dict(record_base(phase), pairs=selected, sources=sources, outputs=[], passed=True,
                  counts_or_targets_read=False, source_nonoverlap=True)
    c.atomic_json(marker, record); return record


def native(phase):
    c.require_compute(); guard(phase)
    marker = ROOT/phase/'NATIVE.json'
    if marker.exists():
        return receipt(marker, phase)
    manifest_path = OLD/phase/'targets/density'/f'AbacusSummit_base_c000_{phase}_z0.200_ngrid2048_ab10_tsc_counts.manifest.json'
    manifest = json.loads(manifest_path.read_text()); build = manifest['build']; contract = manifest['target_contract']
    if (manifest['phase'] != phase or manifest['preflight_only'] or build['processed_file_count'] != 136
            or build['ngrid'] != 2048 or build['boxsize_mpc_h'] != 2000. or build['dtype'] != 'float32'
            or contract['redshift'] != .2 or contract['cosmology'] != 'c000' or contract['mass_assignment'] != 'TSC'
            or contract['particle_subsamples']['total_fraction'] != .1):
        raise ValueError('Native legacy matter contract mismatch')
    path = checked_path(build['output'], phase)
    file = c.file_record(path, content_hash=True)
    previous_hash = False
    if phase in ('ph004', 'ph005'):
        index_path = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v1/data_20260908/native_truth/NATIVE_TRUTH_COMPLETE.json')
        entry = next(r for r in json.loads(index_path.read_text())['phases'] if r['phase'] == phase)
        if file['sha256'] != entry['density_sha256'] or c.sha256(manifest_path) != entry['density_manifest_sha256']:
            raise ValueError('Native grid differs from independent historical truth index')
        previous_hash = True
    # Every original packed particle header/count is cheap to verify. The
    # independent old payload hash is stronger than an unbound new grid hash.
    import asdf
    count = 0
    for name in build['processed_files']:
        particle = checked_path(name, phase)
        with asdf.open(particle, lazy_load=True) as f:
            header = f.tree['header']; array = f.tree['data']['rvint']
            if (header['SimName'] != f'AbacusSummit_base_c000_{phase}' or float(header['Redshift']) != .2
                    or float(header['BoxSizeHMpc']) != 2000. or array.shape[1] != 3):
                raise ValueError('Legacy particle source identity changed')
            count += int(array.shape[0])
    if count != build['particle_count']:
        raise ValueError('Original particle count changed')
    grid = np.load(path, mmap_mode='r', allow_pickle=False)
    if grid.shape != (2048,)*3 or grid.dtype != np.float32:
        raise ValueError('Native grid shape/dtype changed')
    for low in range(0, 2048, 16):
        if not np.isfinite(grid[low:low+16]).all() or np.min(grid[low:low+16]) < 0:
            raise ValueError('Invalid native counts')
    total = float(grid.sum(dtype=np.float64))
    error = abs(total-count)/count
    if error > 2e-6 or abs(total-build['deposited_count']) > 64*np.spacing(total):
        raise ValueError('Legacy native mass conservation changed')
    del grid
    record = dict(record_base(phase), outputs=[file], legacy_manifest=c.file_record(manifest_path, content_hash=True),
                  native_particle_count=count, deposited_count=total, relative_count_error=error,
                  all136_particle_headers_counts_verified=True, independent_historical_grid_hash=previous_hash,
                  particle_payload_crc_rechecked=False, passed=True)
    c.atomic_json(marker, record); return record


def build_targets(phase, workers=32):
    c.require_compute(); guard(phase)
    final = ROOT/phase/'TARGETS.json'
    if final.exists(): return receipt(final, phase)
    geo = make_geometry(phase); source = native(phase); rows = geo['pairs']
    directory = ROOT/phase/'targets'; directory.mkdir(exist_ok=True)
    spectrum, mean = legacy_spectrum(source['outputs'][0]['path'], phase, workers)
    density = fft.irfftn(spectrum.copy(), s=(2048,)*3, workers=workers, overwrite_x=True)
    products = []
    for row in rows:
        parts = {}
        for kind, key in [('joint', 'rho_joint'), ('wide', 'coarse_rho_extended')]:
            starts, average = targets.starts_for(row, kind)
            parts[key] = 1+op.sample_averaged_local(density, starts, average, row['grid'])
        targets.density_qa(parts['rho_joint'], parts['coarse_rho_extended'])
        products.append(parts)
    del density; gc.collect()
    from workflows.sbi.e2e_field_error_budget import inverse_component
    components = []
    for component in op.COMPONENTS:
        field = inverse_component(spectrum, 2000., component, workers=workers)
        values = []
        for row in rows:
            pair = []
            for side in (0, 1):
                center = np.asarray(row['center'])+[32*side, 0, 0]
                starts = [v-16+2*np.arange(16) for v in center]
                pair.append(op.sample_averaged_local(field, starts, 2, row['grid']))
            values.append(np.stack(pair))
        components.append(np.stack(values)); del field; gc.collect()
        print('legacy target component', phase, component, flush=True)
    del spectrum; gc.collect(); tensor = np.stack(components, axis=-1)
    outputs = []
    for i, (row, values) in enumerate(zip(rows, products)):
        targets.fullbox_qa(values['rho_joint'], tensor[i])
        path = directory/(row['pair_id']+'.npz')
        np.savez_compressed(path, **values, fullbox_tensor_cores=tensor[i])
        outputs.append(dict(pair_id=row['pair_id'], **c.file_record(path, content_hash=True)))
    result = dict(record_base(phase), outputs=outputs, geometry_sha256=c.sha256(ROOT/phase/'GEOMETRY.json'),
                  native_sha256=c.sha256(ROOT/phase/'NATIVE.json'), native_mean=mean,
                  smoothing_mpc_h=7., smoothing_count=1, tensor_reference='full periodic native2048 cube', passed=True)
    c.atomic_json(final, result); return result


def legacy_spectrum(path, phase, workers):
    """Same native R7 arithmetic under the separate, explicit legacy authority.

    Do not call or modify the coupled-data reader: its guard correctly rejects
    these phases. The caller first qualifies the native matter receipt.
    """
    c.require_compute(); guard(phase)
    counts = np.load(checked_path(path, phase), mmap_mode='r', allow_pickle=False)
    if counts.shape != (2048,)*3 or counts.dtype != np.float32:
        raise ValueError('Wrong native legacy grid')
    mean = float(counts.mean(dtype=np.float64))
    if not mean > 0:
        raise ValueError('Nonpositive native mass')
    delta = np.empty(counts.shape, dtype=np.float64)
    for lo in range(0, 2048, 16):
        value = counts[lo:lo+16]
        if not np.isfinite(value).all() or value.min() < 0:
            raise ValueError('Invalid native mass')
        delta[lo:lo+16] = value/mean-1
    spectrum = fft.rfftn(delta, workers=workers, overwrite_x=True)
    del delta, counts; gc.collect()
    k = 2*np.pi*fft.fftfreq(2048, d=2000./2048)
    kz = 2*np.pi*fft.rfftfreq(2048, d=2000./2048)
    for i in range(2048):
        spectrum[i] *= np.exp(-.5*7.**2*(k[i]**2+k[:, None]**2+kz[None, :]**2))
    spectrum[0, 0, 0] = 0
    return spectrum, mean


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['observations', 'geometry', 'targets'])
    p.add_argument('--phase', required=True); p.add_argument('--workers', type=int, default=32)
    a = p.parse_args()
    if a.action == 'observations': observations(a.phase)
    elif a.action == 'geometry': make_geometry(a.phase)
    else: build_targets(a.phase, a.workers)
