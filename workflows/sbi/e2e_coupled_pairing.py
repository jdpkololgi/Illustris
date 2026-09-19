"""Verify CutSky -> forFA -> successful DA2 rows without tidal-label access.

Only the source-specific P10 matching algorithms are reused. Legacy registries,
phase guards and frozen products are neither extended nor modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import fitsio
import numpy as np

from workflows.sbi import e2e_coupled_contract as c
from workflows.abacus_tweb import p10_build_bright_parent as parent


REQUIRED = {
    'cutsky': parent.CUTSKY_COLUMNS,
    'forfa': ('TARGETID', 'RA', 'DEC', 'TRUEZ', 'RSDZ', 'R_MAG_APP', 'BGS_TARGET'),
    'full': ('TARGETID', 'RA', 'DEC', 'Z_not4clus', 'ZWARN', 'BGS_TARGET'),
    'random': ('RA', 'DEC', 'PHOTSYS', 'GOODHARDLOC', 'MASKBITS'),
}


def fits_metadata(path, kind):
    result = c.file_record(path)
    with fitsio.FITS(path) as f:
        columns = f[1].get_colnames()
        missing = set(REQUIRED[kind]) - set(columns)
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        result.update(rows=int(f[1].get_nrows()), columns=columns)
        # FITS headers contain scientific provenance, not credentials.
        result['provenance_cards'] = {k: str(v) for k, v in dict(f[1].read_header()).items()
                                      if k.startswith(('SIM', 'COSMO', 'HOD', 'REDSH', 'PHASE'))}
    return result


def inventory(phase):
    sources = c.paths(phase)
    records = {k: fits_metadata(sources[k], k) for k in ('cutsky', 'forfa', 'full')}
    records['randoms'] = [fits_metadata(path, 'random') for path in sources['randoms']]
    records['intermediate_assets'] = {k: c.file_record(sources[k])
                                      for k in ('pota', 'fiberassign')}
    return dict(schema='e2e-coupled-source-inventory-v1', phase=phase, role=c.ROLES[phase],
                **c.provenance(), sources=records, pass_metadata=True,
                pairing_verified=False, matter_verified=False)


def build_parent(phase, directory, chunk=500_000):
    sources = c.paths(phase)
    receipt = directory / 'PARENT_COMPLETE.json'
    if receipt.exists():
        return c.verify_receipt(receipt)
    output = directory / 'bright_parent_linkage.fits'
    if output.exists():
        raise ValueError('unreceipted parent output; inspect before adopting or replacing')
    started = time.monotonic()
    contract = parent.scan_forfa_bright_contract(sources['forfa'], chunk)
    keys, ids = parent.build_forfa_key_index(sources['forfa'], contract['bright_rows'], chunk)
    # A unique attempt leaves any interrupted attempt intact for inspection.
    partial = directory / f'parent_attempt_{time.time_ns()}.fits'
    first, next_id = True, 1
    with fitsio.FITS(partial, 'rw') as writer, fitsio.FITS(sources['cutsky']) as src:
        rows = int(src[1].get_nrows())
        for start in range(0, rows, chunk):
            block = src[1].read(rows=np.arange(start, min(start+chunk, rows)),
                                columns=list(parent.CUTSKY_COLUMNS))
            selected, targetids = parent.match_bright_chunk(
                block, r_limit=19.5, sorted_keys=keys, sorted_targetids=ids)
            if len(selected):
                if not np.array_equal(targetids, np.arange(next_id, next_id+len(selected))):
                    raise ValueError('CutSky/forFA exact match is not sequential and complete')
                compact = parent.compact_parent_block(selected, targetids)
                if np.any(compact['FILE_NUM'] < 0) or np.any(compact['FILE_NUM'] >= 34):
                    raise ValueError('host slab outside the 34-slab matter box')
                if np.any(compact['HALO_INDEX'] < 0):
                    raise ValueError('negative host halo index')
                if first:
                    writer.write(compact, extname='PARENT')
                    first = False
                else:
                    writer[-1].append(compact)
                next_id += len(compact)
            if start % (5*chunk) == 0:
                print(json.dumps(dict(phase=phase, step='parent', rows=start,
                                      matched=next_id-1)), flush=True)
    if first or next_id-1 != contract['bright_rows']:
        raise ValueError('incomplete CutSky/forFA bright match')
    parity = parent.validate_parent_against_forfa(partial, sources['forfa'], next_id-1, chunk)
    os.link(partial, output)
    partial.unlink()  # Only the just-published attempt, not any interrupted data.
    result = dict(schema='e2e-coupled-parent-v1', phase=phase, role=c.ROLES[phase],
                  **c.provenance(), source_code_sha256=c.sha256(__file__),
                  matching_code_sha256=c.sha256(parent.__file__),
                  sources={k: c.file_record(sources[k], content_hash=True)
                           for k in ('cutsky', 'forfa')},
                  contract=contract, parity=parity, outputs=[c.file_record(output, content_hash=True)],
                  elapsed_seconds=time.monotonic()-started, target_columns_present=False,
                  **{'pass': True})
    c.atomic_json(receipt, result)
    return result


def successful(block):
    z = block['Z_not4clus']
    return np.isfinite(z) & (z > 0) & (block['ZWARN'] == 0)


def validate_successful_join(block, link):
    ids = np.asarray(block['TARGETID'], dtype=np.int64)
    if not np.array_equal(ids, link['TARGETID']):
        raise ValueError('successful TARGETID mismatch')
    for name in ('RA', 'DEC'):
        if not np.isfinite(block[name]).all() or not np.allclose(
                block[name], link[name], rtol=0, atol=1e-10):
            raise ValueError(f'successful sky mismatch: {name}')
    if np.any((block['BGS_TARGET'] & 2) == 0):
        raise ValueError('non-BRIGHT successful observation')
    # Check, rather than assume, the observed-redshift convention of this mock.
    if not np.allclose(block['Z_not4clus'], link['Z'], rtol=0, atol=2e-7):
        raise ValueError('successful LSS redshift differs from CutSky RSD redshift')


def build_observed(phase, directory, parent_receipt, chunk=500_000):
    receipt = directory / 'OBSERVED_COMPLETE.json'
    if receipt.exists():
        return c.verify_receipt(receipt)
    sources = c.paths(phase)
    output = directory / 'observed_geometry.fits'
    if output.exists():
        raise ValueError('unreceipted observed output')
    started = time.monotonic()
    linkage = fitsio.read(parent_receipt['outputs'][0]['path'])
    seen = np.zeros(len(linkage), dtype=bool)
    partial = directory / f'observed_attempt_{time.time_ns()}.fits'
    first, n_selected, context = True, 0, 0
    with fitsio.FITS(sources['full']) as src, fitsio.FITS(partial, 'rw') as dst:
        rows = int(src[1].get_nrows())
        for start in range(0, rows, chunk):
            block = src[1].read(rows=np.arange(start, min(start+chunk, rows)),
                                columns=list(REQUIRED['full']))
            block = block[successful(block)]
            ids = np.asarray(block['TARGETID'], dtype=np.int64)
            if not len(ids):
                continue
            if ids.min() < 1 or ids.max() > len(linkage):
                raise ValueError('successful TARGETID out of parent bounds')
            if len(np.unique(ids)) != len(ids) or seen[ids-1].any():
                raise ValueError('duplicated successful TARGETID')
            link = linkage[ids-1]
            validate_successful_join(block, link)
            seen[ids-1] = True
            compact = np.empty(len(block), dtype=[('TARGETID','i8'), ('RA','f8'),
                                                 ('DEC','f8'), ('Z','f8')])
            for key in ('TARGETID','RA','DEC'):
                compact[key] = block[key]
            compact['Z'] = block['Z_not4clus']
            if first:
                dst.write(compact, extname='OBSERVED')
                first = False
            else:
                dst[-1].append(compact)
            z = compact['Z']
            context += int(((z >= .10) & (z < .60) & ~((z >= .585) & (z < .595))).sum())
            n_selected += len(block)
    if first:
        raise ValueError('no observed rows')
    os.link(partial, output)
    partial.unlink()
    result = dict(schema='e2e-coupled-observed-v1', phase=phase, role=c.ROLES[phase],
                  **c.provenance(), source_code_sha256=c.sha256(__file__),
                  parent_receipt_sha256=c.sha256(directory/'PARENT_COMPLETE.json'),
                  source=c.file_record(sources['full'], content_hash=True),
                  outputs=[c.file_record(output, content_hash=True)],
                  observed_rows=n_selected, context_rows=context, input_rows=rows,
                  target_columns_present=False, exact_id_sky_rsd_join=True,
                  elapsed_seconds=time.monotonic()-started, **{'pass': True})
    c.atomic_json(receipt, result)
    return result


def prepare_phase(phase, metadata_only=False):
    c.phase_guard(phase)
    c.bind_run()
    directory = c.guarded(c.ROOT/'observations'/phase, phase, output=True)
    with c.single_writer(directory):
        record = inventory(phase)
        inventory_path = directory/'SOURCE_INVENTORY.json'
        if inventory_path.exists():
            old = json.loads(inventory_path.read_text())
            if old['sources'] != record['sources'] or old['config_sha256'] != record['config_sha256']:
                raise ValueError('source inventory drift')
        else:
            c.atomic_json(inventory_path, record)
        if metadata_only:
            return record
        c.require_compute()
        result = build_parent(phase, directory)
        return build_observed(phase, directory, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phases', nargs='+', default=list(c.ROLES))
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--stop-after-seconds', type=float, default=1e10)
    args = parser.parse_args()
    started = time.monotonic()
    for phase in args.phases:
        if time.monotonic()-started > args.stop_after_seconds:
            return 75
        record = prepare_phase(phase, args.metadata_only)
        print(json.dumps(dict(phase=phase, complete=True,
                              metadata_only=args.metadata_only,
                              rows=record.get('observed_rows'))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
