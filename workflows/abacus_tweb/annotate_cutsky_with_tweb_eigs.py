#!/usr/bin/env python3
"""Annotate qualified BGS v0.1 galaxies at their own real-space positions.

A successful coordinate-transform qualification is mandatory. Every population
uses its own RA/DEC/Z_COSMO; INDEP_POS records exact independent checks separately. Legacy host-row
lookup is intentionally unavailable here: HALO_INDEX is not a raw halo row for
the low-z population. Derived inputs must carry the original SOURCE_ROW and the
required truth metadata; each row is checked against the qualified raw CutSky.
Existing products are never overwritten. Output and manifest are versioned.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import fitsio
import numpy as np

WORKFLOW=Path(__file__).resolve().parent
if str(WORKFLOW) not in sys.path: sys.path.insert(0,str(WORKFLOW))
from galaxy_position_mapping import MAPPING_VERSION, reconstruct_native, field_indices, mapping_definition
from tweb_grid_sampling import TWebGrid, LABEL_COLUMNS
from galaxy_position_validation_status import (LABEL_POLICY_VERSION, INDEPENDENT_POSITION_CODES,
    load_independent_index, independent_status, load_ambiguity_rows, ambiguity_status)

TRUTH_COLUMNS=('RA','DEC','Z_COSMO','RES','CEN','HALO_MASS')
STATUS_VALID=1


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cutsky',type=Path,required=True)
    p.add_argument('--tweb-dir',type=Path,required=True)
    p.add_argument('--phase',required=True)
    p.add_argument('--qualification',type=Path,required=True,help='Successful independent validation report.json')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--output-name',default=None)
    p.add_argument('--chunk-size',type=int,default=100_000)
    return p.parse_args()


def load_qualification(path,phase,tweb_dir):
    if phase == 'ph001':
        raise ValueError('ph001 is sealed; this migration does not authorize unsealing')
    if phase in {f'ph{i:03d}' for i in range(14,20)}:
        raise ValueError('Reserved confirmation targets remain closed in the galaxy-position migration')
    receipt=json.loads(Path(path).read_text())
    if receipt.get('completed') is not True or receipt.get('pass_supported_population_gate') is not True:
        raise ValueError('Coordinate qualification is incomplete or failed')
    if receipt.get('label_policy_version') != LABEL_POLICY_VERSION or receipt.get('allow_all_coordinate_populations') is not True:
        raise ValueError('Qualification lacks the all-population v2 label policy')
    if receipt.get('mapping',{}).get('version')!=MAPPING_VERSION:
        raise ValueError('Coordinate qualification version mismatch')
    if receipt.get('helper_sha256')!=sha256_file(WORKFLOW/'galaxy_position_mapping.py'):
        raise ValueError('Coordinate implementation changed since qualification')
    tested=receipt.get('phases',{}).get(phase)
    if not tested or tested.get('pass_supported_population_gate') is not True:
        raise ValueError(f'Phase not independently qualified: {phase}')
    if Path(tested['field_path']).resolve()!=Path(tweb_dir).resolve():
        raise ValueError('T-Web directory differs from qualified phase field')
    if not tested.get('field_sources'):
        raise ValueError('Qualification lacks a bound T-Web file inventory')
    for record in tested['field_sources']:
        st = Path(record['path']).stat()
        if (st.st_size, st.st_mtime_ns) != (record['bytes'], record['mtime_ns']):
            raise ValueError('T-Web field changed since qualification')
    raw=tested['sources'][0]
    source=Path(raw['path']); st=source.stat()
    if st.st_size!=raw['bytes'] or st.st_mtime_ns!=raw['mtime_ns']:
        raise ValueError('Raw CutSky changed since qualification')
    return receipt,source


def verify_source_rows(chunk,source_hdu,rows):
    """Use galaxy row identity, never a host tuple; preserve duplicates/order."""
    rows=np.asarray(rows)
    if not np.issubdtype(rows.dtype,np.integer) or np.any(rows<0) or np.any(rows>=source_hdu.get_nrows()):
        raise ValueError('SOURCE_ROW is not a valid zero-based original CutSky row')
    unique,inverse=np.unique(rows.astype(np.int64),return_inverse=True)
    reference=source_hdu.read(rows=unique,columns=list(TRUTH_COLUMNS))[inverse]
    for key in TRUTH_COLUMNS:
        if not np.array_equal(chunk[key],reference[key]):
            raise ValueError(f'SOURCE_ROW identity mismatch in {key}; rebuild metadata from the original CutSky')


def annotate_chunk(chunk,source_rows,grid,independent_index=None,ambiguous_rows=None):
    missing=set(TRUTH_COLUMNS)-set(chunk.dtype.names or ())
    if missing: raise ValueError(f'Missing galaxy-coordinate/identity metadata: {sorted(missing)}')
    if np.any(~np.isin(chunk['RES'],[0,1])) or np.any(~np.isin(chunk['CEN'],[0,1])):
        raise ValueError('Unknown RES/CEN population encoding')
    native=reconstruct_native(chunk['RA'],chunk['DEC'],chunk['Z_COSMO'])
    indices=field_indices(native,ngrid=grid.ngrid,boxsize=grid.boxsize)
    labels=grid.sample_indices(indices)
    extra=[('SOURCE_ROW','i8'),('TWEB_STATUS','u1'),('HAS_LABEL','u1'),('INDEP_POS','u1'),('ID_AMBIG','u1'),('CWEB','u1'),
           ('TWEB_X','f4'),('TWEB_Y','f4'),('TWEB_Z','f4'),('TWEB_IX','u2'),('TWEB_IY','u2'),('TWEB_IZ','u2')]
    extra += [(name,'f4') for name in LABEL_COLUMNS]
    replaced={x[0] for x in extra}
    dtype=[d for d in chunk.dtype.descr if d[0] not in replaced]+extra
    out=np.empty(len(chunk),dtype=dtype)
    for name in chunk.dtype.names:
        if name not in replaced: out[name]=chunk[name]
    out['SOURCE_ROW']=source_rows
    out['TWEB_STATUS']=STATUS_VALID;out['HAS_LABEL']=1
    out['INDEP_POS']=independent_status(source_rows,independent_index)
    out['ID_AMBIG']=ambiguity_status(source_rows,ambiguous_rows)
    if np.any((out['ID_AMBIG']!=0)&(out['INDEP_POS']!=0)):
        raise ValueError('Ambiguous identity cannot be independently identified')
    for j,letter in enumerate('XYZ'):
        out[f'TWEB_{letter}']=native[:,j];out[f'TWEB_I{letter}']=indices[:,j]
    for key,values in labels.items(): out[key]=values
    return out


def main():
    args=parse_args()
    if not(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):
        raise RuntimeError('Annotation requires an approved Slurm compute step')
    if not 1<=args.chunk_size<=1_000_000: raise ValueError('chunk-size must be 1..1000000')
    receipt,source=load_qualification(args.qualification,args.phase,args.tweb_dir)
    independent_index=load_independent_index(receipt['phases'][args.phase]['independent_position_index'])
    ambiguous_rows=load_ambiguity_rows(receipt['phases'][args.phase]['independent_position_index'])
    ambiguity_count=0
    grid=TWebGrid(args.tweb_dir)
    if (grid.ngrid,grid.boxsize,grid.rsmooth)!=(2048,2000.,7.):
        raise ValueError('Grid resolution/box/smoothing outside this qualification')
    output=args.output_dir/(args.output_name or f'{args.cutsky.stem}_galaxy_realspace_v2_tweb.fits')
    manifest=Path(str(output)+'.manifest.json')
    temporary=output.with_name('.'+output.name+f'.{os.getpid()}.partial')
    if any(p.exists() for p in [output,manifest,temporary]): raise FileExistsError(f'Refusing to overwrite annotation: {output}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    source_is_input=args.cutsky.resolve()==source.resolve()
    started=time.monotonic();count=0;digest=hashlib.sha256();groups={};evidence_counts={str(k):0 for k in INDEPENDENT_POSITION_CODES}
    with fitsio.FITS(args.cutsky) as fin, fitsio.FITS(source) as fs:
        names=set(fin[1].get_colnames())
        missing=set(TRUTH_COLUMNS)-names
        if missing: raise ValueError(f'Input lacks {sorted(missing)}; rebuild metadata, do not substitute observed Z')
        if not source_is_input and 'SOURCE_ROW' not in names:
            raise ValueError('Derived input lacks original SOURCE_ROW; host triples are not galaxy IDs')
        nrows=fin[1].get_nrows()
        if nrows==0: raise ValueError('Empty annotation input')
        with fitsio.FITS(temporary,'rw',clobber=False) as fout:
            for start in range(0,nrows,args.chunk_size):
                stop=min(start+args.chunk_size,nrows); chunk=fin[1][start:stop]
                ids=np.arange(start,stop,dtype=np.int64) if source_is_input else chunk['SOURCE_ROW']
                if not source_is_input: verify_source_rows(chunk,fs[1],ids)
                out=annotate_chunk(chunk,ids,grid,independent_index,ambiguous_rows)
                ambiguity_count+=int(np.sum(out['ID_AMBIG']))
                for code in INDEPENDENT_POSITION_CODES:evidence_counts[str(code)]+=int(np.sum(out['INDEP_POS']==code))
                digest.update(np.asarray(ids,dtype='<i8').tobytes())
                for key in TRUTH_COLUMNS: digest.update(np.ascontiguousarray(chunk[key]).tobytes())
                for res in [0,1]:
                    for cen in [0,1]:
                        key=f'RES{res}_CEN{cen}';groups[key]=groups.get(key,0)+int(np.sum((chunk['RES']==res)&(chunk['CEN']==cen)))
                if start==0:
                    fout.write(out,extname='GALAXY_TWEB',header={'TWMAP':MAPPING_VERSION,'SIMPHASE':args.phase,
                        'TWTARGET':'galaxy_real_space','TWQUAL':sha256_file(args.qualification),'TWPOLICY':LABEL_POLICY_VERSION,
                        'TWRSCALE':grid.rsmooth,'TWTHRESH':grid.threshold})
                else: fout[1].append(out)
                count+=len(out);print(f'Annotated {count:,}/{nrows:,} galaxy positions',flush=True)
    payload=dict(schema_version='galaxy-position-tweb-annotation-v2',phase=args.phase,mapping=mapping_definition(),label_policy_version=LABEL_POLICY_VERSION,
        input=dict(path=str(args.cutsky.resolve()),bytes=args.cutsky.stat().st_size,mtime_ns=args.cutsky.stat().st_mtime_ns),
        source_cutsky=receipt['phases'][args.phase]['sources'][0],source_identity='phase + zero-based SOURCE_ROW',
        source_rows_verified=True,mapping_input_digest_sha256=digest.hexdigest(),mapping_digest_chunk_size=args.chunk_size,
        qualification=dict(path=str(args.qualification.resolve()),sha256=sha256_file(args.qualification)),
        field=grid.definition(),rows=count,populations=groups,status_codes={'1':'finite_ordered_galaxy_realspace_labels'},
        labelled_rows=count,unsupported_rows=0,identity_ambiguous_rows=ambiguity_count,
        identity_ambiguity_scope='Only exact audited source rows; zero does not assert unique identity',input_rows_fully_labelled=True,
        independent_position_codes=INDEPENDENT_POSITION_CODES,independent_position_counts=evidence_counts,
        all_input_positions_individually_cross_checked=evidence_counts['0']==0,
        unresolved_satellite_rows=groups.get('RES0_CEN0',0),
        full_mock_population_independently_qualified=receipt.get('full_population_pass',False),
        source_code_sha256={name:sha256_file(WORKFLOW/name) for name in [Path(__file__).name,'galaxy_position_mapping.py','tweb_grid_sampling.py','galaxy_position_validation_status.py']},
        output=dict(path=str(output.resolve()),bytes=temporary.stat().st_size,sha256=sha256_file(temporary)),
        job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,completed=True)
    # A completion manifest is written only after all rows pass; interrupted partials
    # remain visibly incomplete and cannot be mistaken for a qualified catalogue.
    os.replace(temporary,output)
    manifest_tmp=manifest.with_name('.'+manifest.name+f'.{os.getpid()}.partial')
    manifest_tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    os.replace(manifest_tmp,manifest)
    print(f'Completed {output}',flush=True)

if __name__=='__main__': main()
