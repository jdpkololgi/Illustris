#!/usr/bin/env python3
"""Certify coordinate labels while separating identity and target equivalence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from galaxy_position_validation_status import LABEL_POLICY_VERSION, REFERENCE_KIND_CODES, INDEPENDENT_POSITION_CODES
from galaxy_position_mapping import periodic_delta
from galaxy_label_equivalence import RULE_VERSION, validate_equivalence


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_phase_evidence(info, data, tolerance):
    rows=data['original_row'];kind=data['reference_kind']
    if len(np.unique(rows))!=len(rows):raise ValueError('Duplicate sampled source identity')
    codes=np.zeros(len(rows),dtype='u1')
    for name,code in REFERENCE_KIND_CODES.items():codes[kind==name]=code
    records=info.get('label_equivalence_records',[])
    ambiguous=np.asarray(data.get('label_equivalent_ambiguity',np.zeros(len(rows),dtype=bool)),dtype=bool)
    if ambiguous.shape!=rows.shape or np.any(ambiguous&(codes!=0)):
        raise ValueError('Ambiguous identity cannot carry an independent identity code')
    if records and info.get('ambiguity_rule_version')!=RULE_VERSION:
        raise ValueError('Unknown label equivalence rule')
    record_rows=[int(v['original_row']) for v in records]
    if len(set(record_rows))!=len(record_rows) or set(record_rows)!=set(map(int,rows[ambiguous])):
        raise ValueError('Ambiguous row evidence does not cover the sample flags exactly')
    for record in records:
        validate_equivalence(record,tolerance)
        i=int(np.flatnonzero(rows==record['original_row'])[0])
        if not np.array_equal(data['reconstructed_native'][i],record['query_position']):
            raise ValueError('Candidate evidence belongs to a different coordinate')
        expected_res={'resolved_central':1,'unresolved_central':0}[record['population']]
        if not(data['CEN'][i]==1 and data['RES'][i]==expected_res and data['FILE_NUM'][i]==record['slab'] and data['BOX_INDEX'][i]<0):
            raise ValueError('Candidate evidence population/slab mismatch')
    for check in info['native_checks']:
        count=sum(v['population']==check['population'] and v['slab']==check['slab'] for v in records)
        unique=int(check.get('accepted',check['rows']))
        if count!=check.get('label_equivalent_ambiguous',0) or unique+count!=check['rows']:
            raise ValueError('Native unique and label-equivalent coverage incomplete')
        if check['rows']<=0 or not check.get('pass_label_gate',check['pass_gate']):
            raise ValueError('Native label qualification failed')
    if len(info['native_checks'])<4 or not info['field_gate']:
        raise ValueError('Native field qualification incomplete')
    if not all(v['rows']>0 and v['identity_matches']==v['rows'] and v['within_tolerance']==v['rows'] for v in info['cubic'].values()):
        raise ValueError('Cubic qualification failed')
    selected=codes>0
    residual=np.linalg.norm(periodic_delta(data['reconstructed_native'][selected],data['independent_native'][selected]),axis=1)
    if not np.isfinite(residual).all() or np.any(residual>=tolerance):
        raise ValueError('Independent row residual failed')
    expected={1:sum(v['rows'] for v in info['cubic'].values()),
              2:sum(v.get('accepted',v['rows']) for v in info['native_checks'] if v['population']=='resolved_central'),
              3:sum(v.get('accepted',v['rows']) for v in info['native_checks'] if v['population']=='unresolved_central')}
    if any(np.sum(codes==k)!=v for k,v in expected.items()):
        raise ValueError('Independent row coverage differs from audit')
    order=np.argsort(rows[selected])
    return rows[selected][order],codes[selected][order],np.sort(rows[ambiguous]),expected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--satellite-hosts',type=Path)
    p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit=json.loads(args.audit.read_text())
    if audit.get('completed') is not True:raise ValueError('Incomplete coordinate audit')
    if any(ph in {'ph001',*[f'ph{i:03d}' for i in range(14,20)]} for ph in audit['phases']):
        raise ValueError('Reserved phases remain closed')
    certificate=copy.deepcopy(audit)
    certificate.update(schema_version='galaxy-position-coordinate-certificate-v2',
        label_policy_version=LABEL_POLICY_VERSION,allow_all_coordinate_populations=True,
        ambiguity_rule_version=RULE_VERSION,independent_position_codes=INDEPENDENT_POSITION_CODES,
        audit_source={'path':str(args.audit.resolve()),'sha256':digest(args.audit)},
        labelled_populations=['all RES/CEN populations with valid RA/DEC/Z_COSMO and source-row identity'],
        evidence_boundary='INDEP_POS marks unique independent sampled references only; ID_AMBIG marks exact sampled rows with ambiguous identity but exhaustive cell/label equivalence. Zero flags do not assert identity.',
        label_policy='Sample own galaxy coordinates; independent identity never masks finite labels.')
    if args.output.exists():raise FileExistsError(args.output)
    validated={}
    for ph,info in certificate['phases'].items():
        # Refuse certificates based on changed input or target fields.
        for record in [*info['sources'],*info['field_sources'],*[v['source'] for v in info['native_checks']]]:
            stat=Path(record['path']).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(record['bytes'],record['mtime_ns']):
                raise ValueError('Qualification source/field changed')
        sample=args.samples/f'{ph}_sample.npz'
        with np.load(sample,allow_pickle=False) as f:data={k:f[k] for k in f.files}
        validated[ph]=validate_phase_evidence(info,data,audit['tolerance_mpc_h'])
        info['pass_supported_population_gate']=True
    if args.satellite_hosts:
        satellite=json.loads(args.satellite_hosts.read_text())
        if satellite.get('completed') is not True:raise ValueError('Supplemental satellite audit incomplete')
        certificate['satellite_host_closure']={'path':str(args.satellite_hosts.resolve()),'sha256':digest(args.satellite_hosts),'pass':satellite.get('host_closure_pass'),'independent_satellite_positions':False}
    evidence_dir=args.output.parent/(args.output.stem+'_rows');evidence_dir.mkdir(parents=True,exist_ok=False)
    for ph,(rows,codes,ambiguous,counts) in validated.items():
        target=evidence_dir/f'{ph}.npz'
        with target.open('xb') as f:np.savez_compressed(f,source_row=rows,independent_position_code=codes,ambiguous_source_row=ambiguous)
        certificate['phases'][ph]['independent_position_index']={
            'path':str(target.resolve()),'sha256':digest(target),'counts':counts,
            'ambiguous_identity_count':len(ambiguous),'sample_sha256':digest(args.samples/f'{ph}_sample.npz')}
    certificate['pass_supported_population_gate']=True
    certificate['full_population_pass']=bool(certificate.get('pass_gate',False))
    with args.output.open('x') as f:json.dump(certificate,f,indent=2,allow_nan=False);f.write('\n')
    print('Label qualification passed; historical full unique-position gate:',certificate['full_population_pass'])

if __name__=='__main__':main()
