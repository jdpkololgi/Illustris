"""Bounded reviewed chart inputs; no publication or external data transfer."""
import argparse
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_coupled_contract as c


def emit(path,payload):
    if path.exists():
        if json.loads(path.read_text())!=payload:
            raise ValueError(f'Refusing to replace changed delivery data: {path}')
        return
    c.atomic_json(path,payload)


def build(scores,out):
    c.require_compute()
    summary=json.loads((scores/'SCORES.json').read_text())
    verified=json.loads((scores/'VERIFIED.json').read_text())
    if not verified['passed'] or verified['score_sha256']!=c.sha256(scores/'SCORES.json'):
        raise ValueError('Unverified or changed scores cannot be delivered')
    out.mkdir(parents=True,exist_ok=True)
    names={'isolated':'Zero exterior','periodic':'Periodic cap',
           'conditional_d3':'Conditional 12 modes','oracle_d3':'Oracle 12 modes'}
    labels=['0–13.5','13.5–27.1','27.1–54.1','54.1–108.3','>108.3']
    rows=[]
    for index,label in enumerate(labels):
        for name,title in names.items():
            values=[];counts=[]
            for phase in ('ph016','ph017'):
                entries=[v['arms'][name]['distance_bins'][index] for v in summary['cores'] if v['phase']==phase]
                entries=[v for v in entries if v is not None]
                values.append(float(np.mean([v['class_flip'] for v in entries])) if entries else None)
                counts.append(len(entries))
            mean=100*sum(values)/2 if all(v is not None for v in values) else None
            rows.append(dict(distance_bin=label,treatment=title,class_mismatch_pct=mean,
                ph016_cores=counts[0],ph017_cores=counts[1]))
    factor=summary.get('source_factor',2)
    cell=3.383*factor
    source=dict(label='Matched mock boundary experiment',files=[{'label':'SCORES.json'},{'label':'VERIFIED.json'}],
        metricDefinitions=[dict(label='Class mismatch',definition=
            'Class mismatch is the fraction of fully supported cells whose mean-field web class differs from full-box truth at lambda 0.2. Cores are equally weighted within each phase and phases are equally weighted.')],
        caveats=[f'Sources use {cell:g} Mpc/h cells and R7 smoothing; evaluation cells remain 6.766 Mpc/h.',
            'Only two exposed evaluation phases are used; cells and cores within a phase are correlated.',
            'Inputs are true observed density, not galaxy-inferred density.',
            'The oracle uses evaluation truth and is not a deployable prediction.',
            'Periodic cap closure repeats one cap box and holds the other cap contribution isolated.'],
        evidenceFlow=[dict(kind='validation',title='Saved-array checks',detail=
            'Original support masks and 64 independently recomputed core tensor errors agree with the saved products. Selected full-footprint FFT values agree with direct cuboid sums.')])
    chart=dict(schemaVersion=1,id='boundary-class-errors',queryId='boundary-distance-errors',
        title='Class errors versus distance from the mask boundary',
        chart=dict(type='line',x='distance_bin',y='class_mismatch_pct',series='treatment',
            xLabel='Distance to non-full support (Mpc/h)',yLabel='Mean-field class mismatch (%)',
            showXAxisLabel=True,startAtZero=True),
        rows=rows,columns=list(rows[0]),source=source,theme='codex-classic')
    emit(out/'boundary-class-errors.json',chart)
    receipt=dict(schemaVersion=1,items=[dict(id='boundary-distance-errors',
        title='How do boundary treatments affect classification?',queries=[dict(id='boundary-distance-errors',
            source=source,rows=rows,columns=list(rows[0]),preview=dict(kind='aggregate',
                note='Phase-balanced averages across registered cores; counts show which cores contribute to each distance bin.'))])])
    emit(out/'boundary-class-sources.json',receipt)
    comparison=[dict(group=group,treatment=name,**v)
        for group in ('all','stratum:boundary','stratum:interior')
        for name,v in summary['groups'][group].items()]
    # Keep a compact provenance receipt for non-charted tensor/calibration claims.
    native_rows=[dict(group=v['group'],treatment=v['treatment'],tensor_rmse=v['tensor_rmse'],
        class_mismatch_pct=100*v['class_flip'],
        coverage_lambda1_pct=100*v['central90_coverage'][0] if 'central90_coverage' in v else None,
        coverage_lambda2_pct=100*v['central90_coverage'][1] if 'central90_coverage' in v else None,
        coverage_lambda3_pct=100*v['central90_coverage'][2] if 'central90_coverage' in v else None) for v in comparison]
    extra_source=dict(source)
    extra_source['metricDefinitions']=[dict(label='Tensor RMSE',definition=
        'Tensor RMSE is the square root of the equally weighted core mean of squared matrix Frobenius errors.'),
        dict(label='Eigenvalue coverage',definition=
        'Coverage uses 64 conditional draws and order-statistic indices 3 and 60, with an exchangeable reference of 57/65 or 87.6923 percent.')]
    extra_source['caveats']=source['caveats']+[
        'Independent core-local harmonic-response draws are not coherent full-survey exterior density realizations.',
        'A two-resolution comparison does not prove the continuum limit near mask edges.']
    extra=dict(schemaVersion=1,items=[dict(id='boundary-model-results',title='Boundary model accuracy and coverage',
        queries=[dict(id='boundary-model-results',source=extra_source,rows=native_rows,
            columns=list(native_rows[0]),preview=dict(kind='aggregate',
                note='All 64 evaluation cores and separate 32-core boundary/interior strata, equally weighted.'))])])
    emit(out/'boundary-model-sources.json',extra)
    emit(scores/'DISTANCE_SUMMARY.json',dict(rows=rows,source_sha256=c.sha256(scores/'SCORES.json')))
    coverage_rows=[]
    for group,label in (('stratum:boundary','Boundary cores'),('stratum:interior','Interior cores')):
        for index,value in enumerate(summary['groups'][group]['conditional_d3']['central90_coverage']):
            coverage_rows.append(dict(region=label,eigenvalue=f'lambda {index+1}',coverage_pct=100*value))
    coverage_source=dict(extra_source)
    coverage_source['metricDefinitions']=[extra_source['metricDefinitions'][1]]
    coverage_chart=dict(schemaVersion=1,id='boundary-coverage',queryId='boundary-coverage',
        title='Conditional coverage by location — reference 87.7%',
        chart=dict(type='bar',x='region',y='coverage_pct',series='eigenvalue',
            xLabel='Registered core stratum',yLabel='Interval coverage (%)',startAtZero=True),
        rows=coverage_rows,columns=list(coverage_rows[0]),source=coverage_source,theme='codex-classic')
    emit(out/'boundary-coverage.json',coverage_chart)
    emit(out/'boundary-coverage-sources.json',dict(schemaVersion=1,items=[
        dict(id='boundary-coverage',title='Where are conditional intervals calibrated?',queries=[
            dict(id='boundary-coverage',source=coverage_source,rows=coverage_rows,columns=list(coverage_rows[0]))])]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.scores,a.out)
