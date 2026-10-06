"""Receipt/invariant audit and measured resource estimate; no coverage claim."""
import json
from pathlib import Path
from workflows.sbi import e2e_coupled_contract as c

BASE = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1')
ROOT = BASE.parent/'cfm_mixture_pilot_20261006_v1'


def run():
    c.require_compute()
    summary = json.loads((ROOT/'SUMMARY.json').read_text())
    if not summary['passed'] or summary['scientific_coverage_result'] != 'not tested':
        raise ValueError('Wrong scientific scope')
    receipts=[]; phase_rows=[]; checks=0
    for phase in ('ph012','ph013'):
        records={}
        for mode in ('reference17','seed17','seed29'):
            folder=ROOT/phase/mode
            record=json.loads((folder/'PILOT.json').read_text())
            binding=json.loads((folder/'BINDING.json').read_text())
            benchmark=json.loads((folder/'BENCHMARK.json').read_text())
            assert record['phase']==phase and record['cap']=='SGC' and record['mode']==mode
            assert record['effective_nfe']==benchmark['nfe']==2 and record['draws']==4
            assert benchmark['posterior_output'] is False and not (folder/'FIELD.h5').exists()
            assert record['coverage_improvement_tested'] is False and record['full_nfe_parity_tested'] is False
            assert 0<=record['decoder_max_mass_error']<2e-6 and record['decoder_positive']
            assert record['passed'] and benchmark['passed']
            for name in ('source','sampler','contract','binding','input_baseline'):
                rec=record[name]
                assert c.sha256(rec['path'])==rec['sha256'];checks+=1
            assert record['benchmark']==benchmark
            assert benchmark['binding']==c.sha256(folder/'BINDING.json')
            assert all(record['traces'][s]['finite'] for s in ('coarse','fine'))
            records[mode]=(record,binding)
            receipts.append(c.file_record(folder/'PILOT.json',content_hash=True))
            checks+=9
        ref,b17,b29=(records[k] for k in ('reference17','seed17','seed29'))
        assert ref[0]['traces']==b17[0]['traces']
        for key in ('normalizer','prepared_sha256','checkpoints'):
            assert ref[1][key]==b17[1][key];checks+=1
        for key in ('normalizer','prepared_sha256'):
            assert b17[1][key]==b29[1][key];checks+=1
        assert b17[1]['model_seed']==17 and b29[1]['model_seed']==29
        for stage in ('coarse','fine'):
            assert b17[1]['checkpoints'][stage]!=b29[1]['checkpoints'][stage]
            assert b17[0]['traces'][stage]['sha256']!=b29[0]['traces'][stage]['sha256']
            assert b17[0]['traces'][stage]['shape']==b29[0]['traces'][stage]['shape']
            assert b17[0]['traces'][stage]['coverage_sha256']==b29[0]['traces'][stage]['coverage_sha256']
            checks+=4
        choices=summary['future_component_assignments'][phase]
        assert len(choices)==32 and set(choices)<={17,29}
        phase_rows.append(dict(phase=phase,cap='SGC',exact_short_replay=True,
            seed29_distinct=True,component17_draws=choices.count(17),component29_draws=choices.count(29),
            windows=b17[0]['benchmark']['windows'],
            sampling_seconds={k:v[0]['benchmark']['sampling_seconds'] for k,v in records.items()},
            peak_gpu_gib=max(v[0]['benchmark']['peak_gpu_gib'] for v in records.values()),
            peak_host_gib=max(v[0]['peak_host_gib'] for v in records.values()),
            max_mass_error=max(v[0]['decoder_max_mass_error'] for v in records.values())))
        checks+=3
    historical=[]
    for phase in ('ph012','ph013'):
        for cap in ('NGC','SGC'):
            density_path=BASE/'fullcap_density'/phase/cap/'d000/COMPLETE.json'
            tidal_path=BASE/'fullcap_tides'/phase/cap/'d000/COMPLETE.json'
            d=json.loads(density_path.read_text());t=json.loads(tidal_path.read_text())
            assert d['passed'] and t['passed'] and d['draws']==t['draws']==4
            assert d['phase']==t['phase']==phase and d['cap']==t['cap']==cap
            historical.append(dict(phase=phase,cap=cap,sampling_seconds=d['sampling_seconds'],
                loading_seconds=d['loading_seconds'],tidal_seconds=t['seconds'],
                tidal_setup_seconds=t['setup_seconds'],density_bytes=d['field']['bytes'],
                tidal_bytes=t['output']['bytes'],density=c.file_record(density_path,content_hash=True),
                tidal=c.file_record(tidal_path,content_hash=True)))
            checks+=2
    estimate=dict(second_model_sampling_gpu_hours=8*sum(r['sampling_seconds'] for r in historical)/3600,
        second_model_tides_gpu_hours=8*sum(r['tidal_seconds'] for r in historical)/3600,
        second_model_loading_and_tidal_setup_hours=8*sum(r['loading_seconds']+r['tidal_setup_seconds'] for r in historical)/3600,
        second_model_density_and_tidal_gib=8*sum(r['density_bytes']+r['tidal_bytes'] for r in historical)/2**30,
        optional_four_cap_full_nfe_baseline_replay_sampling_hours=sum(r['sampling_seconds'] for r in historical)/3600,
        interpretation='Estimate from existing NFE128 seed17 timings, not a measured seed29 full-length run; excludes verification, corrected-operator replay, scoring, queue delays and retries.')
    result=dict(passed=True,checks=checks,phase_results=phase_rows,historical=historical,estimate=estimate,
        inputs=receipts,summary=c.file_record(ROOT/'SUMMARY.json',content_hash=True),
        source=c.file_record(__file__,content_hash=True),
        scientific_question='Does mixing independently trained models repair tidal-gap and interior-fraction coverage?',
        answer='Unanswered: this pilot establishes shortened-trajectory comparability only.',
        full_nfe_parity=False,tidal_operator_tested=False,coverage_tested=False,
        posterior_qualified=False,confirmation_access=False,training=False)
    c.atomic_json(ROOT/'REVIEW.json',result)
    print(json.dumps(dict(passed=True,checks=checks,phase_results=phase_rows,estimate=estimate)),flush=True)


if __name__=='__main__':
    run()
