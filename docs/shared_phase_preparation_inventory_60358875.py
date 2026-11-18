"""Receipt inventory for the approved 20/3/2 preparation; no final-test payload IO."""
import json
from pathlib import Path
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_audit_worker as audits
from workflows.sbi import e2e_wholefield_pilot_data as whole


def inventory():
    c.require_compute()
    if not c.SHARED_POLICY:raise ValueError('Explicit shared campaign required')
    registry=c.REPO/'configs/p10_phase_registry_v3.json'
    reg=json.loads(registry.read_text())
    p12=Path(reg['path_templates']['phase_output'].format(phase='ph000')).parent
    phases=list(c.TRAIN+c.DEVELOPMENT);rows={}
    for phase in phases:
        c.phase_guard(phase)
        def passed(path,key='pass'):
            return path.is_file() and json.loads(path.read_text()).get(key) is True
        rows[phase]=dict(p12_phase=passed(p12/phase/'PHASE_COMPLETE.json'),
            p12_adapters=all(passed(p12/'training_contract_v1/adapters'/phase/k/'adapter_manifest.json') for k in ('graph','field')),
            native=passed(c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json'),
            particles=passed(c.ROOT/'matter'/phase/'PARTICLES_VERIFIED.json'),
            geometry=passed(coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'),
            conditions=passed(coord.ROOT/'conditions'/phase/'CONDITIONS_COMPLETE.json'),
            targets=passed(coord.ROOT/'targets'/phase/'TARGETS_COMPLETE.json'),
            audit=audits.qualified(phase),
            wholefield=passed(whole.ROOT/'data'/phase/'COMPLETE.json','passed'),
            p12_response=passed(p12/phase/'p3b_random_response_v1/P12A_RESPONSE_INPUT_READY.json'))
    contracts={}
    for name,path in {
        'p12_full':p12/'training_contract_v1/TRAINING_LOADER_READY.json',
        'p12_crossfit':p12/'crossfit_v1/P12_CROSSFIT_CONTRACTS_READY.json',
        'p12_responses':p12/'response_inputs_v1/adapter_inventory.json',
        'cfm_normalizer':coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json',
        'cfm_loaders':c.ROOT/'readiness/DATA_LOADER_CANARIES.json',
    }.items():
        value=json.loads(path.read_text()) if path.is_file() else {}
        contracts[name]=dict(path=str(path),passed=value.get('pass') is True,
                            sha256=c.sha256(path) if path.is_file() else None)
    counts={key:sum(row[key] for row in rows.values()) for key in next(iter(rows.values()))}
    return dict(**c.provenance(),phase_policy=c.SHARED_POLICY,preparation_phases=phases,
        counts=counts,phases=rows,contracts=contracts,
        all_preparation_receipts_pass=all(all(row.values()) for row in rows.values()) and all(v['passed'] for v in contracts.values()),
        final_test_payloads_opened=False,learned_models_fitted=False,
        scope='Deterministic data preparation; learned encoder summaries and posterior fits are subsequent work')

if __name__=='__main__':print(json.dumps(inventory(),indent=2))
