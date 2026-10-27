"""Collect metadata evidence only; payload hashes are checked on compute nodes."""
from pathlib import Path
import hashlib,json,re
out=Path(__file__).resolve().parent
root=Path('/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020')
endpoint=json.loads((out/'ENDPOINT_READY.json').read_text())
encoder=json.loads((out/'ENCODER_READY.json').read_text())
replay=json.loads((out/'ENDPOINT_ENTRYPOINT_REPLAY.json').read_text())
folds=json.loads((out/'CROSSFIT_AUDIT.json').read_text())
progress=root/'p12a_response_inputs_20261026_v1/PROGRESS.json'
inventory=root/'p12a_response_inputs_20261026_v1/adapter_inventory.json'
full=json.loads((root/'training_contract_20261021/TRAINING_LOADER_READY.json').read_text())
visible=full['roles']['training']+full['roles']['validation_and_selection']
response={'pass':False,'required_phases':visible,'complete_phases':[]}
if progress.exists():response['complete_phases']=json.loads(progress.read_text())['complete']
if inventory.exists():
 record=json.loads(inventory.read_text())
 assert record['pass'] and set(record['phases'])==set(visible)
 assert record['full_contract_sha256']==hashlib.sha256((root/'training_contract_20261021/TRAINING_LOADER_READY.json').read_bytes()).hexdigest()
 for phase,row in record['phases'].items():
  assert hashlib.sha256(Path(row['parent_response']).read_bytes()).hexdigest()==row['parent_response_sha256']
 response.update({'pass':True,'inventory':str(inventory),'inventory_sha256':hashlib.sha256(inventory.read_bytes()).hexdigest(),
                  'complete_phases':sorted(record['phases']),
                  'parent_rows':sum(row['rows'] for row in record['phases'].values()),
                  'supported_parent_rows':sum(row['support_rows'] for row in record['phases'].values())})
 (out/'RESPONSE_INPUTS_READY.json').write_text(json.dumps(record,indent=2)+'\n')
log=(out/'final-inputs-60300734.log').read_text()
tests=re.search(r'Ran (\d+) tests in [^\n]+\n\nOK',log)
result={'scope':'training-input and numerical readiness; no scientific fits or posterior qualification',
        'endpoint_training_ready':bool(endpoint['training_ready'] and replay['pass']),
        'p12a_encoder_training_ready':bool(encoder['encoder_training_ready'] and folds['pass']),
        'p12a_crossfit_folds':len(folds['folds']),'response_inputs':response,
        'p12a_posterior_training_ready':False,
        'remaining':['register and fit 15 fixed-budget out-of-fold encoders plus one full-training encoder',
                     'export corrected-label halo48 predictions','build and validate aligned posterior tables'],
        'focused_regression_tests':{'pass':bool(tests),'count':int(tests.group(1)) if tests else None},
        'graph_index_refresh_pass':bool(re.search(r'FINAL_EXIT_CODES inputs=\d+ tests=0 index=0',log)),
        'reserved_phases_opened':False,'scientific_fits_launched':False,'allocation':'60300734'}
if not response['pass']:
 result['remaining'].insert(0,'finish visible-phase response inputs and publish verified complete inventory')
(out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
