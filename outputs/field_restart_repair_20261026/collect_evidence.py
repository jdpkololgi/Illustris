from pathlib import Path
import hashlib,json,shutil
repo=Path('/global/homes/d/dkololgi/TNG/Illustris');out=repo/'outputs/field_restart_repair_20261026'
scratch=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/restart_repair_20261026_v1')
products=Path('/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,source in {'ENDPOINT_READY.json':scratch/'endpoint_fixed/FIELD_UNET_READY.json',
 'ENCODER_READY.json':scratch/'encoder_fixed_pool/P12A_ENCODER_READY.json',
 'ENCODER_REPLAY.json':scratch/'encoder_fixed_pool/ENCODER_REPLAY.json',
 'ENDPOINT_ENTRYPOINT_REPLAY.json':scratch/'endpoint_entrypoint/REPLAY.json',
 'CROSSFIT_READY.json':products/'p12a_crossfit_20261026_v1/P12_CROSSFIT_CONTRACTS_READY.json'}.items():
 shutil.copy2(source,out/name)
paths=['shared/deterministic_interpolation.py','workflows/abacus_tweb/p8_train_unet_patch.py',
 'workflows/abacus_tweb/p10_train_arm_a.py','workflows/sbi/e2e_wholefield_pilot_train.py',
 'workflows/sbi/e2e_wholefield_unet_readiness.py','workflows/sbi/p12a_field_training_readiness.py',
 'workflows/sbi/p12_prepare_crossfit_contracts.py','workflows/sbi/p12_export_unet_summaries.py',
 'workflows/sbi/p12_prepare_base_response_dataset.py','workflows/sbi/p12_train_base_response_fmpe.py',
 'workflows/sbi/p12a_prepare_response_inputs.py','workflows/sbi/p12a_prepare_visible_responses.py',
 'workflows/sbi/p12a_parent_response.py','workflows/abacus_tweb/p3br_build_random_response.py',
 'configs/field_restart_numerics_20261026.json','configs/p12a_corrected_expanded_inputs_20261026.json']
(out/'SOURCE_BINDINGS.json').write_text(json.dumps({p:digest(repo/p) for p in paths},indent=2)+'\n')
folds={}
for fold in sorted((products/'p12a_crossfit_20261026_v1').glob('omit_*')):
 marker=fold/'TRAINING_LOADER_READY.json';r=json.loads(marker.read_text())
 omitted=fold.name.removeprefix('omit_');roles=r['roles']
 assert r['pass'] and omitted not in roles['training'] and len(roles['training'])==14
 assert roles['validation_and_selection']==omitted
 assert 'fixed training budget' in r['checkpoint_policy']
 assert roles['sealed_blind_test']==['ph001'] and roles['reserved_confirmation']==['ph014','ph015','ph016','ph017','ph018','ph019']
 folds[omitted]={'marker_sha256':digest(marker),'training_phases':roles['training'],'pass':True}
assert len(folds)==15
(out/'CROSSFIT_AUDIT.json').write_text(json.dumps({'pass':True,'folds':folds},indent=2)+'\n')
