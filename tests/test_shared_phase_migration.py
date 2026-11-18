import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from shared.phase_policy import REPO, binding, check_roles, load_policy, p12_roles
from workflows.abacus_tweb.p10_phase_assets import load_registry, validate_registry
from workflows.sbi.shared_phase_migration import records


class SharedPhaseMigrationTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()
        self.record = {'phase_policy': binding()}

    def test_full_and_all_twenty_oof_contracts(self):
        roles = p12_roles(self.policy)
        check_roles(self.record, roles)
        for phase in self.policy['training']:
            fold = copy.deepcopy(roles)
            fold['training'].remove(phase)
            fold['validation_and_selection'] = phase
            check_roles(self.record, fold, omitted=phase)
            fold['training'].append(phase)
            with self.assertRaises(ValueError): check_roles(self.record,fold,omitted=phase)

    def test_changed_hash_and_role_rejected(self):
        roles = p12_roles(self.policy)
        self.record['phase_policy']['sha256'] = 'bad'
        with self.assertRaises(ValueError): check_roles(self.record,roles)
        self.record['phase_policy'] = binding()
        roles['training'][0] = 'ph018'
        with self.assertRaises(ValueError): check_roles(self.record,roles)

    def test_old_registry_unchanged_and_new_registry_roles(self):
        old = load_registry(REPO/'configs/p10_phase_registry_v2.json')
        new = load_registry(REPO/'configs/p10_phase_registry_v3.json')
        self.assertEqual(old['phases']['ph001']['role'],'sealed_blind')
        self.assertEqual(new['phases']['ph001']['role'],'training')
        self.assertEqual(new['phases']['ph018']['role'],'sealed_blind')
        new['path_templates']['phase_output'] = new['reuse_source_root']+'/{phase}'
        with self.assertRaises(ValueError): validate_registry(new)

    def test_explicit_cfm_campaign_and_final_test_guard(self):
        env = dict(os.environ, GRAPHWEB_CFM_DATA_CONFIG='configs/e2e_coupled_data_20phase_20261027.json')
        code = '''from workflows.sbi import e2e_coupled_contract as c
assert len(c.TRAIN)==20 and len(c.DEVELOPMENT)==3
assert c.config()['target']['gaussian_radius_mpc_h']==7
c.phase_guard('ph001'); c.phase_guard('ph006')
for phase in ('ph018','ph019'):
    try: c.phase_guard(phase)
    except PermissionError: pass
    else: raise AssertionError('final test opened')
from workflows.sbi import e2e_coupled_coordinates as coord
assert str(coord.ROOT).startswith(str(c.ROOT))
'''
        subprocess.run([sys.executable,'-c',code],env=env,cwd=REPO,check=True)

    def test_preparation_and_loader_leave_final_tests_closed(self):
        from tempfile import TemporaryDirectory
        from workflows.abacus_tweb import p10_prepare_training_contract as prep
        from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
        registry=load_registry(REPO/'configs/p10_phase_registry_v3.json')
        try:
            prep.configure_phase_roles(registry)
            self.assertEqual(set(prep.PHASES),set(self.policy['training']+self.policy['development_and_calibration']))
            self.assertEqual(len(prep.PHASES),23)
            self.assertEqual(set(prep.BLIND_PHASES),{'ph018','ph019'})
            with TemporaryDirectory() as tmp:
                marker=Path(tmp)/'TRAINING_LOADER_READY.json'
                marker.write_text(json.dumps({'pass':True,'phase_policy':binding(),'roles':p12_roles(self.policy)}))
                with self.assertRaises(PermissionError):P10PhaseBalancedLoader(tmp,include_blind=True)
        finally:
            prep.configure_phase_roles(load_registry(REPO/'configs/p10_phase_registry_v1.json'))

    def test_wholefield_full_epoch_covers_all_twenty_phases(self):
        env = dict(os.environ, GRAPHWEB_CFM_DATA_CONFIG='configs/e2e_coupled_data_20phase_20261027.json')
        code = """from workflows.sbi.e2e_wholefield_pilot_train import selected, EPOCH_STEPS, STEPS
ids = [(phase, domain) for phase in range(20) for domain in range(48)]
assert EPOCH_STEPS == 480 and STEPS == 32 * EPOCH_STEPS
for epoch in (0, 31):
    order = [item for step in range(epoch*EPOCH_STEPS, (epoch+1)*EPOCH_STEPS) for item in selected(ids, 17, step)]
    assert len(order) == len(ids) and sorted(order) == ids
"""
        subprocess.run([sys.executable,'-c',code],env=env,cwd=REPO,check=True)

    def test_historical_cfm_defaults(self):
        env = dict(os.environ);env.pop('GRAPHWEB_CFM_DATA_CONFIG',None)
        code = "from workflows.sbi import e2e_coupled_contract as c; assert len(c.TRAIN)==13; assert c.config()['schema']=='e2e-coupled-data-v1'"
        subprocess.run([sys.executable,'-c',code],env=env,cwd=REPO,check=True)

    def test_hash_bound_artifact_records(self):
        value = {'artifacts': {'a':'/tmp/a'},'artifact_sha256':{'a':'hash'},'outputs':[{'path':'/tmp/b','sha256':'other'}]}
        self.assertEqual({r['path'] for r in records(value)}, {'/tmp/a','/tmp/b'})

    def test_repeated_sources_hash_once_and_reject_conflicts(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from workflows.sbi import shared_phase_migration as migration
        with TemporaryDirectory() as tmp:
            source = Path(tmp)/'payload'; source.write_bytes(b'example')
            record = {'path':str(source),'sha256':migration.sha256(source),'bytes':7}
            verified = {}
            with patch.object(migration,'sha256',wraps=migration.sha256) as hashing:
                migration.verify_file(record,verified)
                migration.verify_file(record,verified)
                self.assertEqual(hashing.call_count,1)
                with self.assertRaises(ValueError):
                    migration.verify_file(dict(record,sha256='wrong'),verified)
                source.write_bytes(b'changed-length')
                with self.assertRaises(ValueError): migration.verify_file(record,verified)


if __name__ == '__main__': unittest.main()
