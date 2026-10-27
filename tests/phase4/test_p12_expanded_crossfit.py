import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from workflows.sbi import p12_prepare_crossfit_contracts as c
from workflows.abacus_tweb.p10_training_contract import PatchRef


class ExpandedCrossfitTests(unittest.TestCase):
    def fixtures(self, root):
        full = root / 'full'
        (full / 'transforms/field').mkdir(parents=True)
        roles = {'training': ['ph000', 'ph007', 'ph024'], 'validation_and_selection': ['ph006','ph012','ph013'],
                 'sealed_blind_test': ['ph001'], 'reserved_confirmation': ['ph014']}
        (full / 'TRAINING_LOADER_READY.json').write_text(json.dumps({'pass': True, 'roles': roles}))
        diagnostics = {p: {'mean': [float(i)]*3, 'std': [1.]*3} for i,p in enumerate(roles['training'])}
        (full / 'transforms/target_scaler.json').write_text(json.dumps({'per_phase_diagnostics': diagnostics}))
        return full, roles

    def test_scaler_excludes_omitted_phase(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); full, _ = self.fixtures(root)
            output = root/'out'; (output/'transforms').mkdir(parents=True)
            result = c.leave_one_out_target_scaler(output=output, train=('ph000','ph024'), full_contract=full)
            np.testing.assert_allclose(result['mean'], [1]*3)
            np.testing.assert_allclose(result['std'], [2**.5]*3)
            self.assertNotIn('ph007', result['per_phase_diagnostics'])

    def test_expanded_roles_preserved_and_omission_enforced(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); full, roles = self.fixtures(root)
            loader = MagicMock()
            loader.training_epoch.return_value = (PatchRef('ph000',0,0,1.), PatchRef('ph024',0,1,1.))
            loader.validation_refs.return_value = (PatchRef('ph007',0,0,1.),)
            with patch.object(c, 'materialize_phase_contract', return_value={'pass':True}), \
                 patch.object(c, 'selection_fit', return_value={'pass':True}), \
                 patch.object(c, 'leave_one_out_field_transform', return_value={'pass':True}), \
                 patch.object(c, 'leave_one_out_target_scaler', return_value={'pass':True}), \
                 patch.object(c, 'P10PhaseBalancedLoader', return_value=loader):
                result = c.build_contract(root=root/'folds', omitted='ph007', full_contract=full)
            self.assertEqual(result['roles']['training'], ['ph000','ph024'])
            self.assertEqual(result['roles']['reserved_confirmation'], ['ph014'])
            self.assertEqual(result['roles']['sealed_blind_test'], ['ph001'])
            self.assertTrue(result['pass'])
            with self.assertRaises(ValueError):
                c.build_contract(root=root/'bad', omitted='ph014', full_contract=full)

    def test_stale_existing_array_and_link_fail(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); source=root/'source'; dest=root/'dest'
            source.write_bytes(b'new'); dest.write_bytes(b'old')
            with self.assertRaises(RuntimeError): c.hardlink_or_copy(source,dest)
            with self.assertRaises(RuntimeError): c.link_directory(source,dest)

    def test_selection_uses_only_fit_phase_statistics(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp)
            record={'counts':np.full((2,100),10,dtype=np.int64), 'volume':np.ones((2,100)), 'audit':{}}
            stats={'ph000':record,'ph007':{'counts':np.full((2,100),100000),'volume':np.ones((2,100)),'audit':{}}}
            def fit(centers, counts, volume, grid, **kwargs):
                np.testing.assert_array_equal(counts,10)
                return np.full(len(grid),10.), {}
            with patch.object(c, 'fit_log_spline', side_effect=fit):
                result=c.selection_fit(root=out,output=out,train=('ph000',),omitted='ph007',statistics=stats)
            self.assertTrue(result['pass']); self.assertEqual(list(result['sources']),['ph000'])

if __name__ == '__main__': unittest.main()
