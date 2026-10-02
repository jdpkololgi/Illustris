import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_exterior_final_score as scorer
from workflows.sbi.e2e_exterior_final_score import frozen, truth


class FinalAuthorityTests(unittest.TestCase):
    def test_sealed_and_training_phases_before_io(self):
        with patch.object(Path, 'read_text', side_effect=AssertionError('Unexpected file access')):
            for phase in ('ph001', 'ph000', 'ph002', 'ph024', 'Loa'):
                with self.assertRaises(PermissionError):
                    frozen(phase, Path('/not/opened'))
                with self.assertRaises(PermissionError):
                    truth(phase, 'unused', Path('/not/opened'))

    def test_fullcap_readback_metrics_and_joint_assembly(self):
        # A deterministic homogeneous synthetic universe is a useful exact
        # oracle for array placement, class probabilities and proper scores.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); freeze = root/'FREEZE.json'; freeze.write_text('{}')
            rows = []
            for cap in ('NGC', 'SGC'):
                for case in range(8):
                    rows.append(dict(pair_id=f'ph012_{cap}_oracle{case}', cap=cap, shell=case//2,
                        support_stratum='interior', center=[48, 48, 48], grid=dict(origin_mpc_h=[0., 0., 0.])))
                for first in range(0, 32, 4):
                    directory = root/'samples'/'ph012'/cap/f'd{first:03d}'; directory.mkdir(parents=True)
                    output = directory/'FIELD.h5'
                    with h5py.File(output, 'w') as f:
                        f.create_dataset('rho_completed', shape=(4, 128, 112, 112), dtype='f4', fillvalue=1.)
                        f.create_dataset('eigenvalues', shape=(4, 64, 48, 48, 3), dtype='f4', fillvalue=0.)
                        f.create_dataset('support', shape=(64, 48, 48), dtype='f4', fillvalue=1.)
                        f.attrs.update(origin_mpc_h=[0., 0., 0.], cell_mpc_h=6.766, padding_cells=32)
                    c.atomic_json(directory/'BINDING.json', dict(freeze_sha256=c.sha256(freeze)))
                    c.atomic_json(directory/'COMPLETE.json', dict(phase='ph012', cap=cap, first=first,
                        draws=4, method='learned', passed=True, binding=c.sha256(directory/'BINDING.json'),
                        output=c.file_record(output, content_hash=True)))
            values = dict(rho_joint=np.ones((64, 48, 48)), fullbox_tensor_cores=np.zeros((2, 16, 16, 16, 6)))
            with patch.object(scorer, 'frozen', return_value=dict(draws=32, method='learned')), \
                 patch.object(scorer, 'geometry', return_value=rows), \
                 patch.object(scorer, 'truth', return_value=(values, {})), \
                 patch.object(c, 'require_compute'), \
                 contextlib.redirect_stdout(io.StringIO()):
                scorer.score('ph012', root/'samples', freeze, root/'scores')
            result = json.loads((root/'scores'/'SCORES.json').read_text())
            self.assertTrue(result['panel_complete'])
            self.assertEqual(len(result['cases']), 16)
            for case in result['cases']:
                metrics = case['metrics']
                self.assertEqual(metrics['density']['mse'], 0.)
                self.assertEqual(metrics['density']['coverage']['0.9'], 1.)
                self.assertEqual(metrics['brier_0.0'], 0.)
                self.assertEqual(metrics['brier_0.2'], 0.)
            for name, expected_mass, expected_tidal in [('NGC', 16, 48), ('SGC', 16, 48),
                    ('both_caps_independence_approximation', 32, 96)]:
                joint = result['joint_fullcap_summaries'][name]
                self.assertEqual(joint['mass']['rank']['dimensions'], expected_mass)
                self.assertEqual(joint['tidal']['rank']['dimensions'], expected_tidal)
                self.assertEqual(joint['tidal']['energy'], 0.)


if __name__ == '__main__':
    unittest.main()
