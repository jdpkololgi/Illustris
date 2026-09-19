import builtins
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_interface_qualify as qa


class InterfaceQualificationTests(unittest.TestCase):
    def test_observation_io_guard_allows_only_own_conditions_chart_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); condition=root/'conditions/ph007/input.txt'
            condition.parent.mkdir(parents=True); condition.write_text('ok')
            target=root/'targets/ph007/target.h5'; target.parent.mkdir(parents=True)
            with h5py.File(target,'w') as f: f['x']=[1]
            chart=root/'normalization/NORMALIZATION_COMPLETE.json'
            chart.parent.mkdir(); chart.write_text('{}')
            with patch.object(qa.coord,'ROOT',root), qa.observation_io_only('ph007'):
                self.assertEqual(condition.read_text(),'ok')
                with builtins.open(chart) as f: self.assertEqual(f.read(),'{}')
                for opener in (lambda:target.read_bytes(),lambda:builtins.open(target),lambda:h5py.File(target,'r')):
                    with self.assertRaises(PermissionError): opener()
                with self.assertRaises(PermissionError): condition.write_text('bad')
                with self.assertRaises(PermissionError): (root/'conditions/ph008/input.txt').read_text()
            self.assertEqual(condition.read_text(),'ok')
            with h5py.File(target) as f: self.assertEqual(f['x'][0],1)

    def test_frozen_positive_roundtrip_tolerance(self):
        self.assertLess(qa.relative_error([1.+1e-6],[1.]),2e-6)
        for actual in ([1.01],[0.],[-1.],[np.nan],[1.,2.]):
            with self.assertRaises(ValueError): qa.relative_error(actual,[1.])

    def test_missing_normalizer_fails_before_any_phase_access(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(qa.coord,'ROOT',Path(tmp)), \
                patch.object(qa.c,'require_compute'),patch.object(qa.coord,'require_host_checks'), \
                patch.object(qa,'qualify_phase',side_effect=AssertionError('premature payload access')):
            with self.assertRaises(FileNotFoundError): qa.run()


if __name__=='__main__': unittest.main()
