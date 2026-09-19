from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from workflows.sbi import e2e_coupled_loader_benchmark as bench


class LoaderBenchmarkTests(unittest.TestCase):
    def test_non_training_rejected_before_receipt_access(self):
        with patch.object(bench.audits, 'qualified', side_effect=AssertionError('premature IO')):
            for phase in ('ph001', 'ph006', 'ph012', 'ph019'):
                with self.assertRaises(PermissionError):
                    bench.phase_cases(phase)

    def test_fixed_panel_spans_all_offsets_and_two_pairs_per_phase(self):
        record = {'pair_receipts': [{'path': f'/unused/pair_{i:03}.json'} for i in range(128)]}
        with patch.object(bench.audits, 'qualified', return_value=True), \
                patch.object(bench.coord, 'verify_receipt', return_value=record):
            cases = [case for phase in bench.c.TRAIN for case in bench.phase_cases(phase)]
        self.assertEqual(len(cases), 26)
        self.assertEqual({tuple(case['offset']) for case in cases},
                         {tuple(offset) for offset in bench.op.layout()['context_offsets_raw']})
        self.assertEqual({case['pair_id'] for case in cases}, {'pair_000', 'pair_064'})

    def test_cpu_collation_preserves_each_pair_without_modifying_arrays(self):
        first = ({'joint': np.ones((2, 3), dtype='f4')}, {'fine_residual': np.zeros((1, 3), dtype='f4')})
        second = ({'joint': np.full((2, 3), 2, dtype='f4')}, {'fine_residual': np.ones((1, 3), dtype='f4')})
        actual = bench.collate([first, second])
        np.testing.assert_array_equal(actual[0]['joint'].numpy(), np.stack([first[0]['joint'], second[0]['joint']]))
        actual[0]['joint'][0] = 9
        self.assertTrue(np.all(first[0]['joint'] == 1))
        with self.assertRaises(ValueError):
            bench.collate([first])

    def test_missing_full_normalizer_stops_before_any_data_probe(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench.coord, 'ROOT', Path(tmp)), \
                patch.object(bench.c, 'require_compute'), \
                patch.object(bench.norm, 'fit', side_effect=AssertionError('premature fit')):
            with self.assertRaises(FileNotFoundError):
                bench.run()


if __name__ == '__main__':
    unittest.main()
