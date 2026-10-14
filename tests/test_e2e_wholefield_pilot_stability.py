import unittest

from workflows.sbi.e2e_wholefield_pilot_stability import select_pairs


class StabilitySelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(cap=cap, shell=shell, support_stratum=stratum,
                     pair_id=f'{cap}_{shell}_{stratum}')
                for cap in ('NGC', 'SGC') for shell in range(4)
                for stratum in ('boundary', 'interior')]

    def test_geometry_balance_and_order_invariance(self):
        rows = self.rows()
        a = select_pairs('ph012', rows)
        self.assertEqual(a, select_pairs('ph012', rows[::-1]))
        b = select_pairs('ph013', rows)
        self.assertEqual(len(a), 8)
        self.assertEqual(sum(r['support_stratum'] == 'boundary' for r in a), 4)
        self.assertEqual({r['pair_id'] for r in a} & {r['pair_id'] for r in b}, set())
        self.assertEqual({r['pair_id'] for r in a+b}, {r['pair_id'] for r in rows})

    def test_no_evaluation_or_sealed_selection(self):
        for phase in ('ph000', 'ph014', 'ph018', 'ph019'):
            with self.assertRaises(PermissionError):
                select_pairs(phase, self.rows())

    def test_missing_or_duplicate_geometry_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            select_pairs('ph012', rows[1:])
        with self.assertRaises(ValueError):
            select_pairs('ph012', rows+[rows[0]])


if __name__ == '__main__':
    unittest.main()
