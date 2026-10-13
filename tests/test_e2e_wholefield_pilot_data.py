import unittest
import numpy as np
from workflows.sbi import e2e_wholefield_pilot_data as d


class DataTests(unittest.TestCase):
    def test_sealed_before_io(self):
        for phase in ('ph018', 'ph019', 'ph001'):
            with self.assertRaises(PermissionError):
                d.select_rows(phase, [])

    def test_exposure_and_selection(self):
        rows = [dict(cap=cap, pair_id=f'ph000_{cap}_{i:02d}') for cap in ('NGC', 'SGC') for i in range(64)]
        chosen = d.select_rows('ph000', rows)
        self.assertEqual(len(chosen), 48)
        self.assertEqual(len({r['pair_id'] for r in chosen}), 48)
        self.assertEqual(48*np.prod(d.SHAPE), 128*64*48*48)

    def test_old_domain_alignment(self):
        self.assertEqual(tuple(s.stop-s.start for s in d.OLD_CROP), (64, 48, 48))
        self.assertEqual(tuple(4*(s.stop-s.start) for s in d.COARSE_CROP), d.SHAPE)


if __name__ == '__main__':
    unittest.main()
