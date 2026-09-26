import unittest
import math
from workflows.sbi.e2e_coarse_regularization import lr_at,evaluation_panel
from workflows.sbi.e2e_coarse_controls import panel,OPEN

class RegularizationTests(unittest.TestCase):
    def test_schedule_not_restarted(self):
        for step in (13312,19968,26623):
            expected=1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*step/26624))
            self.assertAlmostEqual(lr_at(step),expected,places=15)
        self.assertLess(lr_at(13312),lr_at(0))

    def test_panel_only_existing_exposed_cases(self):
        chosen=evaluation_panel();self.assertEqual(len(chosen),24)
        self.assertTrue(set(chosen)<=set(panel()))
        for phase in OPEN:self.assertEqual(sum(p==phase for p,_ in chosen),4)
        self.assertFalse(any(p in ('ph016','ph017','ph018','ph019') for p,_ in chosen))
        keys={tuple(pid.split('_')[1:4]) for p,pid in chosen if p in OPEN}
        self.assertEqual(len(keys),16)

if __name__=='__main__':unittest.main()
