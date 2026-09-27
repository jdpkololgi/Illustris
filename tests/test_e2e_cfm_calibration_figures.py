import unittest
import numpy as np
from workflows.sbi.e2e_cfm_calibration_figures import regional,tarp_curves
from workflows.sbi.e2e_cfm_pilot_evaluate import regional as reference

class CalibrationFigureTests(unittest.TestCase):
    def test_region_parity(self):
        x=np.random.default_rng(5).normal(size=(2,64,48,48))
        np.testing.assert_allclose(regional(x),np.stack([reference(v) for v in x]),atol=1e-14)
    def test_randomized_tarp_null(self):
        rng=np.random.default_rng(12)
        x=rng.normal(size=(2048,32,2));y=rng.normal(size=(2048,2))
        alpha,curves=tarp_curves(x,y)
        self.assertLess(abs(curves.mean(0)-alpha).max(),.04)
        _,narrow=tarp_curves(.1*x,y)
        self.assertGreater(abs(narrow.mean(0)-alpha).max(),.1)
    def test_tarp_ties(self):
        alpha,curves=tarp_curves(np.zeros((2048,32,2)),np.zeros((2048,2)))
        self.assertLess(abs(curves.mean(0)-alpha).max(),.02)
