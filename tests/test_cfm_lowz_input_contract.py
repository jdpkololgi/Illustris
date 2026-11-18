"""Small, dependency-free guard/geometry tests; no scientific payload access."""
import copy
import json
from pathlib import Path
import unittest

from workflows.sbi.prepare_cfm_lowz_inputs import box_min_radius, validate_plan


class LowzInputContractTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'configs/e2e_cfm_lowz_input_canary_20261028.json'
        self.plan = json.loads(path.read_text())

    def test_frozen_scope(self):
        validate_plan(self.plan)

    def test_final_test_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan['frozen_training_phases'][0] = 'ph018'
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_phase_overlap_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan['frozen_training_phases'][0] = 'ph012'
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_bad_bins_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan['catalogue_audit_bin_edges'][2] = 0.
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_box_contains_origin(self):
        row = dict(center=[0, 0, 0], grid=dict(origin_mpc_h=[-16, 0, 0], cell_mpc_h=1.))
        self.assertEqual(box_min_radius(row, [2, 2, 2], 1.), 0.)

    def test_box_nearest_face(self):
        row = dict(center=[0, 0, 0], grid=dict(origin_mpc_h=[0, 0, 0], cell_mpc_h=1.))
        self.assertEqual(box_min_radius(row, [2, 2, 2], 1.), 15.)

    def test_box_nearest_corner(self):
        row = dict(center=[0, 0, 0], grid=dict(origin_mpc_h=[-12, 5, 0], cell_mpc_h=1.))
        self.assertEqual(box_min_radius(row, [2, 2, 2], 1.), 5.)


if __name__ == '__main__':
    unittest.main()
