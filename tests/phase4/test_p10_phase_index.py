import unittest

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from workflows.abacus_tweb.p10_build_phase_index import cartesian_points, shell_and_masks


class P10PhaseIndexTests(unittest.TestCase):
    def test_galactic_cap_matches_astropy(self):
        ra = np.asarray([0.0, 45.0, 120.0, 210.0, 359.0])
        dec = np.asarray([-40.0, 0.0, 25.0, 60.0, -5.0])
        points = cartesian_points(ra, dec, np.full(len(ra), 0.2))
        expected = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic.b.deg > 0
        self.assertEqual(points[:, 3].astype(bool).tolist(), expected.tolist())

    def test_shells_active_and_context_are_distinct(self):
        z = np.asarray([0.09, 0.12, 0.20, 0.30, 0.40, 0.50, 0.59, 0.61])
        valid = np.ones(len(z), dtype=bool)
        shell, active, context = shell_and_masks(z, valid)
        self.assertEqual(shell.tolist(), [-1, -1, 0, 1, 2, 3, -1, -1])
        self.assertEqual(active.tolist(), [False, False, True, True, True, True, False, False])
        self.assertEqual(context.tolist(), [False, True, True, True, True, True, False, False])

    def test_invalid_truth_never_active(self):
        shell, active, context = shell_and_masks(np.asarray([0.2, 0.2]), np.asarray([True, False]))
        self.assertEqual(shell.tolist(), [0, 0])
        self.assertEqual(active.tolist(), [True, False])
        self.assertEqual(context.tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
