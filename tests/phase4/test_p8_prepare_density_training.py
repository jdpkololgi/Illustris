import unittest

from astropy.cosmology import Planck18
import numpy as np

from workflows.abacus_tweb.p8_prepare_density_training import (
    RunningMoments,
    shell_index_for_core,
    unit_array,
)


class PrepareDensityTrainingTests(unittest.TestCase):
    def test_running_moments_matches_numpy(self):
        values = np.array([-2.0, -0.5, 1.0, 3.0])
        moments = RunningMoments()
        moments.add(values[:2])
        moments.add(values[2:])
        got = moments.as_dict()
        self.assertEqual(got["n"], len(values))
        self.assertAlmostEqual(got["mean"], float(values.mean()))
        self.assertAlmostEqual(got["std"], float(values.std()))
        self.assertEqual(got["minimum"], -2.0)
        self.assertEqual(got["maximum"], 3.0)

    def test_shell_index_uses_observer_mpc(self):
        centres = np.array([
            Planck18.comoving_distance(z).value for z in (0.2, 0.3, 0.4, 0.5)
        ])
        # One x cell per shell with negligible y/z offsets.
        for expected, centre in enumerate(centres):
            got = shell_index_for_core(
                np.array([0, 0, 0]), np.array([1, 1, 1]),
                origin_mpc=np.array([centre - 2.5, -2.5, -2.5]), cell_mpc=5.0,
            )
            self.assertEqual(int(got[0, 0, 0]), expected)

    def test_unit_array_preserves_exact_identifiers(self):
        rows = unit_array([(7, 11, 1, 3, 2, 419, 0.0)])
        self.assertEqual(int(rows["output_core_id"][0]), 7)
        self.assertEqual(int(rows["nominal_core_id"][0]), 11)
        self.assertEqual(int(rows["supported_voxels"][0]), 419)


if __name__ == "__main__":
    unittest.main()
