import unittest

import healpy as hp
import numpy as np

from workflows.abacus_tweb.p10_build_r2_assignment_overlays import (
    R2_MODEL_CHANNELS,
    R2_STORED_CHANNELS,
    angular_pixels,
    build_component,
)
from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec


class TestP10R2AssignmentOverlays(unittest.TestCase):
    def test_angular_pixels_follow_cell_centres(self):
        spec = GridSpec(
            origin=(10.0, 20.0, 30.0),
            shape=(2, 2, 2),
            cell_mpc=5.0,
            padding_mpc=0.0,
        )
        found = angular_pixels(spec, (slice(0, 2), slice(0, 2), slice(0, 2)))
        axes = [np.asarray([12.5, 17.5]), np.asarray([22.5, 27.5]), np.asarray([32.5, 37.5])]
        xx, yy, zz = np.meshgrid(*axes, indexing="ij")
        radius = np.sqrt(xx * xx + yy * yy + zz * zz)
        expected = hp.vec2pix(256, xx / radius, yy / radius, zz / radius, nest=False)
        np.testing.assert_array_equal(found, expected)

    def test_constant_redshift_success_is_stored_but_not_model_input(self):
        self.assertIn("c_z", R2_STORED_CHANNELS)
        self.assertIn("c_z_informative", R2_STORED_CHANNELS)
        self.assertNotIn("c_z", R2_MODEL_CHANNELS)
        self.assertNotIn("c_z_informative", R2_MODEL_CHANNELS)
        self.assertIn("c_fibre_defined", R2_MODEL_CHANNELS)

    def test_blind_phase_is_rejected_before_io(self):
        with self.assertRaises(ValueError):
            build_component(None, None, "ph001", "NGC")


if __name__ == "__main__":
    unittest.main()
