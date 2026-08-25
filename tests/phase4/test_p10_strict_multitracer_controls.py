import unittest

import healpy as hp
import numpy as np
import torch

from workflows.abacus_tweb.p10_build_strict_multitracer_controls import (
    DONOR_MAPS,
    fine_stratum,
    subpixel_angles,
    validate_donor_map,
)
from workflows.abacus_tweb.p10_multitracer_training import P10MultitracerUPatch


class StrictMultitracerControlTest(unittest.TestCase):
    def test_donor_maps_are_complete_derangements(self):
        for mapping in DONOR_MAPS.values():
            validate_donor_map(mapping)
            self.assertTrue(all(recipient != donor for recipient, donor in mapping.items()))

    def test_fine_stratum_contract(self):
        redshift = np.array([0.10, 0.1099, 0.11, 0.5999, 0.60])
        np.testing.assert_array_equal(fine_stratum(redshift), [0, 0, 1, 49, -1])

    def test_subpixels_remain_inside_parent(self):
        parent = np.array([0, 1, 1000, hp.nside2npix(256) - 1], dtype=np.int64)
        ra, dec = subpixel_angles(parent, np.random.default_rng(42))
        recovered = hp.ang2pix(256, ra, dec, nest=False, lonlat=True)
        np.testing.assert_array_equal(recovered, parent)

    def test_u_patch_is_pointwise_not_three_channel_voxel_output(self):
        model = P10MultitracerUPatch(base=2, latent_channels=4, head_width=8)
        field = torch.zeros((1, 6, 8, 8, 8), dtype=torch.float32)
        points = torch.zeros((1, 1, 1, 3, 3), dtype=torch.float32)
        latent = model.unet(field)
        prediction = model(field, points)
        self.assertEqual(tuple(latent.shape), (1, 4, 8, 8, 8))
        self.assertEqual(tuple(prediction.shape), (3, 3))


if __name__ == "__main__":
    unittest.main()
