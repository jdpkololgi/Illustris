import unittest
import numpy as np

from workflows.sbi.e2e_field_numerics import coordinate_noise, evolve_high, haar, low_projection


class NumericsTests(unittest.TestCase):
    def test_haar_roundtrip_and_norm(self):
        rng = np.random.default_rng(72)
        x = rng.normal(size=(16,)*3)
        for depth in (1,2):
            wave = haar(x,depth)
            np.testing.assert_allclose(haar(wave,depth,inverse=True),x,atol=3e-15)
            self.assertAlmostEqual(float(np.sum(wave**2)),float(np.sum(x*x)),places=9)

    def test_low_high_projectors_and_mean(self):
        x = np.random.default_rng(3).normal(size=(16,)*3)+.3
        low = low_projection(x)
        np.testing.assert_allclose(low_projection(low),low,atol=5e-16)
        np.testing.assert_allclose(low_projection(x-low),0,atol=5e-16)
        self.assertAlmostEqual(float(low.mean()),float(x.mean()))

    def test_noise_global_identity(self):
        big = coordinate_noise([-4,0,16],[16]*3,"phase:sample42")
        small = coordinate_noise([0,4,20],[8]*3,"phase:sample42")
        np.testing.assert_array_equal(big[4:12,4:12,4:12],small)
        self.assertFalse(np.array_equal(small,coordinate_noise([0,4,20],[8]*3,"phase:sample43")))

    def test_synchronized_heun_tiling(self):
        noise = coordinate_noise([0]*3,[16]*3,"test")
        direct = evolve_high(noise)
        np.testing.assert_array_equal(direct,evolve_high(noise,tiled=True,offset=8))
        np.testing.assert_allclose(low_projection(direct),0,atol=6e-16)


if __name__ == "__main__":
    unittest.main()
