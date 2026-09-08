from pathlib import Path
import tempfile
import unittest

import numpy as np

from workflows.sbi.e2e_field_native_reference import (
    stored_npz_array, sample_native_tensor, potential_from_counts,
)


class NativeReferenceTests(unittest.TestCase):
    def test_stored_npz_mapping(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"source.npz"
            x = np.arange(3*4*8*8, dtype=np.float32).reshape(3,4,8,8)
            np.savez(path, eig_vals=x, x_start=0)
            actual = stored_npz_array(path, "eig_vals")
            self.assertIsInstance(actual, np.memmap)
            self.assertFalse(actual.flags.writeable)
            np.testing.assert_array_equal(actual, x)

    def test_compressed_native_fails(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"compressed.npz"
            np.savez_compressed(path, eig_vals=np.zeros((3,4,8,8)))
            with self.assertRaises(ValueError):
                stored_npz_array(path, "eig_vals")

    def test_derivative_matches_native_vendored_implementation(self):
        from cactus.ext import fiesta
        rng = np.random.default_rng(871)
        phi = rng.normal(size=(12,12,12))
        cell = .9765625
        axes = [np.array([0,1,5,11])]*3
        actual = sample_native_tensor(phi, axes, cell)
        x = (np.arange(12)+.5)*cell
        derivatives = [fiesta.maths.dfdx, fiesta.maths.dfdy, fiesta.maths.dfdz]
        for column, (a,b) in enumerate(((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))):
            full = derivatives[b](x, derivatives[a](x,phi,periodic=True),periodic=True)
            np.testing.assert_allclose(actual[...,column],full[np.ix_(*axes)],atol=2e-15)

    def test_plane_wave_potential_normalization_and_sign(self):
        n, box, smooth = 16, 16., .7
        x = np.arange(n)[:,None,None]
        counts = (4*(1+.1*np.cos(2*np.pi*x/n))*np.ones((1,n,n))).astype(np.float32)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"counts.npy"
            np.save(path, counts)
            phi, mean = potential_from_counts(path, box, smooth, workers=1)
            k = 2*np.pi/box
            expected = -.1*np.exp(-.5*smooth*smooth*k*k)*np.cos(2*np.pi*x/n)/(k*k)
            np.testing.assert_allclose(phi, np.broadcast_to(expected,phi.shape), atol=2e-7)
            self.assertAlmostEqual(mean,4.0,places=6)


if __name__ == "__main__":
    unittest.main()
