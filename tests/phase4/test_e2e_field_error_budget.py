from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import fft

from workflows.sbi.e2e_field_error_budget import (
    inverse_component, pair_sums, sample_fd_tensor, sample_grid,
    science, smoothed_spectrum, tensor_metrics,
)
from workflows.sbi.e2e_field_native_reference import sample_native_tensor


class ErrorBudgetTests(unittest.TestCase):
    def test_fullbox_spectral_plane_wave(self):
        n,box = 16,16.
        coords = np.indices((n,)*3)
        phase = 2*np.pi*(coords[0]+2*coords[1]+coords[2])/n
        counts = (4*(1+.1*np.cos(phase))).astype(np.float32)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"counts.npy"
            np.save(path,counts)
            spectrum,mean = smoothed_spectrum(path,box,.7,workers=1)
            expected = .1*np.exp(-.5*.7**2*(2*np.pi/n)**2*6)*np.cos(phase)
            xy = inverse_component(spectrum,box,(0,1),workers=1)
            np.testing.assert_allclose(xy,expected/3,atol=3e-8)
            trace = sum(inverse_component(spectrum,box,(a,a),workers=1) for a in range(3))
            np.testing.assert_allclose(trace,expected,atol=3e-8)
            self.assertAlmostEqual(mean,4.,places=6)

    def test_derivative_order_and_native_parity(self):
        n = 32
        grid = np.indices((n,)*3)
        wave = np.cos(2*np.pi*(grid[0]+grid[1]+grid[2])/n)
        axes = [np.array([0.,2.,13.,31.])]*3
        order2 = sample_fd_tensor(wave,axes,1.,2)
        np.testing.assert_allclose(order2,sample_native_tensor(wave,[a.astype(int) for a in axes],1.),atol=1e-14)
        exact = -(2*np.pi/n)**2*wave[np.ix_(*(a.astype(int) for a in axes))]
        errors = [np.max(np.abs(sample_fd_tensor(wave,axes,1.,order)[...,1]-exact)) for order in (2,4,8)]
        self.assertLess(errors[1],errors[0]/100)
        self.assertLess(errors[2],errors[1]/1000)

    def test_periodic_sampling_and_polynomial_cubic(self):
        grid = np.indices((12,)*3,dtype=float)
        x = grid[0]**3+2*grid[1]**2+grid[2]
        coords = [np.array([2.2,4.7])]*3
        expected = coords[0][:,None,None]**3+2*coords[1][None,:,None]**2+coords[2][None,None,:]
        np.testing.assert_allclose(sample_grid(x,coords,"local_cubic_lagrange"),expected,atol=1e-12)
        a = [np.array([-.3,11.7])]*3
        out = sample_grid(x,a,"linear")
        np.testing.assert_allclose(out,np.full((2,)*3,out[0,0,0]),atol=1e-10)
        np.testing.assert_array_equal(sample_grid(x,[np.array([11.7])]*3,"nearest_round"),x[:1,:1,:1])

    def test_pair_counts_match_brute_force(self):
        rng = np.random.default_rng(723)
        mask = rng.random((4,)*3)>.2
        mark = rng.random((4,)*3)>.5
        actual = pair_sums(mark,mask,1.,[1.,2.,4.])
        points = np.argwhere(mask)
        distance = np.linalg.norm(points[:,None,:]-points[None,:,:],axis=-1)
        marked = mark[mask]
        for record,(lo,hi) in zip(actual,[(1,2),(2,4)]):
            choose = (distance>=lo)&(distance<hi)
            self.assertEqual(record["ordered_supported_pairs"],int(choose.sum()))
            self.assertEqual(record["marked_pairs"],int((choose&marked[:,None]&marked[None,:]).sum()))

    def test_constant_tensor_error_is_not_random_scatter(self):
        rng = np.random.default_rng(19)
        truth = rng.normal(size=(4,4,4,6))
        pred = truth+np.array([.01,.02,0.,-.01,0.,0.])
        out = tensor_metrics(pred,truth,np.ones((4,)*3,dtype=bool))
        self.assertAlmostEqual(out["constant_tensor_residual_energy_fraction"],1.,places=12)
        self.assertLess(out["oracle_constant_removed_tensor_rmse"],1e-15)

    def test_science_functionals_and_terminal_connectivity(self):
        eigen = np.ones((32,32,32,3))
        mask = np.ones((32,)*3,dtype=bool)
        config = {"threshold":.2,"pair_bin_edges_mpc_h":[10.,20.,40.,80.]}
        out = science(eigen,mask,3.383,config)
        self.assertEqual(out["filling_fraction"],1.)
        self.assertEqual(out["connections_xyz"],[True,True,True])
        self.assertEqual(out["largest_void_fraction"],0.)
        self.assertTrue(all(x["value"]==1. for x in out["pair"]))
        eigen[:] = 0
        out = science(eigen,mask,3.383,config)
        self.assertEqual(out["connections_xyz"],[False,False,False])
        self.assertEqual(out["largest_void_fraction"],1.)


if __name__ == "__main__":
    unittest.main()
