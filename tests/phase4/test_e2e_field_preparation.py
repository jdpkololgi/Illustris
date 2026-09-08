"""Scientific failure checks for allocation-free E2E preparation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from workflows.sbi.e2e_field_prepare_data import (
    REPO, footprints_overlap, periodic_intervals, propose_parents, read_json,
    safe_path, validate_config,
)
from workflows.sbi.e2e_field_calibration_fixture import gaussian_condition, make_fixture, mira_per_parent


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((REPO / "configs/e2e_field_data_prep_v1.json").read_text())

    def test_reserved_phase_rejected_before_paths_open(self):
        self.config["development_pool"].append("ph001")
        with self.assertRaises(PermissionError):
            validate_config(self.config)
        with self.assertRaises(PermissionError):
            read_json(Path("/absent/ph001/manifest.json"))

    def test_reserved_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "ph001"
            source.mkdir()
            (root / "alias").symlink_to(source, target_is_directory=True)
            with self.assertRaises(PermissionError):
                safe_path(root / "alias/manifest.json")

    def test_unit_mismatch_rejected(self):
        self.config["coordinate_contract"]["cell_mpc_h"] = 5.0
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_payload_read_cannot_be_enabled(self):
        self.config["bounded_io"]["read_hdf5_payload"] = True
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_periodic_alias_overlaps_across_edge(self):
        a = periodic_intervals(1990, 2010, 2000)
        b = periodic_intervals(0, 5, 2000)
        self.assertTrue(footprints_overlap([a] * 3, [b] * 3))
        self.assertFalse(footprints_overlap([a] * 3, [[[10, 20]]] * 3))

    def test_matched_geometry_and_child_ownership(self):
        grid = {"shape": [160, 176, 192], "cell_mpc": 5.0, "origin_mpc": [-500.0] * 3}
        parents = propose_parents("ph000", "NGC", grid, self.config)
        self.assertEqual(parents, propose_parents("ph000", "NGC", grid, self.config))
        for parent in parents:
            side = parent["stop"][0] - parent["start"][0]
            self.assertAlmostEqual(parent["side_mpc_h"], side * 3.383)
            self.assertFalse(parent["support_screened"])
            self.assertFalse(parent["split_assigned"])
            occupied = set()
            for child in parent["children"]:
                key = tuple(child["core_start"])
                self.assertNotIn(key, occupied)
                occupied.add(key)
                for axis in range(3):
                    self.assertLessEqual(child["evaluation_start"][axis], child["core_start"][axis])
                    self.assertGreaterEqual(child["evaluation_stop"][axis], child["core_stop"][axis])
            self.assertEqual(len(occupied) * 32**3, side**3)


class ScientificFixtureTests(unittest.TestCase):
    def test_scalar_gaussian_posterior_exact(self):
        mean, cov = gaussian_condition(np.array([[4.0]]), np.ones((1, 1)), np.array([[1.0]]), np.array([3.0]))
        np.testing.assert_allclose(mean, [2.4])
        np.testing.assert_allclose(cov, [[0.8]])

    def test_covariance_and_gain_identity(self):
        arrays, report = make_fixture(parents=8, draws=8)
        for i in range(8):
            cov = arrays["posterior_covariance"][i]
            self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)
            A = arrays["observation_matrix"]
            precision = np.linalg.inv(arrays["prior_covariance"]) + A.T @ A / arrays["noise_sigma"][i]**2
            np.testing.assert_allclose(cov @ precision, np.eye(12), atol=1e-12)
        self.assertLess(report["analytic_independent_sibling_sum_variance"], report["analytic_mean_sum_variance"])
        self.assertFalse(report["mira_activated"])

    def test_mira_excludes_boundary_sample(self):
        draws = np.array([[[0.0], [2.0], [1.0]]])
        score = mira_per_parent(np.array([[0.5]]), draws, np.zeros((1, 1)))
        np.testing.assert_allclose(score, [0.5])  # n=1, N=2; boundary draw excluded

    def test_orthogonal_transform_alone_does_not_change_cfm_loss(self):
        rng = np.random.default_rng(71)
        q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        error = rng.normal(size=12)
        self.assertAlmostEqual(float(error @ error), float((q @ error) @ (q @ error)))

    def test_shared_noise_does_not_make_independent_crop_evolution_equal(self):
        initial = np.arange(12, dtype=float) ** 2
        def velocity(x):
            padded = np.pad(x, 1)
            return padded[:-2] + padded[2:] - 2 * x
        parent, a, b = initial.copy(), initial[:8].copy(), initial[4:].copy()
        for _ in range(4):
            parent += 0.1 * velocity(parent)
            a += 0.1 * velocity(a)
            b += 0.1 * velocity(b)
        # Identical initial overlap cannot prevent boundary effects propagating inward.
        self.assertGreater(np.max(np.abs(a[4:8] - b[:4])), 0.01)
        # Synchronous tiling evaluates each halo from the same current parent state.
        tiled = initial.copy()
        for _ in range(4):
            update = np.empty_like(tiled)
            update[:6] = velocity(tiled[:7])[:6]
            update[6:] = velocity(tiled[5:])[1:]
            tiled += 0.1 * update
        np.testing.assert_allclose(tiled, parent, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
