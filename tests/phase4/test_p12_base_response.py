import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch

from workflows.sbi.p12_prepare_base_response_dataset import (
    sample_random_support_distance,
    softplus_coordinates,
    stratified_indices,
)
from workflows.sbi.p12_train_base_response_fmpe import (
    paired_posterior_log_prob,
    theta_to_eigenvalues,
    weighted_coverage,
    weighted_r2,
)


class P12BaseResponseTests(unittest.TestCase):
    def test_softplus_coordinates_round_trip_and_order(self):
        eigen = np.asarray(
            [[-0.4, -0.1, 0.2], [0.1, 0.100001, 0.9]], dtype=np.float64
        )
        theta = softplus_coordinates(eigen)
        recovered = theta_to_eigenvalues(theta)
        np.testing.assert_allclose(recovered, eigen, atol=2.0e-6)
        self.assertTrue(np.all(np.diff(recovered, axis=1) >= 0.0))

    def test_stratification_is_reproducible_and_covers_every_shell(self):
        shell = np.repeat(np.arange(4, dtype=np.int8), [1000, 500, 100, 10])
        first, weights, audit = stratified_indices(shell, 400, 7)
        second, weights2, _ = stratified_indices(shell, 400, 7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(weights, weights2)
        self.assertEqual(len(first), 400)
        self.assertEqual(set(shell[first].tolist()), {0, 1, 2, 3})
        self.assertEqual(sum(row["selected"] for row in audit.values()), 400)

    def test_weighted_metrics(self):
        truth = np.asarray([0.0, 1.0, 2.0])
        weight = np.asarray([1.0, 2.0, 1.0])
        self.assertAlmostEqual(weighted_r2(truth, truth, weight), 1.0)
        samples = np.stack(
            [
                np.column_stack((truth - 1.0, truth - 1.0, truth - 1.0)),
                np.column_stack((truth + 1.0, truth + 1.0, truth + 1.0)),
            ],
            axis=1,
        )
        coverage = weighted_coverage(samples, np.column_stack((truth,) * 3), weight, 0.68)
        np.testing.assert_allclose(coverage, 1.0)

    def test_paired_log_prob_uses_non_iid_vector_field_potential(self):
        class Flow:
            def __init__(self, context):
                self.context = context

            def log_prob(self, theta):
                return theta[..., 0] + self.context[:, 0]

        class Prior:
            def log_prob(self, theta):
                return torch.zeros(len(theta), dtype=theta.dtype)

        class Potential:
            def __init__(self):
                self.x_is_iid = None
                self.prior = Prior()

            def set_x(self, context, x_is_iid):
                self.x_is_iid = x_is_iid
                self.flow = Flow(context)

        posterior = type("Posterior", (), {"potential_fn": Potential()})()
        theta = torch.asarray([[1.0, 0.0], [2.0, 0.0]])
        context = torch.asarray([[3.0], [4.0]])
        value = paired_posterior_log_prob(posterior, theta, context)
        torch.testing.assert_close(value, torch.asarray([4.0, 6.0]))
        self.assertFalse(posterior.potential_fn.x_is_iid)

    def test_random_support_distance_samples_disconnected_caps(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"caps": {}}
            for cap_name, value in (("SGC", 3.0), ("NGC", 7.0)):
                path = root / f"{cap_name}.h5"
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("counts", data=np.zeros((2, 2, 2)))
                    handle.create_dataset(
                        "distance_to_support_boundary",
                        data=np.full((2, 2, 2), value, dtype=np.float32),
                    )
                    handle.create_dataset(
                        "support_random",
                        data=np.ones((2, 2, 2), dtype=np.uint8),
                    )
                manifest["caps"][cap_name] = {
                    "field_path": str(path),
                    "origin_mpc": [0.0, 0.0, 0.0],
                    "cell_mpc": 1.0,
                }
            points = np.asarray(
                [[0.2, 0.2, 0.2, 0.0], [1.2, 1.2, 1.2, 1.0]], dtype=np.float32
            )
            distance, support = sample_random_support_distance(
                manifest, points, np.asarray([0, 1], dtype=np.int64)
            )
            np.testing.assert_allclose(distance, [3.0, 7.0])
            np.testing.assert_array_equal(support, [True, True])

    def test_random_support_distance_accepts_canonical_p3br_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "schema_version": "p3br-response-overlay-manifest-v1",
                "components": {},
            }
            for cap_name, value in (("SGC", 2.0), ("NGC", 5.0)):
                path = root / f"{cap_name}.h5"
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("counts", data=np.zeros((2, 2, 2)))
                    handle.create_dataset(
                        "distance_to_support_boundary",
                        data=np.full((2, 2, 2), value, dtype=np.float32),
                    )
                    handle.create_dataset(
                        "support_random",
                        data=np.ones((2, 2, 2), dtype=np.uint8),
                    )
                manifest["components"][cap_name] = {
                    "file": str(path),
                    "grid": {"origin_mpc": [0.0, 0.0, 0.0], "cell_mpc": 1.0},
                }
            points = np.asarray(
                [[0.2, 0.2, 0.2, 0.0], [1.2, 1.2, 1.2, 1.0]], dtype=np.float32
            )
            distance, support = sample_random_support_distance(
                manifest, points, np.asarray([0, 1], dtype=np.int64)
            )
            np.testing.assert_allclose(distance, [2.0, 5.0])
            np.testing.assert_array_equal(support, [True, True])


if __name__ == "__main__":
    unittest.main()
