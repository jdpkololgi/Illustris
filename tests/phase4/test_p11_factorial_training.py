import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from workflows.abacus_tweb.p11_factorial_training import (
    CHANNELS,
    P11_SEALED_PHASE,
    P11_TRAINING_PHASES,
    P11_VALIDATION_PHASE,
    final_view_adapter,
)
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p6_field_patch_utils import derive_selection_channels


class P11DenseResponseContractTest(unittest.TestCase):
    def test_roles_keep_validation_and_blind_out_of_fit(self):
        self.assertEqual(P11_TRAINING_PHASES, ("ph002", "ph003", "ph004", "ph005"))
        self.assertEqual(P11_VALIDATION_PHASE, "ph006")
        self.assertEqual(P11_SEALED_PHASE, "ph001")
        self.assertNotIn(P11_VALIDATION_PHASE, P11_TRAINING_PHASES)
        self.assertNotIn(P11_SEALED_PHASE, P11_TRAINING_PHASES)

    def test_three_channel_interface_is_capacity_matched(self):
        self.assertEqual(
            CHANNELS, ("counts", "exposure_apodized", "log_count_ratio")
        )

    def test_r1_final_view_never_falls_back_to_p3a_selection_derivation(self):
        loader = SimpleNamespace(
            manifest={"schema_version": "p3br-r1-training-loader-ready-v1"}
        )
        sentinel = object()
        with mock.patch.object(
            P10RandomResponseLoader, "field_adapter", return_value=sentinel
        ) as random_adapter, mock.patch.object(
            P10PhaseBalancedLoader, "field_adapter"
        ) as p3a_adapter:
            self.assertIs(final_view_adapter(loader, "ph002"), sentinel)
        random_adapter.assert_called_once_with(loader, "ph002")
        p3a_adapter.assert_not_called()

    def test_response_weight_changes_mu_not_observed_counts(self):
        counts = np.ones((2, 2, 2), dtype=np.float32)
        support = np.ones_like(counts)
        redshift = np.full_like(counts, 0.3, dtype=np.float64)
        curve = dict(grid_z=np.array([0.1, 0.6]), ntilde=np.array([1e-3, 1e-3]))
        low = derive_selection_channels(
            counts,
            0.5 * support,
            redshift,
            cell_mpc=5.0,
            **curve,
        )
        high = derive_selection_channels(
            counts,
            support,
            redshift,
            cell_mpc=5.0,
            **curve,
        )
        np.testing.assert_allclose(high["expected_counts"], 2.0 * low["expected_counts"])
        self.assertTrue(np.all(low["log_count_ratio"] > high["log_count_ratio"]))


if __name__ == "__main__":
    unittest.main()
