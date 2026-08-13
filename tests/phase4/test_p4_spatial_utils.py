import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).parents[2] / "workflows/abacus_tweb/p4_probe_core_sizes.py"
SPEC = importlib.util.spec_from_file_location("p4_probe", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

BUILD_PATH = Path(__file__).parents[2] / "workflows/abacus_tweb/p4_build_spatial_manifest.py"
BUILD_SPEC = importlib.util.spec_from_file_location("p4_build", BUILD_PATH)
BUILD = importlib.util.module_from_spec(BUILD_SPEC)
BUILD_SPEC.loader.exec_module(BUILD)

SUPPORT_PATH = Path(__file__).parents[2] / "workflows/abacus_tweb/p4_attach_field_support.py"
SUPPORT_SPEC = importlib.util.spec_from_file_location("p4_support", SUPPORT_PATH)
SUPPORT = importlib.util.module_from_spec(SUPPORT_SPEC)
SUPPORT_SPEC.loader.exec_module(SUPPORT)


class P4SpatialUtilsTests(unittest.TestCase):
    def test_half_open_core_ownership(self):
        xyz = np.array([[0.0, 0.0, 0.0], [9.999, 1.0, 1.0], [10.0, 1.0, 1.0]])
        got = MOD.core_indices(xyz, np.zeros(3), 10.0)
        np.testing.assert_array_equal(got[:, 0], [0, 0, 1])

    def test_unique_rows_and_counts(self):
        idx = np.array([[1, 2, 3], [1, 2, 3], [0, 0, 0]], dtype=np.int32)
        unique, inverse, counts = MOD.rows_to_counts(idx)
        self.assertEqual(int(counts.sum()), 3)
        np.testing.assert_array_equal(unique[inverse], idx)

    def test_context_upper_count(self):
        core = np.array([[0, 0, 0]], dtype=np.int32)
        lookup = {(0, 0, 0): 2, (1, 0, 0): 3, (2, 0, 0): 100}
        got = MOD.conservative_context_counts(core, lookup, 10.0, 5.0)
        self.assertEqual(int(got[0]), 5)

    def test_host_component_union(self):
        dsu = BUILD.DisjointSet(4)
        dsu.union(0, 2)
        dsu.union(2, 3)
        self.assertEqual(dsu.find(0), dsu.find(3))
        self.assertNotEqual(dsu.find(0), dsu.find(1))

    def test_fold_assignment_is_deterministic_and_complete(self):
        group_counts = np.ones((10, 2, 4), dtype=np.int64)
        group_context = np.arange(1, 11, dtype=np.int64)
        group_cores = np.ones(10, dtype=np.int64)
        left = BUILD.greedy_balanced_folds(group_counts, group_context, group_cores)
        right = BUILD.greedy_balanced_folds(group_counts, group_context, group_cores)
        np.testing.assert_array_equal(left, right)
        self.assertEqual(set(left.tolist()), set(range(5)))

    def test_registered_balance_weights_change_only_the_geometry_objective(self):
        group_counts = np.ones((10, 2, 4), dtype=np.int64)
        group_context = np.arange(1, 11, dtype=np.int64)
        group_cores = np.ones(10, dtype=np.int64)
        frozen = BUILD.greedy_balanced_folds(
            group_counts, group_context, group_cores,
            context_weight=0.05, core_weight=0.05,
        )
        fallback = BUILD.greedy_balanced_folds(
            group_counts, group_context, group_cores,
            context_weight=0.0, core_weight=0.05,
        )
        self.assertEqual(set(frozen.tolist()), set(range(5)))
        self.assertEqual(set(fallback.tolist()), set(range(5)))
        np.testing.assert_array_equal(
            fallback,
            BUILD.greedy_balanced_folds(
                group_counts, group_context, group_cores,
                context_weight=0.0, core_weight=0.05,
            ),
        )

    def test_grouped_support_quantiles(self):
        group = np.array([1, 0, 1, 0], dtype=np.int32)
        value = np.array([10.0, 1.0, 20.0, 3.0], dtype=np.float32)
        got = SUPPORT.grouped_quantiles(group, value, 3)
        self.assertEqual(float(got["min"][0]), 1.0)
        self.assertEqual(float(got["median"][1]), 15.0)
        self.assertTrue(np.isnan(got["median"][2]))

    def test_splitmix_is_stable(self):
        values = np.array([1, 2, 3], dtype=np.int64)
        np.testing.assert_array_equal(BUILD.splitmix64(values), BUILD.splitmix64(values))
        self.assertEqual(len(np.unique(BUILD.splitmix64(values))), 3)


if __name__ == "__main__":
    unittest.main()
