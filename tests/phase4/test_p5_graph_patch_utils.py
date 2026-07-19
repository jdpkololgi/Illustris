import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


PATH = Path(__file__).parents[2] / "workflows/abacus_tweb/p5_graph_patch_utils.py"
SPEC = importlib.util.spec_from_file_location("p5_utils", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def csr(pairs, n_nodes):
    incident = [[] for _ in range(n_nodes)]
    for edge_id, (left, right) in enumerate(pairs):
        incident[int(left)].append(edge_id)
        incident[int(right)].append(edge_id)
    offsets = [0]
    flat = []
    for values in incident:
        flat.extend(values)
        offsets.append(len(flat))
    return np.asarray(offsets, dtype=np.int64), np.asarray(flat, dtype=np.int32)


class P5GraphPatchUtilsTests(unittest.TestCase):
    def setUp(self):
        self.pairs = np.asarray([[0, 1], [1, 2], [2, 3], [1, 4]], dtype=np.int32)
        self.edge = np.asarray([
            [1.0, 1.0, 0.0, 0.0, 2.0],
            [2.0, 0.0, 1.0, 0.0, 3.0],
            [3.0, 0.0, 0.0, 1.0, 4.0],
            [4.0, -1.0, 0.0, 0.0, 5.0],
        ], dtype=np.float32)
        self.x = np.arange(25, dtype=np.float32).reshape(5, 5)
        self.offsets, self.incident = csr(self.pairs, 5)

    def test_directed_reverse_dependency_is_not_forward_traversal(self):
        senders = np.asarray([0, 1, 3], dtype=np.int32)
        receivers = np.asarray([1, 2, 2], dtype=np.int32)
        got = MOD.reverse_k_hop_directed([2], senders, receivers, 2)
        np.testing.assert_array_equal(got, [0, 1, 2, 3])

    def test_bidirectional_reverse_features_match_contract(self):
        senders, receivers, attr = MOD.make_bidirectional(self.pairs, self.edge)
        np.testing.assert_array_equal(senders[:4], self.pairs[:, 0])
        np.testing.assert_array_equal(receivers[:4], self.pairs[:, 1])
        np.testing.assert_array_equal(attr[4:, 1:4], -self.edge[:, 1:4])
        np.testing.assert_allclose(attr[4:, 4], 1.0 / self.edge[:, 4])

    def test_exact_two_hop_context_and_canonical_features(self):
        patch = MOD.assemble_patch(
            core_id=7, fold=2, core_parent_ids=np.asarray([2]),
            loss_parent_ids=np.asarray([2]), num_passes=2, dependency_hops=2,
            node_features=self.x, union_pairs=self.pairs,
            union_edge_features=self.edge, offsets=self.offsets,
            incident_edge_id=self.incident,
        )
        np.testing.assert_array_equal(patch.parent_node_id, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(patch.node_features, self.x)
        self.assertEqual(patch.n_edge, 8)
        self.assertEqual(int(patch.authoritative_core_mask.sum()), 1)
        self.assertEqual(int(patch.loss_mask.sum()), 1)

    def test_primary_loss_and_strict_support_are_independent_masks(self):
        patch = MOD.assemble_patch(
            core_id=7, fold=2, core_parent_ids=np.asarray([1, 2]),
            loss_parent_ids=np.asarray([1, 2]),
            strict_parent_ids=np.asarray([2]),
            loss_policy="authoritative",
            num_passes=1, dependency_hops=1,
            node_features=self.x, union_pairs=self.pairs,
            union_edge_features=self.edge, offsets=self.offsets,
            incident_edge_id=self.incident,
        )
        self.assertEqual(int(patch.authoritative_core_mask.sum()), 2)
        self.assertEqual(int(patch.loss_mask.sum()), 2)
        self.assertEqual(int(patch.strict_support_mask.sum()), 1)
        self.assertEqual(patch.loss_policy, "authoritative")
        padded = MOD.pad_patch(patch)
        self.assertEqual(int(padded["strict_support_mask"].sum()), 1)
        self.assertEqual(int(padded["loss_mask"].sum()), 2)

    def test_padding_masks_and_dummy_edges_do_not_touch_real_nodes(self):
        patch = MOD.assemble_patch(
            core_id=1, fold=0, core_parent_ids=np.asarray([0]),
            loss_parent_ids=np.asarray([0]), num_passes=1, dependency_hops=1,
            node_features=self.x, union_pairs=self.pairs,
            union_edge_features=self.edge, offsets=self.offsets,
            incident_edge_id=self.incident,
        )
        padded = MOD.pad_patch(patch)
        self.assertEqual(int(padded["node_mask"].sum()), patch.n_node)
        self.assertEqual(int(padded["edge_mask"].sum()), patch.n_edge)
        self.assertTrue(np.all(padded["senders"][patch.n_edge:] == patch.n_node))
        self.assertTrue(np.all(padded["receivers"][patch.n_edge:] == patch.n_node))

    def test_bucket_refuses_truncation(self):
        patch = MOD.assemble_patch(
            core_id=1, fold=0, core_parent_ids=np.asarray([0]),
            loss_parent_ids=np.asarray([0]), num_passes=1, dependency_hops=1,
            node_features=self.x, union_pairs=self.pairs,
            union_edge_features=self.edge, offsets=self.offsets,
            incident_edge_id=self.incident,
        )
        with self.assertRaises(ValueError):
            MOD.pad_patch(patch, bucket_nodes=patch.n_node, bucket_edges=patch.n_edge)

    def test_loss_must_be_subset_of_core(self):
        with self.assertRaises(ValueError):
            MOD.assemble_patch(
                core_id=1, fold=0, core_parent_ids=np.asarray([0]),
                loss_parent_ids=np.asarray([1]), num_passes=1, dependency_hops=1,
                node_features=self.x, union_pairs=self.pairs,
                union_edge_features=self.edge, offsets=self.offsets,
                incident_edge_id=self.incident,
            )


if __name__ == "__main__":
    unittest.main()
