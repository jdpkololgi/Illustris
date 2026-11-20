import unittest
import numpy as np
from workflows.abacus_tweb.lowz_graph_patch_index import Segments, oracle
from workflows.abacus_tweb.p5_graph_patch_utils import assemble_patch


class GraphIndexTest(unittest.TestCase):
    def test_multicomponent_exact_patch(self):
        # Line with a cycle, an edge just outside the four-hop limit, and an
        # entirely disconnected component, distributed across source tables.
        parts = [np.array(x, np.int32).reshape(-1,2) for x in
                 ([[2,3],[5,6],[0,1]], [], [[3,4],[1,2],[7,8],[0,2],[4,5]])]
        pairs = np.concatenate(parts)
        attr = np.arange(len(pairs)*5, dtype=np.float32).reshape(-1,5)+1
        tab = Segments(parts)
        ids = np.array([5,0,2,5,4,7])
        np.testing.assert_array_equal(tab[ids], pairs[ids])
        with self.assertRaises(ValueError):
            tab[np.array([-1])]
        incident = [np.where((pairs == i).any(axis=1))[0] for i in range(9)]
        ptr = np.r_[0, np.cumsum([len(p) for p in incident])]
        for core, expected in (([0], [0,1,2,3,4,5]), ([8], [7,8])):
            nodes, edge_ids = oracle(core, parts, 9, 4)
            np.testing.assert_array_equal(nodes, expected)
            patch = assemble_patch(core_id=0, fold=-1, core_parent_ids=core, loss_parent_ids=core,
                strict_parent_ids=[], num_passes=2, dependency_hops=4, node_features=np.eye(9),
                union_pairs=tab, union_edge_features=Segments([attr[:3],attr[3:3],attr[3:]]),
                offsets=ptr, incident_edge_id=np.concatenate(incident))
            np.testing.assert_array_equal(nodes, patch.parent_node_id)
            np.testing.assert_array_equal(nodes[patch.loss_mask], core)
            np.testing.assert_array_equal(patch.union_edge_id, np.tile(edge_ids,2))
            np.testing.assert_array_equal(patch.edge_features[:len(edge_ids)], attr[edge_ids])
            np.testing.assert_array_equal(nodes[patch.senders[:len(edge_ids)]], pairs[edge_ids,0])
            np.testing.assert_array_equal(nodes[patch.receivers[:len(edge_ids)]], pairs[edge_ids,1])
