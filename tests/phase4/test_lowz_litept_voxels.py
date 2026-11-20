"""Preserve pretrained LitePT token ordering and the exact galaxy inverse."""
import unittest
import numpy as np
from workflows.abacus_tweb.lowz_litept import voxels


def reference(xyz, features, cell):
    keys = np.floor(xyz/cell).astype(np.int64)
    cells, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    def mean(x):
        acc = np.zeros((len(cells), x.shape[1]), np.float64)
        np.add.at(acc, inv, x)
        return (acc/counts[:,None]).astype(np.float32)
    return dict(coord=mean(xyz)-xyz.min(0).astype(np.float32),
                grid_coord=(cells-cells.min(0)).astype(np.int32),
                feat=mean(features), offset=np.array([len(cells)],np.int64)), inv


class VoxelTest(unittest.TestCase):
    def test_occupied_order_averages_inverse(self):
        rng = np.random.default_rng(193)
        for n, cell in ((1, 1.), (257, 1.), (5000, .5)):
            xyz = rng.uniform(-24., 24., size=(n,3))
            if n > 1:
                xyz[::5] = [-.001,0.,1.]
                xyz[1::5] = xyz[0]
            for dtype in (np.float32,np.float64):
                f = rng.normal(size=xyz.shape).astype(dtype)
                new, inv = voxels(xyz.astype(dtype),f,cell)
                old, expected = reference(xyz.astype(dtype),f,cell)
                np.testing.assert_array_equal(inv,expected)
                for k in old:
                    self.assertEqual(old[k].dtype,new[k].dtype)
                    np.testing.assert_array_equal(old[k],new[k])
