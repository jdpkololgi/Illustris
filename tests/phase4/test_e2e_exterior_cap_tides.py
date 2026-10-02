import unittest
import numpy as np
import torch
from workflows.sbi.e2e_exterior_cap_tides import CapTidal
from workflows.sbi.e2e_footprint_tidal_operator import cube_tensor


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA operator test needs GPU allocation')
class CapOperatorTests(unittest.TestCase):
    def test_offset_trace_and_direct_cube_sum(self):
        shape, padding = (8, 7, 6), 2
        operator = CapTidal(shape, padding)
        density = np.ones(operator.source_shape)
        points, amplitudes = [(3, 4, 2), (9, 8, 5)], [.7, -.2]
        for point, amplitude in zip(points, amplitudes):
            density[point] += amplitude
        tensor = np.stack(list(operator.components(density)), axis=-1)
        cut = tuple(slice(padding, padding+n) for n in shape)
        np.testing.assert_allclose(tensor[..., [0, 3, 5]].sum(-1), density[cut]-1, atol=1e-13, rtol=0)
        for point in ((0, 0, 0), (1, 2, 0), (5, 4, 3), (7, 6, 5)):
            expected = sum(a*cube_tensor(*(np.array(point)+padding-p)) for p, a in zip(points, amplitudes))
            np.testing.assert_allclose(tensor[point], expected, atol=1e-13, rtol=0)


if __name__ == '__main__':
    unittest.main()
