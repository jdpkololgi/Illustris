import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from workflows.sbi import e2e_wholefield_fullcap_extract as x
from workflows.sbi.e2e_wholefield_sensitivity import fine_vector


class ExtractionTests(unittest.TestCase):
    def test_phase_addresses(self):
        for phase in x.PHASES:
            self.assertIn(phase, x.address(phase, 'NGC', 29, 28))
        for phase in ('ph001', 'ph011', 'ph018', 'ph019', 'Loa'):
            with self.assertRaises(PermissionError):
                x.address(phase, 'NGC', 17, 0)
        for first in (-4, 1, 32):
            with self.assertRaises(ValueError):
                x.address('ph012', 'SGC', 17, first)

    def test_reconstruction_exact_reference(self):
        # Extract only the pure reference function: importing the old command
        # module would install unrelated campaign paths and phase authority.
        path = Path(x.__file__).with_name('e2e_density_only_reconstruction.py')
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reconstruct')
        env = dict(np=np, s=SimpleNamespace(b=SimpleNamespace(op=SimpleNamespace(mean_pool=lambda a, f: x.pool(a)))))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
        rng = np.random.default_rng(190)
        for rho in (np.ones((4, 6, 8)), np.exp(rng.normal(size=(6, 4, 8))*3)):
            got = x.reconstruct(rho)
            np.testing.assert_array_equal(got, env['reconstruct'](rho, 'linear'))
            np.testing.assert_allclose(x.pool(got), rho, rtol=1e-14, atol=1e-14)
            self.assertGreater(got.min(), 0)

    def test_invalid_reconstruction(self):
        for rho in (np.zeros((2, 2, 2)), np.full((2, 2, 2), np.nan), np.ones((2, 2))):
            with self.assertRaises(ValueError):
                x.reconstruct(rho)

    def test_summary_exact_reference(self):
        rng = np.random.default_rng(71)
        for prefix in ((), (4,), (8, 4)):
            rho = rng.uniform(.1, 3, size=(*prefix, 2, 4096))
            eigen = np.sort(rng.normal(size=(*rho.shape, 3)), axis=-1)
            np.testing.assert_array_equal(x.summary(rho, eigen), fine_vector(rho, eigen)[0])

    def test_cuts_alignment(self):
        row = dict(center=[16, 16, 16], grid=dict(origin_mpc_h=[0, 0, 0]))
        core = x.cuts(row, np.zeros(3))
        self.assertEqual([(s.start, s.stop) for s in core], [(0, 32), (0, 16), (0, 16)])
        patch = x.cuts(row, np.full(3, -32*6.766))
        self.assertEqual([s.start for s in patch], [32, 32, 32])
        with self.assertRaises(ValueError):
            x.cuts(row, np.ones(3))


if __name__ == '__main__':
    unittest.main()
