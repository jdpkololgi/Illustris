from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from workflows.sbi import e2e_coupled_postprocess_benchmark as bench


class PostprocessBenchmarkTests(unittest.TestCase):
    def test_fixed_bounded_full_shape_specification(self):
        spec=bench.specification()
        self.assertEqual(spec['operator_draws'],8)
        self.assertEqual(spec['score_draws'],[32,128,256])
        self.assertEqual(spec['joint_shape'],[64,48,48])
        self.assertEqual(spec['owned_shape'],[2,16,16,16])
        self.assertLessEqual(spec['max_seconds'],600)
        self.assertLessEqual(spec['artifact_reserve_bytes'],256*1024**2)

    def test_small_score_fixture_keeps_timings_not_scientific_scores(self):
        result=bench.score_probe(4,[2,2,2,2],np.random.default_rng(5))
        self.assertEqual(result['owned_voxels'],16)
        self.assertFalse(result['score_values_retained'])
        for key,value in result.items():
            if key.endswith('_seconds'): self.assertGreaterEqual(value,0)

    def test_serialization_is_exclusive_and_roundtrip_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic.h5'
            record,seconds=bench.write_field(path,{'rho_joint':np.arange(8).reshape(2,2,2)})
            self.assertEqual(record['sha256'],bench.c.sha256(path))
            self.assertGreater(seconds,0)
            with self.assertRaises(FileExistsError): bench.write_field(path,{'rho_joint':np.zeros(1)})

    def test_deadline_and_compute_guard_fail_closed(self):
        with patch.object(bench.time,'monotonic',return_value=601):
            with self.assertRaises(TimeoutError): bench.require_time(0,600)
        with patch.object(bench.c,'require_compute',side_effect=RuntimeError('not compute')), \
                patch.object(bench.c,'config') as config:
            with self.assertRaises(RuntimeError): bench.run()
            config.assert_not_called()


if __name__=='__main__': unittest.main()
