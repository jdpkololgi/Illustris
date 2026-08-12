import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p10_run_tweb import (
    MPI_PICKLE_COUNT_LIMIT,
    TWebBuildError,
    balanced_slice,
    max_fft_transpose_message_bytes,
    validate_density_input,
    validate_mpi_layout,
    validate_rank_outputs,
    write_rank_output,
)


class P10TWebRunnerTests(unittest.TestCase):
    def test_balanced_slices_cover_grid(self):
        slices = [balanced_slice(10, rank, 3) for rank in range(3)]
        self.assertEqual(slices, [(0, 4), (4, 7), (7, 10)])

    def test_production_fft_layout_rejects_eight_ranks(self):
        self.assertEqual(
            max_fft_transpose_message_bytes(2048, 8),
            2**31,
        )
        with self.assertRaisesRegex(TWebBuildError, "unsafe MPI layout"):
            validate_mpi_layout(2048, 8)

    def test_production_fft_layout_accepts_sixteen_ranks(self):
        report = validate_mpi_layout(2048, 16)
        self.assertEqual(report["worst_fft_transpose_message_bytes"], 2**29)
        self.assertLess(
            report["worst_fft_transpose_message_bytes"],
            MPI_PICKLE_COUNT_LIMIT,
        )

    def test_rank_output_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rank, (start, end) in enumerate(((0, 2), (2, 4))):
                shape = (end - start, 4, 4)
                write_rank_output(
                    root / f"abacus_cactus_tweb_rank{rank:04d}.npz",
                    cweb=np.zeros(shape, dtype=np.uint8),
                    eig_vals=np.zeros((3, *shape), dtype=np.float32),
                    x_start=start,
                    x_end=end,
                    ngrid=4,
                    boxsize=20.0,
                    threshold=0.2,
                    rsmooth=7.0,
                )
            report = validate_rank_outputs(
                root,
                expected_ranks=2,
                ngrid=4,
                boxsize=20.0,
                threshold=0.2,
                rsmooth=7.0,
            )
            self.assertTrue(report["verified"])
            self.assertEqual(report["x_coverage"], [0, 4])

    def test_density_rejects_canary_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            density = root / "density.npy"
            np.save(density, np.zeros((4, 4, 4), dtype=np.float32))
            manifest = root / "density.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "phase": "ph002",
                        "build": {
                            "output": str(density),
                            "processed_file_count": 136,
                            "max_files_per_directory": 1,
                            "relative_count_error": 0.0,
                        },
                    }
                )
            )
            registry = {
                "target_contract": {"grid_size": 4},
            }
            with self.assertRaises(TWebBuildError):
                validate_density_input(density, manifest, registry, "ph002")


if __name__ == "__main__":
    unittest.main()
