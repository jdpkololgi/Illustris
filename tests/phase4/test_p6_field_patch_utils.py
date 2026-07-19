import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np


PATH = Path(__file__).parents[2] / "workflows/abacus_tweb/p6_field_patch_utils.py"
SPEC = importlib.util.spec_from_file_location("p6_utils", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class P6FieldPatchUtilsTests(unittest.TestCase):
    def test_fractional_cell_index_uses_cell_centres(self):
        xyz = np.asarray([[0.5, 1.5, 2.5], [1.5, 2.5, 3.5]])
        got = MOD.fractional_cell_index(xyz, np.zeros(3), 1.0)
        np.testing.assert_allclose(got, [[0, 1, 2], [1, 2, 3]])

    def test_trilinear_sampling_axis_order_and_border(self):
        x, y, z = np.indices((3, 4, 5))
        field = (100 * x + 10 * y + z).astype(np.float32)
        got = MOD.trilinear_sample(
            field, np.asarray([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5], [-1, 9, 9]])
        )
        np.testing.assert_allclose(got, [123.0, 67.5, 34.0])

    def test_registered_channel_transforms(self):
        np.testing.assert_allclose(
            MOD.channel_transform("counts", np.asarray([0.0, 1.0])),
            np.log1p([0.0, 1.0]),
        )
        self.assertTrue(np.isfinite(
            MOD.channel_transform("ntilde_mpc3", np.asarray([0.0]))
        ).all())
        values = np.asarray([1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(
            MOD.channel_transform("los_x", values), values
        )

    def test_adapter_extracts_canonical_view_and_local_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            field_path = root / "sgc.h5"
            shape = (6, 7, 8)
            base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            with h5py.File(field_path, "w") as handle:
                handle.create_dataset("counts", data=base)
                handle.create_dataset("exposure_binary", data=np.ones(shape, dtype=np.uint8))
            manifest = {
                "channel_order": ["counts", "exposure_binary"],
                "caps": {
                    "SGC": {"field_path": str(field_path)},
                    "NGC": {"field_path": str(field_path)},
                },
            }
            (root / "adapter_manifest.json").write_text(json.dumps(manifest))
            np.save(root / "core_voxel_start.npy", np.asarray([[2, 2, 2]], dtype=np.int32))
            np.save(root / "core_voxel_stop.npy", np.asarray([[4, 5, 6]], dtype=np.int32))
            np.save(root / "core_fold.npy", np.asarray([3], dtype=np.int8))
            np.save(root / "core_cap.npy", np.asarray([0], dtype=np.int8))
            np.save(root / "core_active_offsets.npy", np.asarray([0, 2], dtype=np.int64))
            np.save(root / "core_active_parent.npy", np.asarray([10, 11], dtype=np.int64))
            frac = np.asarray([[2.5, 3.5, 4.5], [3.0, 4.0, 5.0]], dtype=np.float32)
            np.save(root / "core_active_frac_index.npy", frac)
            with MOD.CanonicalFieldPatchAdapter(root) as adapter:
                patch = adapter.extract(0, 1)
            np.testing.assert_array_equal(patch.context_start, [1, 1, 1])
            np.testing.assert_array_equal(patch.context_stop, [5, 6, 7])
            np.testing.assert_array_equal(
                patch.authoritative_frac_index_local, frac - 1
            )
            np.testing.assert_array_equal(
                patch.core_values[0], base[2:4, 2:5, 2:6]
            )
            direct = MOD.trilinear_sample(
                np.stack([base, np.ones(shape, dtype=np.float32)]), frac
            )
            np.testing.assert_allclose(MOD.sample_patch(patch), direct)

    def test_frozen_normalization_never_refits_patch(self):
        values = np.asarray([
            [[[0.0, 1.0]]],
            [[[0.2, 0.8]]],
        ], dtype=np.float32)
        patch = MOD.FieldPatch(
            core_id=0, fold=0, cap=0,
            channel_names=("counts", "exposure_binary"),
            values=values,
            context_start=np.zeros(3, dtype=int),
            context_stop=np.asarray([1, 1, 2]),
            core_start=np.zeros(3, dtype=int),
            core_stop=np.asarray([1, 1, 2]),
            core_slice=(slice(None), slice(None), slice(None)),
            authoritative_parent_id=np.empty(0, dtype=int),
            authoritative_frac_index_global=np.empty((0, 3)),
            authoritative_frac_index_local=np.empty((0, 3)),
            available_halo_low=np.zeros(3, dtype=int),
            available_halo_high=np.zeros(3, dtype=int),
        )
        spec = {"channels": {
            "counts": {"policy": "zscore", "mean": 0.0, "std": 2.0},
            "exposure_binary": {"policy": "identity"},
        }}
        got = MOD.apply_frozen_normalization(patch, spec)
        np.testing.assert_allclose(got[0], np.log1p(values[0]) / 2.0)
        np.testing.assert_array_equal(got[1], values[1])


if __name__ == "__main__":
    unittest.main()
