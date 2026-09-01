import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from workflows.abacus_tweb.p11_jepa_canary import (
    ARMS,
    DEFAULT_CONTRACT,
    DEFAULT_FINAL_CONTRACT,
    PairedDegradeJEPA,
    aggregate_file_contract,
    alignment_loss,
    arm_auxiliary_losses,
    auxiliary_cluster_mask,
    deterministic_block_mask,
    deterministic_cluster_mask,
    exact_supported_core_mask,
    load_contract,
    masked_student_values,
    representation_statistics,
    resize_mask,
    supported_core_mask,
    technical_canary_gates,
    unet_features,
    validate_auxiliary_sampling_contract,
    validate_dense_adapter_marker,
    validate_pair,
    validate_r1_ready_manifest,
)
from workflows.abacus_tweb.p11_jepa_mask_feasibility_audit import (
    production_mask_record,
)


REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "configs/p11_paired_degrade_jepa_v1.json"
CONTRACT_V2 = REPO / "configs/p11_paired_degrade_jepa_v2.json"


def patch(exposure, *, parent=None):
    shape = exposure.shape
    values = np.stack((np.ones(shape), exposure, np.zeros(shape))).astype(np.float32)
    return SimpleNamespace(
        core_id=7,
        fold=2,
        cap=1,
        channel_names=("counts", "exposure_apodized", "log_count_ratio"),
        values=values,
        context_start=np.array([0, 0, 0]),
        context_stop=np.array(shape),
        core_start=np.array([2, 2, 2]),
        core_stop=np.array([6, 6, 6]),
        core_slice=(slice(2, 6), slice(2, 6), slice(2, 6)),
        authoritative_parent_id=np.array([11, 12] if parent is None else parent),
        authoritative_frac_index_local=np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]),
    )


def support_patch(exact, *, parent=None):
    result = patch(np.ones_like(exact, dtype=np.float32), parent=parent)
    result.channel_names = ("support_random",)
    result.values = np.asarray(exact, dtype=np.float32)[None]
    return result


class P11JEPAContractTest(unittest.TestCase):
    def test_contract_seals_blind_phase_and_freezes_matched_arms(self):
        contract = load_contract(CONTRACT)
        self.assertEqual(tuple(contract["phase_split"]["training"]), ("ph002", "ph003", "ph004", "ph005"))
        self.assertEqual(contract["phase_split"]["validation_and_selection"], "ph006")
        self.assertEqual(contract["phase_split"]["sealed_blind_test"], "ph001")
        self.assertEqual(set(contract["matched_arms"]), set(ARMS))
        self.assertFalse(contract["scientific_guards"]["jepa_is_posterior"])
        self.assertFalse(contract["scientific_guards"]["exact_latent_equality_required"])
        self.assertEqual(contract["diagnostics"]["latent_export_every_updates"], 250)
        self.assertEqual(
            contract["diagnostics"]["registered_latent_trajectory_steps"],
            [0, 250, 500],
        )
        self.assertEqual(
            DEFAULT_FINAL_CONTRACT,
            Path(
                "/global/homes/d/dkololgi/p11_contracts/"
                "training_contract_r1_random_repair_v2_20260901"
            ),
        )
        self.assertEqual(contract["masking"]["blocks"], 4)
        self.assertEqual(contract["masking"]["fallback"], "none; fail closed if four complete supported cuboids do not exist")
        self.assertEqual(
            contract["masking"]["model_tensor_order"],
            ["zscored_log1p_counts", "density_proxy", "exposure_apodized"],
        )

        contract_v2 = load_contract(CONTRACT_V2)
        self.assertEqual(DEFAULT_CONTRACT, CONTRACT_V2)
        self.assertEqual(contract_v2["masking"]["strategy"], "exact_support_compact_clusters_v2")
        self.assertEqual(contract_v2["masking"]["support_channel"], "support_random")
        self.assertEqual(contract_v2["masking"]["registered_target_fraction"], 0.25)
        self.assertEqual(contract_v2["masking"]["clusters"], 4)
        sampling = validate_auxiliary_sampling_contract(contract_v2["masking"])
        self.assertEqual(sampling["registered_canary_updates"], 500)
        self.assertEqual(sampling["maximum_canary_auxiliary_invalid_updates"], 1)
        self.assertEqual(
            sampling["maximum_population_auxiliary_invalid_fraction"], 0.001
        )
        policy = contract_v2["masking"]["auxiliary_invalid_policy"]
        self.assertIn("unmasked V_final", policy)
        self.assertIn("response_only", policy)
        self.assertIn("globally zeroed cosmological-signal", policy)

    def test_pair_validation_and_common_supported_core(self):
        exposure = np.ones((8, 8, 8), dtype=np.float32)
        exposure[3, 3, 3] = 0.0
        left = patch(exposure)
        right = patch(exposure.copy())
        validate_pair(left, right)
        mask = supported_core_mask(left, right)
        self.assertEqual(int(mask.sum()), 4 ** 3 - 1)
        self.assertFalse(mask[3, 3, 3])
        wrong = patch(exposure, parent=np.array([11, 13]))
        with self.assertRaises(RuntimeError):
            validate_pair(left, wrong)
        mismatched_exposure = exposure.copy()
        mismatched_exposure[4, 4, 4] = 0.0
        with self.assertRaises(RuntimeError):
            supported_core_mask(left, patch(mismatched_exposure))

    def test_block_mask_is_deterministic_supported_and_resume_independent(self):
        eligible = np.zeros((18, 18, 18), dtype=bool)
        eligible[2:16, 2:16, 2:16] = True
        eligible[8:10, 8:10, 8:10] = False
        kwargs = dict(
            seed=42,
            epoch=3,
            phase_index=1,
            core_id=91,
            block_voxels=5,
            blocks=4,
        )
        first = deterministic_block_mask(eligible, **kwargs)
        second = deterministic_block_mask(eligible, **kwargs)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.any(first & ~eligible))
        self.assertEqual(int(first.sum()), 4 * 5 ** 3)
        too_small = np.zeros((8, 8, 8), dtype=bool)
        too_small[:5, :5, :5] = True
        with self.assertRaises(RuntimeError):
            deterministic_block_mask(too_small, **kwargs)

    def test_exact_m_parity_ignores_visible_apodization(self):
        exposure = np.ones((8, 8, 8), dtype=np.float32)
        exposure[3, 3, 3] = 0.0
        exact = np.ones_like(exposure, dtype=np.uint8)
        final = patch(exposure)
        dense = patch(exposure.copy())
        hidden = support_patch(exact)
        mask = exact_supported_core_mask(final, dense, hidden)
        self.assertEqual(int(mask.sum()), 4 ** 3)
        self.assertTrue(mask[3, 3, 3])

        hidden.values[0, 3, 3, 3] = 0.5
        with self.assertRaises(RuntimeError):
            exact_supported_core_mask(final, dense, hidden)
        wrong_geometry = support_patch(exact, parent=np.array([11, 13]))
        with self.assertRaises(RuntimeError):
            exact_supported_core_mask(final, dense, wrong_geometry)

    def test_compact_cluster_mask_is_exact_deterministic_and_m0_safe(self):
        eligible = np.zeros((13, 13, 13), dtype=bool)
        eligible[1:10, 2, 6] = True
        eligible[2, 1:10, 8] = True
        self.assertEqual(int(eligible.sum()), 18)
        spec = load_contract(CONTRACT_V2)["masking"]
        kwargs = dict(
            seed=42,
            epoch=3,
            phase_index=1,
            core_id=91,
            mask_spec=spec,
        )
        first = deterministic_cluster_mask(eligible, **kwargs)
        second = deterministic_cluster_mask(eligible, **kwargs)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.any(first & ~eligible))
        self.assertEqual(int(first.sum()), int(np.rint(0.25 * eligible.sum())))

        too_sparse = np.zeros((8, 8, 8), dtype=bool)
        too_sparse.flat[:15] = True
        with self.assertRaises(RuntimeError):
            deterministic_cluster_mask(too_sparse, **kwargs)
        disabled, valid, reason, counts = auxiliary_cluster_mask(
            too_sparse, **kwargs
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "registered_exact_support_minimum")
        self.assertEqual(int(disabled.sum()), 0)
        self.assertEqual(counts["full_resolution_targets"], 0)

    def test_mask_downsampling_uses_any_pooling_and_preserves_thin_targets(self):
        mask = torch.zeros(1, 17, 17, 17, dtype=torch.bool)
        mask[0, 1, 1, 1] = True
        mask[0, 15, 15, 15] = True
        mask[0, 8, 2, 14] = True
        expected = mask[:, None].float()
        for _ in range(3):
            expected = torch.nn.functional.max_pool3d(
                expected, kernel_size=2, stride=2, ceil_mode=True
            )
        resized = resize_mask(mask, tuple(expected.shape[2:]))
        torch.testing.assert_close(resized, expected[:, 0].bool())
        self.assertGreaterEqual(int(resized.sum()), 2)

    def test_feasibility_audit_calls_the_production_auxiliary_policy(self):
        spec = load_contract(CONTRACT_V2)["masking"]
        common = dict(
            core_start=np.array([4, 4, 4]),
            context_start=np.array([0, 0, 0]),
            context_stop=np.array([24, 24, 24]),
            seed=42,
            epoch=1,
            phase_index=0,
            core_id=7,
            mask_spec=spec,
        )
        valid = production_mask_record(np.ones((16, 16, 16), dtype=bool), **common)
        self.assertTrue(valid["auxiliary_valid"])
        self.assertIsNone(valid["auxiliary_invalid_reason"])
        self.assertEqual(valid["eligible_voxels"], 16 ** 3)
        self.assertEqual(valid["production_target_voxels"], 16 ** 3 // 4)
        self.assertEqual(
            valid["production_target_voxels"],
            valid["production_full_resolution_targets"],
        )
        self.assertGreaterEqual(valid["production_bottleneck_targets"], 2)

        invalid = production_mask_record(np.ones((2, 2, 2), dtype=bool), **common)
        self.assertFalse(invalid["auxiliary_valid"])
        self.assertEqual(
            invalid["auxiliary_invalid_reason"],
            "registered_exact_support_minimum",
        )
        self.assertEqual(invalid["production_target_voxels"], 0)

    def test_frozen_data_inventory_is_fail_closed_and_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.json"
            right = root / "right.npy"
            left.write_text('{"status":"ready"}')
            np.save(right, np.arange(5, dtype=np.int64), allow_pickle=False)
            first = aggregate_file_contract({"left": left, "right": right})
            second = aggregate_file_contract({"right": right, "left": left})
            self.assertEqual(first, second)
            left.write_text('{"status":"changed"}')
            changed = aggregate_file_contract({"left": left, "right": right})
            self.assertNotEqual(first["aggregate_sha256"], changed["aggregate_sha256"])
            with self.assertRaises(FileNotFoundError):
                aggregate_file_contract({"missing": root / "absent.json"})

        dense_marker = {
            "schema_version": "p11-dense-response-adapter-v1",
            "view": "V_dense",
            "tracer": "BGS_BRIGHT",
            "training_phases": ["ph002", "ph003", "ph004", "ph005"],
            "validation_phase": "ph006",
            "sealed_phase": "ph001",
            "sealed_phase_opened": False,
            "truth_or_targets_read": False,
            "pass": True,
            "gates": {
                "finite_nonzero_normalization": True,
                "finite_positive_curves": True,
                "ph001_not_opened": True,
                "ph006_application_only": True,
                "shell_closure_below_10pct": True,
                "training_phases_only": True,
            },
            "channel_order": ["counts", "exposure_apodized", "log_count_ratio"],
            "model_mapping": [
                "zscored_log1p_counts",
                "clipped_expm1_log_count_ratio",
                "common_random_support_exposure",
            ],
            "response_contract": {
                "support": "P3b-R exposure_apodized_random",
                "angular_response": "P3b-R angular_response enters mu, not the third channel",
                "mu": "ntilde_dense(z) * voxel_volume * angular_response * support_exposure",
            },
        }
        validate_dense_adapter_marker(
            dense_marker,
            training=("ph002", "ph003", "ph004", "ph005"),
            validation="ph006",
            sealed="ph001",
        )
        for field, bad_value in (
            ("schema_version", "wrong"),
            ("view", "V_final"),
            ("tracer", "BGS_FAINT"),
            ("sealed_phase_opened", True),
        ):
            bad = json.loads(json.dumps(dense_marker))
            bad[field] = bad_value
            with self.assertRaises(RuntimeError):
                validate_dense_adapter_marker(
                    bad,
                    training=("ph002", "ph003", "ph004", "ph005"),
                    validation="ph006",
                    sealed="ph001",
                )
        bad = json.loads(json.dumps(dense_marker))
        bad["response_contract"]["support"] = "occupancy exposure"
        with self.assertRaises(RuntimeError):
            validate_dense_adapter_marker(
                bad,
                training=("ph002", "ph003", "ph004", "ph005"),
                validation="ph006",
                sealed="ph001",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phases = ("ph002", "ph003", "ph004", "ph005", "ph006")
            adapters = {}
            inventory_phases = {}
            for phase in phases:
                path = root / "adapters" / phase / "field" / "adapter_manifest.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"phase": phase, "pass": True}))
                digest = aggregate_file_contract({"adapter": path})["files"]["adapter"]["sha256"]
                adapters[phase] = {"path": str(path), "sha256": digest, "pass": True}
                inventory_phases[phase] = {
                    "field_manifest": str(path),
                    "field_manifest_sha256": digest,
                }
            inventory_path = root / "adapter_inventory.json"
            inventory_path.write_text(json.dumps({
                "schema_version": "p3br-r1-adapter-inventory-v1",
                "phases": inventory_phases,
                "ph001_product_built": False,
                "pass": True,
            }))
            transform_path = root / "transforms" / "field" / "field_transform.json"
            transform_path.parent.mkdir(parents=True, exist_ok=True)
            transform_path.write_text(json.dumps({"pass": True}))
            inventory_hash = aggregate_file_contract({"inventory": inventory_path})[
                "files"
            ]["inventory"]["sha256"]
            transform_hash = aggregate_file_contract({"transform": transform_path})[
                "files"
            ]["transform"]["sha256"]
            ready = {
                "schema_version": "p3br-r1-training-loader-ready-v1",
                "view": "R1 BRIGHT counts plus random-derived response, capacity matched to R0",
                "field_channels": ["counts", "exposure_apodized", "log_count_ratio"],
                "roles": {
                    "training": ["ph000", "ph002", "ph003", "ph004", "ph005"],
                    "validation_and_selection": "ph006",
                    "sealed_blind_test": "ph001",
                },
                "adapters": adapters,
                "adapter_inventory": str(inventory_path),
                "adapter_inventory_sha256": inventory_hash,
                "field_transform": str(transform_path),
                "field_transform_sha256": transform_hash,
                "ph001_product_built": False,
                "ph001_opened": False,
                "pass": True,
            }
            kwargs = {
                "root": root,
                "training": ("ph002", "ph003", "ph004", "ph005"),
                "validation": "ph006",
                "sealed": "ph001",
            }
            validate_r1_ready_manifest(ready, **kwargs)
            for field, bad_value in (
                ("adapter_inventory", "/stale/base/adapter_inventory.json"),
                ("adapter_inventory_sha256", "0" * 64),
                ("field_transform", "/stale/base/field_transform.json"),
            ):
                bad = json.loads(json.dumps(ready))
                bad[field] = bad_value
                with self.assertRaises(RuntimeError):
                    validate_r1_ready_manifest(bad, **kwargs)
            bad = json.loads(json.dumps(ready))
            bad["adapters"]["ph002"]["sha256"] = "0" * 64
            with self.assertRaises(RuntimeError):
                validate_r1_ready_manifest(bad, **kwargs)

    def test_response_channel_is_never_hidden(self):
        values = torch.randn(1, 3, 8, 8, 8)
        response = values[:, 2].clone()
        mask = torch.zeros(1, 8, 8, 8, dtype=torch.bool)
        mask[:, 2:6, 2:6, 2:6] = True
        masked = masked_student_values(values, mask, response_only=False)
        torch.testing.assert_close(masked[:, 2], response)
        self.assertTrue(torch.all(masked[:, :2].masked_select(mask[:, None]) == 0))
        response_only = masked_student_values(values, mask, response_only=True)
        self.assertTrue(torch.all(response_only[:, :2] == 0))
        torch.testing.assert_close(response_only[:, 2], response)
        no_auxiliary_mask = torch.zeros_like(mask)
        unmasked = masked_student_values(values, no_auxiliary_mask, response_only=False)
        torch.testing.assert_close(unmasked, values)
        response_only_no_auxiliary = masked_student_values(
            values, no_auxiliary_mask, response_only=True
        )
        self.assertTrue(torch.all(response_only_no_auxiliary[:, :2] == 0))
        torch.testing.assert_close(response_only_no_auxiliary[:, 2], response)

    def test_teacher_is_stop_gradient_and_student_head_is_unique(self):
        torch.manual_seed(4)
        model = PairedDegradeJEPA(base=2, latent_channels=4, head_width=8)
        self.assertFalse(hasattr(model.teacher, "head"))
        student_values = torch.randn(1, 3, 16, 16, 16)
        dense_values = torch.randn(1, 3, 16, 16, 16)
        features = model.encode_student(student_values)
        mask = torch.ones(1, 16, 16, 16, dtype=torch.bool)
        losses = arm_auxiliary_losses(
            arm="jepa",
            model=model,
            student_features=features,
            dense_values=dense_values,
            unmasked_final_values=student_values,
            target_mask=mask,
            layer_weights={"latent": 0.75, "bottleneck": 0.25},
        )
        total = losses["alignment"] + 0.01 * losses["spread"] + 0.001 * losses["covariance"]
        total.backward()
        self.assertTrue(any(p.grad is not None for p in model.student.unet.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.predictors.parameters()))
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        before = [p.detach().clone() for p in model.teacher.parameters()]
        with torch.no_grad():
            next(model.student.unet.parameters()).add_(1.0)
        model.ema_update(0.5)
        self.assertFalse(torch.equal(before[0], next(model.teacher.parameters())))

        disabled = arm_auxiliary_losses(
            arm="jepa",
            model=model,
            student_features=features,
            dense_values=dense_values,
            unmasked_final_values=student_values,
            target_mask=torch.zeros_like(mask),
            layer_weights={"latent": 0.75, "bottleneck": 0.25},
            auxiliary_valid=False,
        )
        self.assertTrue(all(float(value.detach()) == 0.0 for value in disabled.values()))

    def test_representation_metrics_flag_collapse(self):
        rng = np.random.default_rng(8)
        spread = rng.normal(size=(128, 8))
        collapsed = np.ones((128, 8))
        spread_row = representation_statistics(spread, spread, spread)
        collapse_row = representation_statistics(collapsed, collapsed, collapsed)
        self.assertGreater(spread_row["student_effective_rank"], collapse_row["student_effective_rank"])
        self.assertEqual(collapse_row["student_collapse_fraction_std_lt_0p05"], 1.0)
        self.assertAlmostEqual(spread_row["predicted_teacher_cosine"], 1.0, places=7)
        control_row = representation_statistics(spread, spread, None)
        self.assertIsNone(control_row["predicted_teacher_cosine"])

    def test_technical_marker_requires_every_registered_gate(self):
        arguments = dict(
            finite_pre_parameters=True,
            finite_post_parameters=True,
            finite_loss=True,
            gradient_norm=0.8,
            mask_fraction=0.2,
            registered_mask_fraction_range=(0.02, 0.5),
            auxiliary_invalid_updates=0,
            maximum_auxiliary_invalid_updates=1,
            canary_updates=500,
            registered_canary_updates=500,
            checkpoint_reload_valid=True,
            latent_snapshot_valid=True,
        )
        passed = technical_canary_gates(**arguments)
        self.assertTrue(passed["pass"])
        for name in (
            "finite_pre_parameters",
            "finite_post_parameters",
            "finite_loss",
            "checkpoint_reload_valid",
            "latent_snapshot_valid",
        ):
            failed = technical_canary_gates(**{**arguments, name: False})
            self.assertFalse(failed["pass"], name)
        self.assertFalse(
            technical_canary_gates(**{**arguments, "gradient_norm": np.nan})["pass"]
        )
        self.assertFalse(
            technical_canary_gates(**{**arguments, "mask_fraction": 0.0})["pass"]
        )
        one_invalid = technical_canary_gates(
            **{**arguments, "auxiliary_invalid_updates": 1}
        )
        self.assertTrue(one_invalid["pass"])
        self.assertFalse(
            technical_canary_gates(
                **{**arguments, "auxiliary_invalid_updates": 2}
            )["pass"]
        )
        self.assertFalse(
            technical_canary_gates(**{**arguments, "canary_updates": 499})["pass"]
        )


if __name__ == "__main__":
    unittest.main()
