import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from workflows.abacus_tweb.p8_train_unet_patch import ChannelLayerNorm3d
from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.sbi.p12f3_d2_contract import (
    DEFAULT_CONFIG,
    digest,
    load_d2_config,
    validate_output_root,
)
from workflows.sbi.p12f3_d2_models import (
    CUDA_DETERMINISM_POLICY,
    D2ConditionalFourierVDenoiser,
    clone_model_state,
    coarsen_support_any,
    configure_d2_determinism,
    parameter_count,
    sample_fourier_d2,
    sample_fourier_d2_batched,
    update_ema_state,
)
from workflows.sbi.p12f3_d2_confirm import confirmation_consistency
from workflows.sbi.p12f3_d2_evaluate import (
    DERIVED_VARIABLES,
    combined_deployable_conditional_error,
    derived_physics_conditionals,
    posterior_mean_spectral_diagnostics,
)
from workflows.sbi.p12f3_d2_decide import (
    paired_energy as paired_ph006_energy,
    paired_energy_materiality,
    sampler_convergence,
    science_gates,
    validate_nfe_candidate_identity,
)
from workflows.sbi.p12f3_d2_roundtrip import roundtrip_metrics
from workflows.sbi.p12f3_d2_select import compare, paired_energy
from workflows.sbi.p12f3_d2_train import (
    _diagnostic_ref_split,
    _read_jsonl_rows,
    _truncate_jsonl_to_update,
    _validate_replication_license,
    build_model,
    checkpoint_payload,
    restore_checkpoint_rng,
    select_earliest_within_one_se,
    validate_stop_position,
)
from workflows.sbi.p12f3_fourier_modes import (
    build_fourier_layout,
    pack_fourier_components,
    unpack_fourier_components,
    whiten_components,
)
from workflows.sbi.p12f3_conditional_models import (
    cosine_alpha_sigma,
    fourier_v_pair,
)


def layout_and_whitening():
    layout = build_fourier_layout(
        (15, 15, 15),
        voxel_mpc_h=5.0,
        band_edges_h_mpc=(0.0, 0.0906899682117109, 0.1813799364234218),
    )
    whitening = {
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0, 1.0],
    }
    return layout, whitening


def diagnostic(energy, *, crps=0.2, loss=0.3, variance=0.1, prefix="x"):
    values = list(map(float, energy))
    return {
        "energy_score": float(np.mean(values)),
        "energy_standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "marginal_crps": float(crps),
        "denoising_loss": float(loss),
        "maximum_absolute_log_band_variance_ratio": float(variance),
        "core_keys": [f"{prefix}{index}" for index in range(len(values))],
        "per_core_energy_score": values,
        "per_core_marginal_crps": [float(crps)] * len(values),
    }


def canary_marker(row, *, weight="raw"):
    return {
        "selected_weights": weight,
        "milestone_selection": {"selected_weights": weight},
        "internal_diagnostics": {
            "selection": {"raw": row, "ema": row},
            "confirmation": None,
        },
    }


class ZeroTargetOracle(nn.Module):
    """Exact VP velocity for a posterior concentrated at x0=0."""

    def forward(self, state, time, condition, **kwargs):
        alpha, sigma = cosine_alpha_sigma(time)
        return alpha[:, None] * state / torch.clamp(sigma[:, None], min=1.0e-8)


def calibration_bundle(curve):
    coverage = {
        level: {"empirical": nominal, "absolute_error": 0.0}
        for level, nominal in (("0.68", 0.68), ("0.90", 0.90))
    }
    scalar = {"coverage": coverage}
    tarp = {
        "alpha": [0.0, 0.5, 1.0],
        "expected_coverage_probability": list(curve),
        "full_max_abs_ecp_minus_alpha": float(
            np.max(np.abs(np.asarray(curve) - np.asarray([0.0, 0.5, 1.0])))
        ),
    }
    common = {
        "tarp": {"ordered_eigenvalues": tarp, "eigengaps": tarp},
        "voxel": scalar,
        "derived_ordered_eigenvalues": {
            name: scalar for name in ("lambda1", "lambda2", "lambda3")
        },
        "derived_eigengaps": {name: scalar for name in ("gap12", "gap23")},
        "proper_scores": {
            "energy": 1.0,
            "coarse_energy": 1.0,
            "marginal_crps": 1.0,
            "variogram_p0p5": 1.0,
        },
    }
    shear = {"joint_tarp_blocked": tarp}
    visual = {"posterior_to_truth_power": [1.0, 1.0]}
    marker = {}
    return marker, common, shear, visual, {}


class P12F3D2Tests(unittest.TestCase):
    @staticmethod
    def _synthetic_field_record(draws=8):
        rng = np.random.default_rng(1209)
        shape = (8, 8, 8)
        truth = rng.normal(size=shape).astype(np.float32)
        samples = (
            truth[None]
            + rng.normal(scale=.2, size=(draws, *shape)).astype(np.float32)
        )
        coordinates = np.stack(
            np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3).astype(np.float32)
        gradient = np.linspace(.1, 1.0, np.prod(shape), dtype=np.float32).reshape(
            shape
        )
        record = {
            "delta_samples": samples,
            "delta_truth": truth,
            "support": np.ones(shape, dtype=np.uint8),
            "angular_response": gradient,
            "boundary_distance_mpc": 20.0 * gradient,
            "tracer_density": .01 + .02 * gradient,
            "frozen_g1_mean_scaled": truth * .7,
            "frozen_g1_log_std": -.5 + .2 * gradient,
            "frozen_g1_traceless_shear_amplitude": .1 + .3 * gradient,
            "core_bounds": np.asarray(((0, 0, 0), shape), dtype=np.int64),
            "galaxy_frac_index_local": coordinates,
        }
        return {"core_id": 11, "shell": 1}, record

    def test_posterior_mean_spectrum_recovers_known_linear_transfer(self):
        metadata, record = self._synthetic_field_record(draws=4)
        scale = 0.7
        record["delta_samples"] = np.repeat(
            scale * record["delta_truth"][None], 4, axis=0
        )
        report = posterior_mean_spectral_diagnostics([(metadata, record)])
        valid = np.asarray(report["mode_count"]) > 0
        for name in ("p_cross_over_p_truth", "sqrt_p_mean_over_p_truth"):
            np.testing.assert_allclose(
                np.asarray(report[name])[valid],
                scale,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        np.testing.assert_allclose(
            np.asarray(report["r_k"])[valid], 1.0, rtol=1.0e-6, atol=1.0e-7
        )
        self.assertIn("diagnostic_only", report["role"])

    def test_derived_conditionals_cover_all_physical_outputs_and_proxies(self):
        metadata, record = self._synthetic_field_record()
        report = derived_physics_conditionals(
            [(metadata, record)], device="cpu", seed=97
        )
        self.assertEqual(set(report["variables"]), set(DERIVED_VARIABLES))
        self.assertEqual(report["galaxies"], 512)
        self.assertEqual(report["authoritative_cores"], 1)
        self.assertTrue(np.isfinite(report["maximum_deployable_error"]))
        self.assertEqual(
            report["maximum_deployable_error"],
            max(report["maximum_deployable_error_by_variable"].values()),
        )
        for variable in DERIVED_VARIABLES:
            self.assertIn("frozen_g1_log_std", report["variables"][variable])
            self.assertIn("true_environment", report["variables"][variable])

    def test_combined_conditional_gate_uses_worse_voxel_or_derived_error(self):
        report = {"maximum_deployable_error": 0.12}
        combined = combined_deployable_conditional_error(0.08, report)
        config, _, _ = load_d2_config(DEFAULT_CONFIG)
        self.assertEqual(combined, 0.12)
        self.assertFalse(
            combined
            <= config["ph006_gate"][
                "deployable_proxy_conditional_coverage_error_maximum"
            ]
        )

    def test_derived_conditionals_reject_out_of_patch_galaxy_coordinates(self):
        metadata, record = self._synthetic_field_record()
        record["galaxy_frac_index_local"][0, 0] = -0.01
        with self.assertRaisesRegex(RuntimeError, "outside the frozen patch"):
            derived_physics_conditionals([(metadata, record)], device="cpu", seed=97)

    def test_frozen_contract_cannot_be_reused_under_another_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "registered"
            other = Path(directory) / "unregistered"
            marker = root / "D2_CONTRACT_FROZEN.json"
            contract = {"frozen": {"output_root": str(root.resolve())}}
            self.assertEqual(validate_output_root(contract, root, marker), root.resolve())
            with self.assertRaises(RuntimeError):
                validate_output_root(contract, other, marker)
            with self.assertRaises(RuntimeError):
                validate_output_root(contract, root, other / marker.name)

    def test_registered_config_and_hard_budget(self):
        config, parent, _ = load_d2_config(DEFAULT_CONFIG)
        self.assertEqual(config["funnel"]["canary_presentations"], 2500)
        self.assertEqual(config["funnel"]["science_total_presentations"], 12500)
        self.assertEqual(config["funnel"]["maximum_programme_presentations_including_replication"], 30000)
        self.assertEqual(config["funnel"]["internal_sample_draws"], 32)
        self.assertEqual(config["sampler"]["deterministic_ladder_network_evaluations"], [50, 100])
        self.assertEqual(
            config["ph006_gate"][
                "primary_paired_energy_relative_improvement_over_f3l2b_minimum"
            ],
            0.02,
        )
        self.assertTrue(
            config["ph006_gate"][
                "positive_paired_core_energy_improvement_over_g1_required"
            ]
        )
        self.assertTrue(
            config["ph006_gate"][
                "positive_paired_core_energy_improvement_over_f3l2b_required"
            ]
        )
        self.assertEqual(config["reproducibility"], CUDA_DETERMINISM_POLICY)
        self.assertEqual(
            config["evaluation"]["deployable_conditioning_gates"][-3:],
            [
                "frozen_g1_mean_scaled",
                "frozen_g1_log_std",
                "frozen_g1_traceless_shear_amplitude",
            ],
        )
        self.assertIn(
            "galaxy-sampled lambda1/lambda2/lambda3/gap12/gap23",
            config["evaluation"]["conditional_coverage_gate_scope"],
        )
        self.assertEqual(parent["schema_version"], "p12f3-conditional-calibration-v1")

    def test_ph006_primary_energy_requires_material_two_percent_gain(self):
        reference = {
            "proper_scores": {"energy": 1.0},
            "per_core_proper_scores": [
                {"core_id": index, "energy": 1.0} for index in range(256)
            ],
        }
        for candidate_energy, expected in ((0.985, False), (0.975, True)):
            candidate = {
                "proper_scores": {"energy": candidate_energy},
                "per_core_proper_scores": [
                    {"core_id": index, "energy": candidate_energy}
                    for index in range(256)
                ],
            }
            paired = paired_ph006_energy(candidate, reference, repeats=200, seed=19)
            # Both deterministic fixtures exclude zero; only the >=2% arm is material.
            self.assertTrue(paired["pass"])
            self.assertEqual(paired_energy_materiality(paired, 0.02)["pass"], expected)

    def test_ph006_requires_paired_energy_improvement_over_g1_separately(self):
        config, _, _ = load_d2_config(DEFAULT_CONFIG)
        proper = {
            "energy": 0.95,
            "coarse_energy": 0.95,
            "marginal_crps": 0.95,
            "variogram_p0p5": 0.95,
        }
        common = {
            "finite_non_degenerate": True,
            "physics_closure": {
                "all_finite": True,
                "all_ordered": True,
                "additional_gaussian_smoothing": False,
                "maximum_trace_max_abs": 0.0,
            },
            "tarp": {
                "ordered_eigenvalues": {"full_max_abs_ecp_minus_alpha": 0.01},
                "eigengaps": {"full_max_abs_ecp_minus_alpha": 0.01},
            },
            "global_coverage_error": {"68": 0.01, "90": 0.01},
            "maximum_deployable_conditional_coverage_error": 0.02,
            "maximum_voxel_deployable_conditional_coverage_error": 0.02,
            "maximum_derived_deployable_conditional_coverage_error": 0.02,
            "proper_scores": proper,
            "per_core_proper_scores": [
                {"core_id": index, "energy": 0.95} for index in range(256)
            ],
        }
        shear = {
            "joint_tarp_blocked": {"full_max_abs_ecp_minus_alpha": 0.01},
            "maximum_marginal_coverage_error": 0.01,
        }
        references = {
            # The global G1 guard passes, while its paired core contrast
            # deliberately contradicts a candidate improvement.
            "g1": {
                "proper_scores": dict(proper, energy=0.96),
                "per_core_proper_scores": [
                    {"core_id": index, "energy": 0.94} for index in range(256)
                ],
            },
            "f3l2b": {
                "proper_scores": {name: 1.0 for name in proper},
                "per_core_proper_scores": [
                    {"core_id": index, "energy": 1.0} for index in range(256)
                ],
            },
            "f3l2d_nfe100": {
                "proper_scores": {name: 1.0 for name in proper},
                "per_core_proper_scores": [
                    {"core_id": index, "energy": 1.0} for index in range(256)
                ],
            },
        }
        result = science_gates(
            ({}, common, shear, {"posterior_to_truth_power": [1.0, 1.0]}, {}),
            references,
            config,
            repeats=100,
            seed=9,
        )
        self.assertFalse(result["checks"]["paired_energy_improves_g1"])
        self.assertTrue(result["checks"]["paired_energy_improves_f3l2b"])
        self.assertTrue(result["checks"]["f3l2b_primary_energy_materiality"])
        self.assertFalse(result["pass"])

    def test_deterministic_policy_is_frozen_and_cpu_validation_is_side_effect_free(self):
        status = configure_d2_determinism(CUDA_DETERMINISM_POLICY, "cpu")
        self.assertFalse(status["applied"])
        changed = dict(CUDA_DETERMINISM_POLICY, allow_tf32=True)
        with self.assertRaises(RuntimeError):
            configure_d2_determinism(changed, "cpu")

    def test_config_rejects_groupnorm_or_phase_drift(self):
        payload = json.loads(DEFAULT_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            payload["diffusion"]["residual_blocks"] = "3d_groupnorm"
            path.write_text(json.dumps(payload))
            with self.assertRaises(RuntimeError):
                load_d2_config(path)
        payload = json.loads(DEFAULT_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            payload["roles"]["training"][0] = "ph001"
            path.write_text(json.dumps(payload))
            with self.assertRaises(PermissionError):
                load_d2_config(path)

    def test_model_is_patch_safe_and_time_conditioned(self):
        config, _, _ = load_d2_config(DEFAULT_CONFIG)
        model = build_model(config, "modern_base4")
        self.assertTrue(any(isinstance(module, ChannelLayerNorm3d) for module in model.modules()))
        forbidden = (nn.GroupNorm, nn.InstanceNorm3d, nn.BatchNorm3d)
        self.assertFalse(any(isinstance(module, forbidden) for module in model.modules()))
        self.assertTrue(
            all(
                module.padding_mode == "zeros"
                for module in model.modules()
                if isinstance(module, nn.Conv3d)
            )
        )
        layout, whitening = layout_and_whitening()
        state = torch.randn(2, layout.components)
        condition = torch.randn(2, 7, 15, 15, 15)
        first = model(
            state,
            torch.full((2,), 0.2),
            condition,
            layout=layout,
            whitening=whitening,
        )
        second = model(
            state,
            torch.full((2,), 0.8),
            condition,
            layout=layout,
            whitening=whitening,
        )
        self.assertEqual(first.shape, state.shape)
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreater(float(torch.max(torch.abs(first - second)).detach()), 1.0e-7)
        first.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_attention_is_capacity_ordered_and_support_aware(self):
        config, _, _ = load_d2_config(DEFAULT_CONFIG)
        base4 = build_model(config, "modern_base4")
        base8 = build_model(config, "modern_base8")
        attention = build_model(config, "modern_base8_attention")
        self.assertEqual(config["matched_contract"]["mask_only_metadata"].split()[0], "support_random")
        self.assertEqual(
            config["arms"]["modern_base8_attention"]["attention_support_metadata"],
            "support_random",
        )
        self.assertLess(parameter_count(base4), parameter_count(base8))
        self.assertLess(parameter_count(base8), parameter_count(attention))
        layout, whitening = layout_and_whitening()
        state = torch.randn(1, layout.components)
        condition = torch.randn(1, 7, 15, 15, 15)
        # Deliberately make apodized exposure disagree with exact support.  The
        # attention path must consume the separate support_random metadata.
        condition[:, 2] = 0.0
        support = torch.ones(1, 1, 15, 15, 15, dtype=torch.bool)
        result = attention(
            state,
            torch.tensor([0.5]),
            condition,
            layout=layout,
            whitening=whitening,
            support_mask=support,
        )
        self.assertTrue(torch.isfinite(result).all())
        support.zero_()
        with self.assertRaises(RuntimeError):
            attention(
                state,
                torch.tensor([0.5]),
                condition,
                layout=layout,
                whitening=whitening,
                support_mask=support,
            )

    def test_thin_support_propagates_through_exact_downsample_receptive_fields(self):
        support = torch.zeros(1, 1, 15, 15, 15)
        support[0, 0, 1, 13, 7] = 1
        coarse = coarsen_support_any(support, (2, 2, 2), levels=3)
        manual = support
        for _ in range(3):
            manual = torch.nn.functional.max_pool3d(
                manual, kernel_size=3, stride=2, padding=1
            )
        self.assertTrue(torch.equal(coarse, manual.bool()))
        self.assertTrue(torch.any(coarse))

    def test_ema_and_samplers_are_replayable(self):
        config, _, _ = load_d2_config(DEFAULT_CONFIG)
        model = build_model(config, "modern_base4")
        ema = clone_model_state(model)
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        before = {name: value.clone() for name, value in ema.items()}
        update_ema_state(ema, model, decay=0.999, update=1)
        self.assertTrue(any(not torch.equal(before[name], ema[name]) for name in ema))
        layout, whitening = layout_and_whitening()
        condition = torch.randn(1, 7, 15, 15, 15)
        deterministic_a = sample_fourier_d2(
            model,
            condition,
            layout=layout,
            whitening=whitening,
            draws=2,
            steps=2,
            generator=torch.Generator().manual_seed(17),
            eta=0.0,
        )
        deterministic_b = sample_fourier_d2(
            model,
            condition,
            layout=layout,
            whitening=whitening,
            draws=2,
            steps=2,
            generator=torch.Generator().manual_seed(17),
            eta=0.0,
        )
        stochastic_a = sample_fourier_d2(
            model,
            condition,
            layout=layout,
            whitening=whitening,
            draws=2,
            steps=2,
            generator=torch.Generator().manual_seed(23),
            eta=1.0,
        )
        stochastic_b = sample_fourier_d2(
            model,
            condition,
            layout=layout,
            whitening=whitening,
            draws=2,
            steps=2,
            generator=torch.Generator().manual_seed(23),
            eta=1.0,
        )
        torch.testing.assert_close(deterministic_a, deterministic_b)
        torch.testing.assert_close(stochastic_a, stochastic_b)
        self.assertEqual(deterministic_a.shape, (2, 15, 15, 15))

    def test_v_inversion_and_zero_target_oracle_ddim(self):
        target = torch.randn(3, 12)
        generator = torch.Generator().manual_seed(991)
        state, time, velocity = fourier_v_pair(target, generator=generator)
        alpha, sigma = cosine_alpha_sigma(time)
        reconstructed = alpha[:, None] * state - sigma[:, None] * velocity
        torch.testing.assert_close(reconstructed, target, atol=2e-6, rtol=2e-6)

        layout, whitening = layout_and_whitening()
        condition = torch.randn(1, 7, 15, 15, 15)
        sample = sample_fourier_d2(
            ZeroTargetOracle(),
            condition,
            layout=layout,
            whitening=whitening,
            draws=3,
            steps=5,
            generator=torch.Generator().manual_seed(4),
            eta=0.0,
        )
        torch.testing.assert_close(sample, torch.zeros_like(sample), atol=2e-5, rtol=0)

    def test_batched_sampler_replays_at_frozen_batch_size(self):
        config, _, _ = load_d2_config(DEFAULT_CONFIG)
        model = build_model(config, "modern_base4")
        layout, whitening = layout_and_whitening()
        condition = torch.randn(1, 7, 15, 15, 15)
        kwargs = dict(
            layout=layout,
            whitening=whitening,
            draws=5,
            draw_batch=2,
            steps=2,
            eta=1.0,
        )
        first = sample_fourier_d2_batched(
            model,
            condition,
            generator=torch.Generator().manual_seed(88),
            **kwargs,
        )
        second = sample_fourier_d2_batched(
            model,
            condition,
            generator=torch.Generator().manual_seed(88),
            **kwargs,
        )
        torch.testing.assert_close(first, second)

    def test_internal_128_127_split_is_complete_and_disjoint(self):
        phases = ("ph000", "ph002", "ph003", "ph004", "ph005")
        internal = {phase: list(range(index * 100, index * 100 + 51)) for index, phase in enumerate(phases)}
        selected, confirmation = _diagnostic_ref_split(
            internal, phases, selection_count=128, confirmation_count=127
        )
        self.assertEqual(len(selected), 128)
        self.assertEqual(len(confirmation), 127)
        self.assertFalse(set(selected) & set(confirmation))
        self.assertEqual(len(set(selected + confirmation)), 255)

    def test_paired_energy_and_capacity_gate(self):
        reference = diagnostic([1.0, 1.1, 0.9, 1.0])
        candidate = diagnostic([0.96, 1.05, 0.86, 0.96])
        paired = paired_energy(candidate, reference)
        self.assertGreater(paired["relative_improvement"], 0.01)
        config, _, _ = load_d2_config(DEFAULT_CONFIG)
        reference_marker = canary_marker(reference)
        candidate_marker = canary_marker(candidate)
        result = compare(candidate_marker, reference_marker, config, 0.01)
        self.assertTrue(result["eligible"])

    def test_capacity_requires_interval_excluding_zero(self):
        config, _, _ = load_d2_config(DEFAULT_CONFIG)
        reference = diagnostic([1.0, 1.0, 1.0, 1.0])
        # Mean improvement exceeds 1%, but uncertainty puts zero inside one SE.
        candidate = diagnostic([0.75, 1.20, 0.75, 1.20])
        result = compare(canary_marker(candidate), canary_marker(reference), config, 0.01)
        self.assertGreater(result["selection"]["paired_energy"]["relative_improvement"], 0.01)
        self.assertLessEqual(result["selection"]["paired_energy"]["one_standard_error_lower"], 0)
        self.assertFalse(result["eligible"])

    def test_earliest_checkpoint_within_one_se_includes_canary_age(self):
        config, _, _ = load_d2_config(DEFAULT_CONFIG)
        milestones = [2500, 5000, 7500, 10000, 12500]
        alternating = np.tile([-0.10, 0.11], 64)
        rows = []
        for presentation in milestones:
            values = np.full(128, 1.0)
            if presentation == 5000:
                values = values - alternating
            row = diagnostic(values, prefix="same")
            rows.append(
                {
                    "presentations": presentation,
                    "optimizer_update": presentation // 2,
                    "selection": {"raw": row, "ema": row},
                    "confirmation": None,
                }
            )
        selected = select_earliest_within_one_se(rows, set(milestones), config)
        self.assertEqual(selected["selected_presentations"], 2500)
        self.assertEqual(selected["selected_weights"], "raw")

    def test_confirmation_fail_closes_instead_of_switching_runner_up(self):
        selection = {"eligible": True}
        contradicted = {"eligible": False}
        result = confirmation_consistency(selection, contradicted)
        self.assertFalse(result["pass"])
        self.assertFalse(result["decision_reproduced"])

    def test_nfe_identity_allows_distinct_export_runs_but_rejects_mixed_seed(self):
        base = {
            "seed": 42,
            "seed_role": "primary",
            "selected_arm": "modern_base8",
            "selected_presentations": 7500,
            "selected_weights": "ema",
            "checkpoint_sha256": "checkpoint",
            "trained_marker_sha256": "trained",
            "second_seed_license_sha256": None,
            "panel_sha256": "panel",
            "draw_batch": 4,
            "sampler": "deterministic",
            "sampler_eta": 0.0,
            "common_evaluator_seed": 42,
            "matched_reference_marker_sha256": "references",
        }
        first = ({**base, "export_frozen_digest": "nfe50"}, {}, {}, {}, {})
        second = ({**base, "export_frozen_digest": "nfe100"}, {}, {}, {}, {})
        validate_nfe_candidate_identity(
            first,
            second,
            expected_seed=42,
            seed_role="primary",
            common_evaluator_seed=42,
            matched_reference_sha256="references",
        )
        mixed = ({**second[0], "seed": 314159}, {}, {}, {}, {})
        with self.assertRaises(RuntimeError):
            validate_nfe_candidate_identity(
                first,
                mixed,
                expected_seed=42,
                seed_role="primary",
                common_evaluator_seed=42,
                matched_reference_sha256="references",
            )

    def test_sampler_convergence_compares_full_curves_not_equal_maxima(self):
        config, _, _ = load_d2_config(DEFAULT_CONFIG)
        nfe50 = calibration_bundle([0.0, 0.6, 1.0])
        nfe100 = calibration_bundle([0.0, 0.4, 1.0])
        result = sampler_convergence(nfe50, nfe100, config)
        self.assertAlmostEqual(
            result["joint_tarp_pointwise_supremum_changes"]["ordered_eigenvalues"],
            0.2,
        )
        self.assertFalse(result["checks"]["joint_tarp"])
        self.assertFalse(result["pass"])

    def test_transform_roundtrip_is_exact_for_constant_location_scale(self):
        layout, whitening = layout_and_whitening()
        field = torch.randn(1, 1, 15, 15, 15)
        low = unpack_fourier_components(pack_fourier_components(field, layout), layout)
        vector = whiten_components(pack_fourier_components(low, layout), whitening, layout)
        result = roundtrip_metrics(
            vector,
            low,
            torch.zeros_like(low),
            torch.zeros_like(low),
            layout,
            whitening,
        )
        self.assertTrue(result["finite"])
        self.assertLess(result["relative_rmse"], 1e-5)
        np.testing.assert_allclose(
            result["reconstructed_to_target_power_by_registered_band"],
            [1.0, 1.0],
            atol=1e-5,
        )

    def test_state_exact_checkpoint_resume_is_numerically_replayable(self):
        torch.manual_seed(123)
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ema = clone_model_state(model)

        def step(target_model, target_optimizer, target_ema, update):
            target_optimizer.zero_grad(set_to_none=True)
            x = torch.randn(4, 3)
            loss = target_model(x).square().mean()
            loss.backward()
            target_optimizer.step()
            update_ema_state(target_ema, target_model, decay=.999, update=update)
            return float(loss.detach())

        step(model, optimizer, ema, 1)
        payload = checkpoint_payload(
            model=model,
            ema_state=ema,
            optimizer=optimizer,
            optimizer_update=1,
            examples_seen=2,
            frozen_digest="test",
            arm="modern_base4",
            seed=123,
            loss_sum=1.0,
            loss_count=1,
        )
        expected_loss = step(model, optimizer, ema, 2)
        expected_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

        replay = nn.Linear(3, 2)
        replay_optimizer = torch.optim.AdamW(replay.parameters(), lr=1e-3)
        replay.load_state_dict(payload["model"])
        replay_optimizer.load_state_dict(payload["optimizer"])
        replay_ema = {name: value.clone() for name, value in payload["ema_model"].items()}
        restore_checkpoint_rng(payload)
        actual_loss = step(replay, replay_optimizer, replay_ema, 2)
        self.assertAlmostEqual(actual_loss, expected_loss, places=7)
        for name, value in replay.state_dict().items():
            torch.testing.assert_close(value, expected_state[name], atol=1e-7, rtol=1e-7)

    def test_trailing_torn_jsonl_is_dropped_but_interior_corruption_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text('{"optimizer_update": 1}\n{"optimizer_update":')
            _truncate_jsonl_to_update(path, 1)
            self.assertEqual(_read_jsonl_rows(path), [{"optimizer_update": 1}])
            path.write_text('{broken}\n{"optimizer_update": 1}\n')
            with self.assertRaises(RuntimeError):
                _read_jsonl_rows(path)

    def test_terminal_checkpoint_can_reconstruct_missing_terminal_marker(self):
        validate_stop_position(1250, 1250, 1250)
        with self.assertRaises(ValueError):
            validate_stop_position(500, 500, 1250)
        with self.assertRaises(ValueError):
            validate_stop_position(499, 500, 1250)

    def test_second_seed_license_is_bound_to_decision_and_internal_freezes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.json"
            selection = root / "selection.json"
            confirmation = root / "confirmation.json"
            decision = root / "decision.json"
            license_path = root / "license.json"
            contract.write_text("{}")
            selection.write_text("{}")
            confirmation.write_text("{}")
            decision_payload = {
                "schema_version": "p12f3-d2-ph006-seed-decision-v1",
                "seed_pass": True,
                "seed_role": "primary",
                "seed": 42,
                "selected_arm": "modern_base8",
                "selected_presentations": 7500,
                "selected_weights": "ema",
            }
            decision.write_text(json.dumps(decision_payload))
            frozen = {
                "contract_digest": "contract-digest",
                "contract_sha256": sha256(contract),
                "final_selection": str(selection.resolve()),
                "final_selection_sha256": sha256(selection),
                "internal_confirmation": str(confirmation.resolve()),
                "internal_confirmation_sha256": sha256(confirmation),
                "seed42_decision": str(decision.resolve()),
                "seed42_decision_sha256": sha256(decision),
                "selected_arm": "modern_base8",
                "selected_presentations": 7500,
                "selected_weights": "ema",
                "ph001_opened": False,
            }
            license_path.write_text(
                json.dumps(
                    {
                        "schema_version": "p12f3-d2-second-seed-license-v1",
                        "licensed": True,
                        **frozen,
                        "frozen_digest": digest(frozen),
                    }
                )
            )
            marker = _validate_replication_license(
                license_path,
                "modern_base8",
                contract_path=contract,
                contract_marker={"frozen_digest": "contract-digest"},
                selection_marker=selection,
                confirmation_marker=confirmation,
            )
            self.assertTrue(marker["licensed"])
            decision_payload["seed_pass"] = False
            decision.write_text(json.dumps(decision_payload))
            with self.assertRaises(PermissionError):
                _validate_replication_license(
                    license_path,
                    "modern_base8",
                    contract_path=contract,
                    contract_marker={"frozen_digest": "contract-digest"},
                    selection_marker=selection,
                    confirmation_marker=confirmation,
                )


if __name__ == "__main__":
    unittest.main()
