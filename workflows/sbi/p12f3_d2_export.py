#!/usr/bin/env python3
"""Export frozen P12-F3-D2 ph006 field draws at one registered NFE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from workflows.abacus_tweb.p8_deterministic_common import (
    acquire_run_lock,
    atomic_json,
    sha256,
)
from workflows.sbi.p12f3_conditional_models import reconstruct_conditional_low
from workflows.sbi.p12f3_d2_confirm import SCHEMA as CONFIRMATION_SCHEMA
from workflows.sbi.p12f3_d2_contract import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    TRAINED_SCHEMA,
    digest,
    utc_now,
    validate_frozen_contract,
    validate_output_root,
)
from workflows.sbi.p12f3_d2_models import (
    configure_d2_determinism,
    load_model_state_copy,
    sample_fourier_d2_batched,
)
from workflows.sbi.p12f3_d2_train import build_d2_example, build_model
from workflows.sbi.p12f3_export_hybrid_archive import (
    EvaluationTargetStore,
    atomic_npz,
    core_bounds,
    lowpass_numpy,
)
from workflows.sbi.p12f3_train_conditional_generative import (
    load_config as load_conditional,
    load_location_scale,
)
from workflows.sbi.p12f3_train_fourier_lowmode_flow import _open_common
from workflows.sbi.p12f3_train_lowmode_flow import load_g1_model
from workflows.sbi.p12f_gaussian_controls import correlated_unit_residuals


ARCHIVE_SCHEMA = "p12f-sample-archive-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--selection-marker", type=Path)
    parser.add_argument("--confirmation-marker", type=Path)
    parser.add_argument("--trained-marker", type=Path)
    parser.add_argument("--second-seed-license", type=Path)
    parser.add_argument("--seed-role", choices=("primary", "replication"), default="primary")
    parser.add_argument("--network-evaluations", type=int, choices=(50, 100), required=True)
    parser.add_argument("--sampler", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--stochastic-control-license", type=Path)
    parser.add_argument("--draw-batch", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=6600.0)
    return parser.parse_args()


def _load_safe(path: Path, schema: str) -> dict:
    marker = json.loads(path.read_text())
    if (
        marker.get("schema_version") != schema
        or not marker.get("pass")
        or marker.get("ph001_opened")
    ):
        raise RuntimeError(f"unsafe D2 export input: {path}")
    return marker


@torch.inference_mode()
def sample_low(
    model,
    condition,
    support,
    layout,
    whitening,
    location,
    log_scale,
    *,
    draws: int,
    steps: int,
    seed: int,
    batch: int,
    eta: float,
) -> np.ndarray:
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    standard = sample_fourier_d2_batched(
        model,
        condition,
        layout=layout,
        whitening=whitening,
        draws=draws,
        draw_batch=batch,
        steps=steps,
        generator=generator,
        eta=eta,
        support_mask=support,
    )
    return (
        reconstruct_conditional_low(standard, location, log_scale, layout)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def main() -> None:
    args = parse_args()
    contract_path = args.contract or args.output_root / "D2_CONTRACT_FROZEN.json"
    contract, config = validate_frozen_contract(contract_path, args.config)
    validate_output_root(contract, args.output_root, contract_path)
    deterministic_runtime = configure_d2_determinism(
        config["reproducibility"], args.device
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("D2 ph006 export requires CUDA")
    frozen_draw_batch = int(config["sampler"]["draw_batch"])
    if int(args.draw_batch) != frozen_draw_batch:
        raise RuntimeError("D2 export draw batch differs from the frozen sampler")
    stochastic_license = None
    if args.sampler == "stochastic":
        if (
            args.seed_role != "primary"
            or args.network_evaluations != 100
            or args.stochastic_control_license is None
        ):
            raise PermissionError(
                "D2 stochastic control is a primary-seed NFE100 diagnostic only"
            )
        stochastic_license = _load_safe(
            args.stochastic_control_license,
            "p12f3-d2-stochastic-control-license-v1",
        )
        if (
            not stochastic_license.get("licensed")
            or stochastic_license.get("role") != "diagnostic_only_never_promotable"
            or stochastic_license.get("contract_digest")
            != contract["frozen_digest"]
            or stochastic_license.get("seed_decision_sha256")
            != sha256(Path(stochastic_license.get("seed_decision", "")))
            or stochastic_license.get("frozen_digest")
            != digest(
                {
                    key: stochastic_license[key]
                    for key in (
                        "role",
                        "reason",
                        "seed_decision",
                        "seed_decision_sha256",
                        "contract_digest",
                        "seed",
                        "selected_arm",
                        "selected_presentations",
                        "selected_weights",
                        "ph001_opened",
                    )
                }
            )
        ):
            raise PermissionError("D2 stochastic control was not licensed")
    elif args.stochastic_control_license is not None:
        raise ValueError("stochastic-control license is invalid for deterministic export")
    selection_path = args.selection_marker or args.output_root / "D2_FINAL_SELECTION.json"
    confirmation_path = (
        args.confirmation_marker or args.output_root / "D2_INTERNAL_CONFIRMATION.json"
    )
    selection = _load_safe(selection_path, "p12f3-d2-funnel-selection-v1")
    confirmation = _load_safe(confirmation_path, CONFIRMATION_SCHEMA)
    arm = str(selection.get("selected_arm"))
    primary_seed = int(config["funnel"]["seed"])
    seed = int(
        config["funnel"][
            "seed" if args.seed_role == "primary" else "replication_seed"
        ]
    )
    if (
        selection.get("stage") != "final"
        or selection.get("contract_digest") != contract["frozen_digest"]
        or confirmation.get("selected_arm") != arm
        or confirmation.get("frozen_inputs", {}).get("final_selection_sha256")
        != sha256(selection_path)
    ):
        raise RuntimeError("D2 export selection/confirmation binding changed")
    trained_path = args.trained_marker or (
        args.output_root / "training" / arm / f"seed{seed}_v1" / "D2_TRAINED.json"
    )
    trained = _load_safe(trained_path, TRAINED_SCHEMA)
    checkpoint = Path(trained["checkpoint"])
    if (
        trained.get("arm") != arm
        or int(trained.get("seed", -1)) != seed
        or trained.get("seed_role") != args.seed_role
        or trained.get("checkpoint_sha256") != sha256(checkpoint)
        or trained.get("selected_weights") not in ("raw", "ema")
    ):
        raise RuntimeError("D2 selected checkpoint changed before ph006 export")
    license_record = None
    if args.seed_role == "replication":
        if args.second_seed_license is None:
            raise PermissionError("D2 replication export requires its frozen license")
        license_record = _load_safe(
            args.second_seed_license, "p12f3-d2-second-seed-license-v1"
        )
        if (
            not license_record.get("licensed")
            or license_record.get("selected_arm") != arm
            or int(license_record.get("selected_presentations", -1))
            != int(trained["selected_presentations"])
            or license_record.get("selected_weights") != trained["selected_weights"]
            or license_record.get("contract_digest") != contract["frozen_digest"]
            or license_record.get("contract_sha256") != sha256(contract_path)
            or license_record.get("final_selection") != str(selection_path.resolve())
            or license_record.get("final_selection_sha256") != sha256(selection_path)
            or license_record.get("internal_confirmation")
            != str(confirmation_path.resolve())
            or license_record.get("internal_confirmation_sha256")
            != sha256(confirmation_path)
            or license_record.get("seed42_decision_sha256")
            != sha256(Path(license_record.get("seed42_decision", "")))
        ):
            raise PermissionError("D2 replication export license changed")
        license_frozen = {
            key: license_record[key]
            for key in (
                "contract_digest",
                "contract_sha256",
                "final_selection",
                "final_selection_sha256",
                "internal_confirmation",
                "internal_confirmation_sha256",
                "seed42_decision",
                "seed42_decision_sha256",
                "selected_arm",
                "selected_presentations",
                "selected_weights",
                "ph001_opened",
            )
        }
        if license_record.get("frozen_digest") != digest(license_frozen):
            raise PermissionError("D2 replication licence digest changed")
    elif int(trained.get("seed", -1)) != primary_seed:
        raise RuntimeError("D2 primary export seed changed")
    if stochastic_license is not None:
        seed_decision_path = Path(stochastic_license["seed_decision"])
        seed_decision = json.loads(seed_decision_path.read_text())
        if (
            seed_decision.get("schema_version")
            != "p12f3-d2-ph006-seed-decision-v1"
            or seed_decision.get("seed_role") != "primary"
            or int(seed_decision.get("seed", -1)) != primary_seed
            or seed_decision.get("selected_arm") != arm
            or int(seed_decision.get("selected_presentations", -1))
            != int(trained["selected_presentations"])
            or seed_decision.get("selected_weights") != trained["selected_weights"]
            or int(stochastic_license.get("seed", -1)) != primary_seed
            or stochastic_license.get("selected_arm") != arm
            or int(stochastic_license.get("selected_presentations", -1))
            != int(trained["selected_presentations"])
            or stochastic_license.get("selected_weights") != trained["selected_weights"]
            or seed_decision.get("frozen_inputs", {}).get("contract_digest")
            != contract["frozen_digest"]
        ):
            raise PermissionError("D2 stochastic-control decision identity changed")

    panel_path = Path(contract["frozen"]["source_paths"]["ph006_panel"])
    panel = json.loads(panel_path.read_text())
    core_ids = [int(value) for value in panel.get("selected_core_id", ())]
    if (
        panel.get("phase") != "ph006"
        or panel.get("selection_uses_truth")
        or panel.get("ph001_opened")
        or len(core_ids) != 256
        or core_ids != contract["frozen"]["reference_contract"]["g1"]["core_ids"]
    ):
        raise RuntimeError("D2 ph006 panel changed")
    output = (
        args.output_root
        / "evaluation"
        / f"seed{seed}_v1"
        / (
            f"d2_{arm}_nfe{args.network_evaluations}"
            + ("_eta1_diagnostic" if args.sampler == "stochastic" else "")
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    lock = acquire_run_lock(output / ".run.lock", purpose="P12-F3-D2 ph006 export")
    archive_path = output / "P12F_SAMPLE_ARCHIVE.json"
    if archive_path.exists():
        archive = _load_safe(archive_path, ARCHIVE_SCHEMA)
        archive_run_path = Path(archive.get("export_run_manifest", ""))
        archive_run = json.loads(archive_run_path.read_text())
        target_scaler_path = Path(archive.get("target_scaler", ""))
        if (
            not archive.get("pass")
            or archive.get("truth_files_read") != ["ph006"]
            or archive.get("ph001_opened")
            or archive.get("phase") != "ph006"
            or int(archive.get("draws", -1)) != 64
            or archive.get("checkpoint_sha256") != sha256(checkpoint)
            or archive.get("d2_contract_sha256") != sha256(contract_path)
            or archive.get("trained_marker_sha256") != sha256(trained_path)
            or int(archive.get("seed", -1)) != seed
            or archive.get("seed_role") != args.seed_role
            or archive.get("selected_arm") != arm
            or int(archive.get("selected_presentations", -1))
            != int(trained["selected_presentations"])
            or archive.get("selected_weights") != trained["selected_weights"]
            or archive.get("panel_sha256") != sha256(panel_path)
            or archive.get("target_scaler_sha256") != sha256(target_scaler_path)
            or archive.get("export_run_manifest_sha256") != sha256(archive_run_path)
            or archive_run.get("schema_version") != "p12f3-d2-export-run-v1"
            or archive_run.get("frozen_digest") != archive.get("export_frozen_digest")
            or archive_run.get("frozen_digest")
            != digest(archive_run.get("frozen", {}))
            or archive_run.get("frozen", {}).get("seed") != seed
            or archive_run.get("frozen", {}).get("seed_role") != args.seed_role
            or archive_run.get("frozen", {}).get("selected_arm") != arm
            or archive_run.get("ph001_opened")
            or int(archive.get("network_evaluations", -1))
            != args.network_evaluations
            or int(archive.get("draw_batch", -1)) != frozen_draw_batch
            or archive.get("second_seed_license_sha256")
            != (None if args.second_seed_license is None else sha256(args.second_seed_license))
            or archive.get("stochastic_control_license_sha256")
            != (
                None
                if args.stochastic_control_license is None
                else sha256(args.stochastic_control_license)
            )
            or archive.get("sampler") != args.sampler
            or float(archive.get("sampler_eta", -1))
            != (1.0 if args.sampler == "stochastic" else 0.0)
            or [int(row["core_id"]) for row in archive.get("entries", ())]
            != core_ids
            or any(sha256(Path(row["path"])) != row["sha256"] for row in archive["entries"])
        ):
            raise RuntimeError("existing D2 archive checkpoint changed")
        print(json.dumps(archive, indent=2, sort_keys=True))
        lock.close()
        return
    if any(path.name != ".run.lock" for path in output.iterdir()) and not args.resume:
        lock.close()
        raise RuntimeError("non-empty D2 archive requires explicit --resume")

    parent_path = Path(config["sources"]["parent_config"])
    if not parent_path.is_absolute():
        parent_path = Path(__file__).resolve().parents[2] / parent_path
    conditional, f3_parent, _ = load_conditional(parent_path)
    _, _, phases, validation, _, loader, store, opened_selected = _open_common(f3_parent)
    response_store = EvaluationTargetStore(Path(f3_parent["sources"]["phase_root"]))
    try:
        if validation != "ph006" or opened_selected != contract["frozen"]["selected_core_ids"]:
            raise RuntimeError("D2 ph006 runtime data contract changed")
        location_args = SimpleNamespace(
            output_root=Path(config["sources"]["conditional_output_root"]),
            gaussian_arm=config["sources"]["conditional_gaussian_arm"],
            gaussian_run=config["sources"]["conditional_gaussian_run"],
        )
        location_model, _, _, _ = load_location_scale(
            location_args, conditional, args.device
        )
        whitening_marker = json.loads(
            Path(config["sources"]["conditional_whitening"]).read_text()
        )
        if (
            not whitening_marker.get("pass")
            or whitening_marker.get("validation_phase_used_for_fit")
            or whitening_marker.get("ph001_opened")
        ):
            raise RuntimeError("unsafe D2 ph006 whitening")
        whitening = whitening_marker["whitening"]
        g1_model, scaler = load_g1_model(f3_parent, args.device)
        g1_filter_path = Path(f3_parent["sources"]["g1_filter"])
        g1_filter = json.loads(g1_filter_path.read_text())
        model = build_model(config, arm).to(args.device)
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        load_model_state_copy(
            model,
            state["model"] if trained["selected_weights"] == "raw" else state["ema_model"],
        )
        model.eval().requires_grad_(False)
        frozen = {
            "contract": str(contract_path.resolve()),
            "contract_sha256": sha256(contract_path),
            "contract_digest": contract["frozen_digest"],
            "deterministic_runtime": deterministic_runtime,
            "selection_sha256": sha256(selection_path),
            "confirmation_sha256": sha256(confirmation_path),
            "trained_marker_sha256": sha256(trained_path),
            "checkpoint_sha256": sha256(checkpoint),
            "selected_presentations": int(trained["selected_presentations"]),
            "selected_weights": trained["selected_weights"],
            "seed": seed,
            "seed_role": args.seed_role,
            "selected_arm": arm,
            "network_evaluations": int(args.network_evaluations),
            "sampler": args.sampler,
            "sampler_eta": 1.0 if args.sampler == "stochastic" else 0.0,
            "draws": 64,
            "draw_batch": frozen_draw_batch,
            "panel_sha256": sha256(panel_path),
            "g1_filter_sha256": sha256(g1_filter_path),
            "second_seed_license_sha256": None
            if args.second_seed_license is None
            else sha256(args.second_seed_license),
            "stochastic_control_license_sha256": None
            if args.stochastic_control_license is None
            else sha256(args.stochastic_control_license),
            "ph001_opened": False,
        }
        frozen_digest = digest(frozen)
        run_path = output / "run_manifest.json"
        if run_path.exists():
            run = json.loads(run_path.read_text())
            if (
                run.get("schema_version") != "p12f3-d2-export-run-v1"
                or run.get("frozen") != frozen
                or run.get("frozen_digest") != frozen_digest
                or run.get("frozen_digest") != digest(run.get("frozen", {}))
                or run.get("ph001_opened")
            ):
                raise RuntimeError("D2 ph006 export resume contract changed")
        else:
            atomic_json(
                run_path,
                {
                    "schema_version": "p12f3-d2-export-run-v1",
                    "created_utc": utc_now(),
                    "frozen": frozen,
                    "frozen_digest": frozen_digest,
                    "truth_files_read": ["ph006"],
                    "ph001_opened": False,
                },
            )
        progress_path = output / "SAMPLE_ARCHIVE_PROGRESS.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if (
                progress.get("schema_version") != "p12f3-d2-sample-progress-v1"
                or progress.get("frozen_digest") != frozen_digest
                or progress.get("ph001_opened")
            ):
                raise RuntimeError("D2 ph006 export progress contract changed")
            entries = list(progress.get("entries", ()))
        else:
            entries = []
        progress_ids = [int(row["core_id"]) for row in entries]
        if (
            len(progress_ids) != len(set(progress_ids))
            or progress_ids != core_ids[: len(progress_ids)]
        ):
            raise RuntimeError("D2 ph006 progress is duplicate or not a panel prefix")
        complete = {int(row["core_id"]): row for row in entries}
        started = time.monotonic()
        adapter = loader.field_adapter("ph006")
        for ordinal, core_id in enumerate(core_ids):
            if core_id in complete:
                if sha256(Path(complete[core_id]["path"])) != complete[core_id]["sha256"]:
                    raise RuntimeError("D2 completed ph006 core changed")
                continue
            condition, _, layout, location, log_scale, patch, support, _ = build_d2_example(
                loader=loader,
                store=store,
                g1_model=g1_model,
                location_model=location_model,
                scaler=scaler,
                phase="ph006",
                core_id=core_id,
                conditional_config=conditional,
                f3_parent=f3_parent,
                device=args.device,
                whitening=whitening,
            )
            # Recompute only the already-frozen conditional mean/log-std needed
            # for the matched high-frequency G1 completion.
            from workflows.sbi.p12f3_conditional_models import proxy_condition

            condition_check, g1_mean, g1_log_std = proxy_condition(
                patch,
                loader.field_normalization,
                g1_model,
                device=args.device,
                arm="proxy7",
            )
            torch.testing.assert_close(condition, condition_check)
            shape = tuple(g1_mean.shape[-3:])
            core_seed = 43_000 + core_id
            unit = correlated_unit_residuals(
                g1_filter, draws=64, seed=core_seed, shape=shape
            )
            g1_residual = np.exp(g1_log_std[0, 0].cpu().numpy())[None] * unit
            g1_low = lowpass_numpy(
                g1_residual,
                voxel_mpc_h=5.0,
                maximum_k=float(config["matched_contract"]["band_edges_h_mpc"][-1]),
            )
            high = g1_residual - g1_low
            low = sample_low(
                model,
                condition,
                support,
                layout,
                whitening,
                location,
                log_scale,
                draws=64,
                steps=int(args.network_evaluations),
                seed=core_seed + 100_000_000,
                batch=int(args.draw_batch),
                eta=1.0 if args.sampler == "stochastic" else 0.0,
            )
            mean_scaled = g1_mean[0, 0].cpu().numpy()
            samples = (
                (mean_scaled[None] + high + low) * np.float32(scaler["std"])
                + np.float32(scaler["mean"])
            ).astype(np.float32)
            target = response_store.extract(patch)
            counts = np.asarray(
                patch.values[patch.channel_names.index("counts")], dtype=np.float32
            )
            core_path = output / f"core_{core_id:08d}.npz"
            atomic_npz(
                core_path,
                delta_samples=samples,
                delta_truth=np.asarray(target["delta"], dtype=np.float32),
                g1_base_mean=(
                    mean_scaled * np.float32(scaler["std"])
                    + np.float32(scaler["mean"])
                ).astype(np.float32),
                frozen_g1_mean_scaled=condition[0, 3].detach().cpu().numpy().astype(
                    np.float32
                ),
                frozen_g1_log_std=condition[0, 4].detach().cpu().numpy().astype(
                    np.float32
                ),
                frozen_g1_traceless_shear_amplitude=condition[0, 5]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                posterior_sample_mean=np.mean(samples, axis=0, dtype=np.float64).astype(
                    np.float32
                ),
                support=np.asarray(target["support"], dtype=np.uint8),
                angular_response=np.asarray(target["angular_response"], dtype=np.float32),
                boundary_distance_mpc=np.asarray(target["boundary_distance"], dtype=np.float32),
                tracer_density=counts / np.float32(5.0**3),
                core_bounds=core_bounds(patch),
                galaxy_frac_index_local=np.asarray(
                    patch.authoritative_frac_index_local, dtype=np.float32
                ),
            )
            row = {
                "core_id": core_id,
                "path": str(core_path.resolve()),
                "sha256": sha256(core_path),
                "seed": core_seed,
                "shape": list(shape),
                "modes": int(layout.modes),
                "components": int(layout.components),
            }
            entries.append(row)
            complete[core_id] = row
            atomic_json(
                progress_path,
                {
                    "schema_version": "p12f3-d2-sample-progress-v1",
                    "frozen_digest": frozen_digest,
                    "entries": entries,
                    "ph001_opened": False,
                },
            )
            print(
                json.dumps(
                    {
                        "method": f"d2_{arm}_nfe{args.network_evaluations}",
                        "core": ordinal + 1,
                        "total": len(core_ids),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            if time.monotonic() - started >= args.max_wall_seconds:
                raise SystemExit(75)
        scaler_path = output / "target_scaler.json"
        atomic_json(scaler_path, scaler)
        archive = {
            "schema_version": ARCHIVE_SCHEMA,
            "created_utc": utc_now(),
            "method": (
                f"d2_{arm}_seed{seed}_nfe{args.network_evaluations}"
                + ("_eta1_diagnostic" if args.sampler == "stochastic" else "")
            ),
            "phase": "ph006",
            "draws": 64,
            "panel_marker": str(panel_path.resolve()),
            "panel_sha256": sha256(panel_path),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "conditioning_contract_sha256": contract["frozen"]["source_hashes"][
                "response_loader_ready"
            ],
            "target_scaler": str(scaler_path.resolve()),
            "target_scaler_sha256": sha256(scaler_path),
            "d2_contract": str(contract_path.resolve()),
            "d2_contract_sha256": sha256(contract_path),
            "trained_marker": str(trained_path.resolve()),
            "trained_marker_sha256": sha256(trained_path),
            "network_evaluations": int(args.network_evaluations),
            "draw_batch": frozen_draw_batch,
            "sampler": args.sampler,
            "sampler_eta": 1.0 if args.sampler == "stochastic" else 0.0,
            "seed": seed,
            "seed_role": args.seed_role,
            "selected_arm": arm,
            "selected_presentations": int(trained["selected_presentations"]),
            "selected_weights": trained["selected_weights"],
            "second_seed_license_sha256": None
            if args.second_seed_license is None
            else sha256(args.second_seed_license),
            "stochastic_control_license_sha256": None
            if args.stochastic_control_license is None
            else sha256(args.stochastic_control_license),
            "export_frozen_digest": frozen_digest,
            "export_run_manifest": str(run_path.resolve()),
            "export_run_manifest_sha256": sha256(run_path),
            "entries": [complete[core_id] for core_id in core_ids],
            "truth_files_read": ["ph006"],
            "ph001_opened": False,
            "pass": True,
        }
        atomic_json(archive_path, archive)
        print(json.dumps(archive, indent=2, sort_keys=True))
    finally:
        response_store.close()
        store.close()
        loader.close()
        lock.close()


if __name__ == "__main__":
    main()
