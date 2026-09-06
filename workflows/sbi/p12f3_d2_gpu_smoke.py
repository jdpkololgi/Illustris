#!/usr/bin/env python3
"""GPU memory/throughput and state-resume preflight for P12-F3-D2.

This stage uses training-phase examples only.  It verifies numerical replay of
an interrupted two-update trajectory on the exact GPU topology and exercises
the frozen 32-draw/NFE50 sampler in batches of four for both base8 variants.
It is not an architecture comparison and consumes no registered science-fit
presentations.
"""
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
from workflows.sbi.p12f3_conditional_models import fourier_v_pair
from workflows.sbi.p12f3_d2_contract import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    digest,
    utc_now,
    validate_frozen_contract,
    validate_output_root,
)
from workflows.sbi.p12f3_d2_models import (
    clone_model_state,
    configure_d2_determinism,
    sample_fourier_d2_batched,
    update_ema_state,
)
from workflows.sbi.p12f3_d2_train import (
    build_d2_example,
    build_model,
    checkpoint_payload,
    restore_checkpoint_rng,
)
from workflows.sbi.p12f3_train_conditional_generative import (
    load_config as load_conditional,
    load_location_scale,
)
from workflows.sbi.p12f3_train_fourier_lowmode_flow import _open_common
from workflows.sbi.p12f3_train_lowmode_flow import load_g1_model


SCHEMA = "p12f3-d2-gpu-smoke-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _maximum_state_difference(left: dict, right: dict) -> float:
    if set(left) != set(right):
        raise RuntimeError("D2 GPU smoke state keys differ")
    values = []
    for name in left:
        if torch.is_floating_point(left[name]):
            values.append(
                float(
                    torch.max(
                        torch.abs(left[name].detach().cpu() - right[name].detach().cpu())
                    )
                )
            )
        elif not torch.equal(left[name].detach().cpu(), right[name].detach().cpu()):
            return float("inf")
    return max(values, default=0.0)


def _maximum_nested_difference(left, right) -> float:
    """Return the maximum absolute difference in a checkpoint state tree."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return float("inf")
        left_cpu = left.detach().cpu()
        right_cpu = right.detach().cpu()
        if left_cpu.shape != right_cpu.shape or left_cpu.dtype != right_cpu.dtype:
            return float("inf")
        if torch.is_floating_point(left_cpu):
            return float(torch.max(torch.abs(left_cpu - right_cpu)))
        return 0.0 if torch.equal(left_cpu, right_cpu) else float("inf")
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)) or set(left) != set(right):
            return float("inf")
        return max(
            (_maximum_nested_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if (
            not isinstance(left, (list, tuple))
            or not isinstance(right, (list, tuple))
            or len(left) != len(right)
        ):
            return float("inf")
        return max(
            (_maximum_nested_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right))
        except (TypeError, ValueError):
            return float("inf")
    return 0.0 if left == right else float("inf")


def main() -> None:
    args = parse_args()
    contract_path = args.contract or args.output_root / "D2_CONTRACT_FROZEN.json"
    contract, config = validate_frozen_contract(contract_path, args.config)
    validate_output_root(contract, args.output_root, contract_path)
    deterministic_runtime = configure_d2_determinism(
        config["reproducibility"], args.device
    )
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("D2 GPU smoke requires one compute GPU")
    roundtrip_path = args.output_root / "D2_TRANSFORM_ROUNDTRIP.json"
    roundtrip = json.loads(roundtrip_path.read_text())
    if (
        roundtrip.get("schema_version") != "p12f3-d2-transform-roundtrip-v1"
        or not roundtrip.get("technical_pass")
        or roundtrip.get("frozen", {}).get("contract_digest")
        != contract["frozen_digest"]
        or roundtrip.get("ph006_used")
        or roundtrip.get("ph001_opened")
    ):
        raise RuntimeError("D2 GPU smoke lacks a safe transform round-trip")
    conditional, f3_parent, _ = load_conditional(
        Path(config["sources"]["parent_config"])
        if Path(config["sources"]["parent_config"]).is_absolute()
        else Path(__file__).resolve().parents[2] / config["sources"]["parent_config"]
    )
    phases = tuple(contract["frozen"]["training_phases"])
    refs = [
        (phase, int(core_id))
        for phase in phases
        for core_id in contract["frozen"]["training_core_ids"][phase]
    ][:4]
    frozen = {
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256(contract_path),
        "contract_digest": contract["frozen_digest"],
        "transform_roundtrip_sha256": sha256(roundtrip_path),
        "resume_topology": "one GPU; four presentations; accumulation two; split after two",
        "sampler_arms": ["modern_base8", "modern_base8_attention"],
        "sampler_draws": int(config["funnel"]["internal_sample_draws"]),
        "sampler_draw_batch": int(config["funnel"]["internal_sample_draw_batch"]),
        "sampler_network_evaluations": int(
            config["funnel"]["internal_sample_network_evaluations"]
        ),
        "numerical_replay_absolute_tolerance": float(
            config["reproducibility"]["numerical_replay_absolute_tolerance"]
        ),
        "deterministic_runtime": deterministic_runtime,
        "ph006_used": False,
        "ph001_opened": False,
    }
    frozen_digest = digest(frozen)
    output = args.output_root / "D2_GPU_SMOKE.json"
    lock = acquire_run_lock(args.output_root / ".gpu_smoke.lock", purpose="P12-F3-D2 GPU smoke")
    loader = store = None
    try:
        if output.exists():
            marker = json.loads(output.read_text())
            if (
                marker.get("schema_version") != SCHEMA
                or not marker.get("pass")
                or marker.get("frozen_digest") != frozen_digest
                or marker.get("ph006_used")
                or marker.get("ph001_opened")
            ):
                raise RuntimeError("existing D2 GPU smoke changed")
            print(json.dumps(marker, indent=2, sort_keys=True))
            return

        _, _, opened_phases, _, _, loader, store, selected = _open_common(f3_parent)
        if tuple(opened_phases) != phases or selected != contract["frozen"]["selected_core_ids"]:
            raise RuntimeError("D2 GPU smoke inherited data contract changed")
        location_model, _, _, _ = load_location_scale(
            SimpleNamespace(
                output_root=Path(config["sources"]["conditional_output_root"]),
                gaussian_arm=config["sources"]["conditional_gaussian_arm"],
                gaussian_run=config["sources"]["conditional_gaussian_run"],
            ),
            conditional,
            args.device,
        )
        g1_model, scaler = load_g1_model(f3_parent, args.device)
        whitening = json.loads(
            Path(config["sources"]["conditional_whitening"]).read_text()
        )["whitening"]

        examples = [
            build_d2_example(
                loader=loader,
                store=store,
                g1_model=g1_model,
                location_model=location_model,
                scaler=scaler,
                phase=phase,
                core_id=core_id,
                conditional_config=conditional,
                f3_parent=f3_parent,
                device=args.device,
                whitening=whitening,
            )
            for phase, core_id in refs
        ]
        seed = 760904
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        initial_model = build_model(config, "modern_base4").to(args.device)
        initial_state = {
            name: value.detach().cpu().clone()
            for name, value in initial_model.state_dict().items()
        }
        initial_torch = torch.get_rng_state().clone()
        initial_cuda = [value.clone() for value in torch.cuda.get_rng_state_all()]
        initial_numpy = np.random.get_state()
        del initial_model

        def new_run():
            model = build_model(config, "modern_base4").to(args.device)
            model.load_state_dict(initial_state)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["funnel"]["learning_rate"]),
                weight_decay=float(config["funnel"]["weight_decay"]),
            )
            ema = clone_model_state(model)
            return model, optimizer, ema

        def restore_initial_rng():
            torch.set_rng_state(initial_torch)
            torch.cuda.set_rng_state_all(initial_cuda)
            np.random.set_state(initial_numpy)

        def update(model, optimizer, ema, pair, update_number):
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=args.device)
            for example in pair:
                condition, target, layout, _, _, _, support, _ = example
                state, time_value, desired = fourier_v_pair(target)
                prediction = model(
                    state,
                    time_value,
                    condition,
                    layout=layout,
                    whitening=whitening,
                    support_mask=support,
                )
                loss = torch.mean(torch.square(prediction - desired))
                (loss / 2).backward()
                total += loss.detach() / 2
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["funnel"]["gradient_clip"])
            )
            optimizer.step()
            update_ema_state(ema, model, decay=.999, update=update_number)
            return float(total.detach().cpu())

        full_model, full_optimizer, full_ema = new_run()
        restore_initial_rng()
        full_losses = [
            update(full_model, full_optimizer, full_ema, examples[:2], 1),
            update(full_model, full_optimizer, full_ema, examples[2:], 2),
        ]

        resumed_model, resumed_optimizer, resumed_ema = new_run()
        restore_initial_rng()
        resumed_losses = [
            update(resumed_model, resumed_optimizer, resumed_ema, examples[:2], 1)
        ]
        state = checkpoint_payload(
            model=resumed_model,
            ema_state=resumed_ema,
            optimizer=resumed_optimizer,
            optimizer_update=1,
            examples_seen=2,
            frozen_digest="gpu-smoke",
            arm="modern_base4",
            seed=seed,
            loss_sum=resumed_losses[0],
            loss_count=1,
        )
        del resumed_model, resumed_optimizer, resumed_ema
        resumed_model, resumed_optimizer, resumed_ema = new_run()
        resumed_model.load_state_dict(state["model"])
        resumed_optimizer.load_state_dict(state["optimizer"])
        resumed_ema = {
            name: value.to(args.device).clone()
            for name, value in state["ema_model"].items()
        }
        checkpoint_restore = {
            "model_maximum_difference": _maximum_state_difference(
                state["model"], resumed_model.state_dict()
            ),
            "ema_maximum_difference": _maximum_state_difference(
                state["ema_model"], resumed_ema
            ),
            "optimizer_maximum_difference": _maximum_nested_difference(
                state["optimizer"], resumed_optimizer.state_dict()
            ),
        }
        checkpoint_restore["exact"] = bool(
            max(checkpoint_restore.values()) == 0.0
        )
        restore_checkpoint_rng(state)
        resumed_losses.append(
            update(resumed_model, resumed_optimizer, resumed_ema, examples[2:], 2)
        )
        replay = {
            "uninterrupted_losses": full_losses,
            "resumed_losses": resumed_losses,
            "maximum_loss_difference": float(
                np.max(np.abs(np.asarray(full_losses) - np.asarray(resumed_losses)))
            ),
            "maximum_model_state_difference": _maximum_state_difference(
                full_model.state_dict(), resumed_model.state_dict()
            ),
            "maximum_ema_state_difference": _maximum_state_difference(
                full_ema, resumed_ema
            ),
        }
        del full_model, full_optimizer, full_ema, resumed_model, resumed_optimizer, resumed_ema
        condition, target, layout, _, _, _, support, _ = examples[0]
        del examples[1:], g1_model, location_model
        torch.cuda.empty_cache()

        sampler_rows = []
        for arm in frozen["sampler_arms"]:
            model = build_model(config, arm).to(args.device).train()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["funnel"]["learning_rate"]),
                weight_decay=float(config["funnel"]["weight_decay"]),
            )
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            train_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            train_generator = torch.Generator(device=args.device).manual_seed(
                seed + 100 + len(sampler_rows)
            )
            state, time_value, desired = fourier_v_pair(
                target, generator=train_generator
            )
            prediction = model(
                state,
                time_value,
                condition,
                layout=layout,
                whitening=whitening,
                support_mask=support,
            )
            train_loss = torch.mean(torch.square(prediction - desired))
            train_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["funnel"]["gradient_clip"])
            )
            optimizer.step()
            torch.cuda.synchronize()
            training_seconds = time.monotonic() - train_started
            training_peak = int(torch.cuda.max_memory_allocated())
            if not torch.isfinite(train_loss) or not torch.isfinite(gradient_norm):
                raise RuntimeError(f"D2 {arm} backward smoke is non-finite")

            model.eval().requires_grad_(False)
            del optimizer, prediction, state, desired
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            sample_started = time.monotonic()
            samples = sample_fourier_d2_batched(
                model,
                condition,
                layout=layout,
                whitening=whitening,
                draws=frozen["sampler_draws"],
                draw_batch=frozen["sampler_draw_batch"],
                steps=frozen["sampler_network_evaluations"],
                generator=torch.Generator(device=args.device).manual_seed(seed + len(sampler_rows)),
                eta=0.0,
                support_mask=support,
            )
            torch.cuda.synchronize()
            sampler_rows.append(
                {
                    "arm": arm,
                    "one_update_loss": float(train_loss.detach().cpu()),
                    "one_update_preclip_gradient_norm": float(
                        gradient_norm.detach().cpu()
                    ),
                    "one_update_elapsed_seconds": training_seconds,
                    "one_update_peak_allocated_bytes": training_peak,
                    "shape": list(samples.shape),
                    "finite": bool(torch.all(torch.isfinite(samples))),
                    "sampler_elapsed_seconds": time.monotonic() - sample_started,
                    "sampler_peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                    "gpu_name": torch.cuda.get_device_name(),
                }
            )
            del samples, model, train_loss, gradient_norm
            torch.cuda.empty_cache()
        tolerance = float(frozen["numerical_replay_absolute_tolerance"])
        passed = bool(
            max(
                replay["maximum_loss_difference"],
                replay["maximum_model_state_difference"],
                replay["maximum_ema_state_difference"],
            )
            <= tolerance
            and checkpoint_restore["exact"]
            and all(row["finite"] for row in sampler_rows)
        )
        marker = {
            "schema_version": SCHEMA,
            "created_utc": utc_now(),
            "pass": passed,
            "frozen": frozen,
            "frozen_digest": frozen_digest,
            "resume_replay": replay,
            "checkpoint_restore": checkpoint_restore,
            "sampler_memory_throughput": sampler_rows,
            "claim": (
                "checkpoint serialization restored model, EMA and optimizer exactly, "
                "and the post-update trajectory was numerically replayable within the "
                "registered tolerance on this tested one-GPU topology; this is not a "
                "universal bitwise-determinism claim"
                if passed
                else "GPU replay or sampler preflight failed; no post-update numerical "
                "replayability claim is made (checkpoint-restore equality is reported "
                "separately)"
            ),
            "truth_files_read": [f"{phase} training delta_R7" for phase in phases],
            "ph006_used": False,
            "ph001_opened": False,
        }
        atomic_json(output, marker)
        if not passed:
            raise RuntimeError(
                "D2 GPU smoke failed resume or sampler feasibility; "
                f"frozen failure evidence: {output}"
            )
        print(json.dumps(marker, indent=2, sort_keys=True))
    finally:
        if store is not None:
            store.close()
        if loader is not None:
            loader.close()
        lock.close()


if __name__ == "__main__":
    main()
