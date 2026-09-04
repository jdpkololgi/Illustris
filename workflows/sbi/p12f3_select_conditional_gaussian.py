#!/usr/bin/env python3
"""Select the train-only P12-F3 conditional Gaussian arm with paired cores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f3_conditional_models import ConditionalLowModeGaussianUNet, conditional_gaussian_nll
from workflows.sbi.p12f3_train_conditional_gaussian import build_example, load_config, split_selected
from workflows.sbi.p12f3_train_fourier_lowmode_flow import _open_common
from workflows.sbi.p12f3_train_lowmode_flow import load_g1_model


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION = REPO_ROOT / "configs/p12f3_conditional_selection_v1.json"
DEFAULT_OUTPUT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_conditional_calibration_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-config", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_selection(path: Path) -> tuple[dict, dict, dict, Path]:
    selection = json.loads(path.read_text())
    if (
        selection.get("schema_version") != "p12f3-conditional-selection-v1"
        or selection.get("sealed_blind_phase") != "ph001"
        or selection.get("validation_phase_used_for_fit")
        or set(selection.get("arms", [])) != {"base3", "proxy7", "proxy7_shuffled"}
    ):
        raise RuntimeError("unsafe conditional-selection contract")
    parent_config_path = REPO_ROOT / selection["parent_config"]
    config, parent, parent_path = load_config(parent_config_path)
    return selection, config, parent, parent_path


def load_models(output_root: Path, selection: dict, config: dict, device: str):
    models = {}
    markers = {}
    for arm in selection["arms"]:
        marker_path = output_root / "gaussian" / arm / selection["run_name"] / "P12F3_CONDITIONAL_GAUSSIAN_TRAINED.json"
        marker = json.loads(marker_path.read_text())
        checkpoint = Path(marker["checkpoint"])
        if (
            marker.get("schema_version") != "p12f3-conditional-gaussian-trained-v1"
            or not marker.get("pass")
            or marker.get("arm") != arm
            or marker.get("ph006_used_for_fit")
            or marker.get("ph001_opened")
            or marker.get("checkpoint_sha256") != sha256(checkpoint)
        ):
            raise RuntimeError(f"unsafe trained Gaussian arm {arm}")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model = ConditionalLowModeGaussianUNet(base=int(config["gaussian_control"]["unet_base"])).to(device)
        model.load_state_dict(state["model"], strict=True)
        models[arm] = model.eval().requires_grad_(False)
        markers[arm] = {"path": str(marker_path.resolve()), "sha256": sha256(marker_path)}
    return models, markers


def bootstrap(differences: np.ndarray, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(differences), size=(repeats, len(differences)))
    means = differences[sampled].mean(axis=1)
    return {
        "mean": float(np.mean(differences)),
        "q025_q50_q975": np.quantile(means, (0.025, 0.5, 0.975)).tolist(),
        "probability_mean_below_zero": float(np.mean(means < 0)),
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace, selection: dict, config: dict, parent: dict):
    _, _, phases, _, _, loader, store, selected = _open_common(parent)
    _, internal = split_selected(
        selected,
        phases,
        float(config["training"]["internal_validation_fraction_per_phase"]),
        int(config["training"]["seed"]),
    )
    models, marker_contract = load_models(args.output_root, selection, config, args.device)
    g1_model, scaler = load_g1_model(parent, args.device)
    rows = []
    try:
        for phase in phases:
            for core_id in internal[phase]:
                losses = {}
                for arm in selection["arms"]:
                    condition, target, mask, _, _, _ = build_example(
                        loader=loader,
                        store=store,
                        g1_model=g1_model,
                        scaler=scaler,
                        phase=phase,
                        core_id=int(core_id),
                        config=config,
                        parent=parent,
                        arm=arm,
                        device=args.device,
                    )
                    location, log_scale = models[arm](condition)
                    losses[arm] = float(conditional_gaussian_nll(location, log_scale, target, mask).cpu())
                rows.append({"phase": phase, "core_id": int(core_id), **losses})
                if len(rows) == 1 or len(rows) % 25 == 0:
                    print(json.dumps({"stage": "paired-internal-nll", "cores": len(rows)}), flush=True)
    finally:
        store.close()
        loader.close()
    values = {arm: np.asarray([row[arm] for row in rows]) for arm in selection["arms"]}
    comparisons = {
        "proxy7_minus_base3": bootstrap(
            values["proxy7"] - values["base3"],
            int(selection["paired_core_bootstrap_repeats"]),
            int(selection["seed"]),
        ),
        "proxy7_minus_proxy7_shuffled": bootstrap(
            values["proxy7"] - values["proxy7_shuffled"],
            int(selection["paired_core_bootstrap_repeats"]),
            int(selection["seed"]) + 1,
        ),
    }
    means = {arm: float(np.mean(value)) for arm, value in values.items()}
    gates = selection["aligned_proxy_gate"]
    relative_base = (means["base3"] - means["proxy7"]) / max(abs(means["base3"]), 1.0e-8)
    relative_shuffled = (means["proxy7_shuffled"] - means["proxy7"]) / max(abs(means["proxy7_shuffled"]), 1.0e-8)
    aligned = bool(
        relative_base >= float(gates["minimum_relative_nll_improvement_over_base3"])
        and relative_shuffled >= float(gates["minimum_relative_nll_improvement_over_shuffled"])
        and comparisons["proxy7_minus_base3"]["q025_q50_q975"][2] < 0
        and comparisons["proxy7_minus_proxy7_shuffled"]["q025_q50_q975"][2] < 0
    )
    by_phase = {
        phase: {
            arm: float(np.mean([row[arm] for row in rows if row["phase"] == phase]))
            for arm in selection["arms"]
        }
        for phase in phases
    }
    return {
        "rows": rows,
        "mean_nll": means,
        "by_phase": by_phase,
        "comparisons": comparisons,
        "relative_improvement_over_base3": float(relative_base),
        "relative_improvement_over_shuffled": float(relative_shuffled),
        "aligned_proxy_gate_pass": aligned,
        "selected_arm": "proxy7" if aligned else "base3",
        "marker_contract": marker_contract,
        "internal_core_ids": internal,
    }


def make_plot(result: dict, output: Path) -> None:
    arms = ("base3", "proxy7", "proxy7_shuffled")
    colors = ("#555555", "#d1495b", "#2878b5")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    phases = list(result["by_phase"])
    x = np.arange(len(phases))
    for arm, color in zip(arms, colors, strict=True):
        axes[0].plot(x, [result["by_phase"][phase][arm] for phase in phases], marker="o", color=color, label=arm)
    axes[0].set(xticks=x, xticklabels=phases, ylabel="held-out internal Gaussian NLL", title="Per-phase internal validation")
    axes[0].legend(); axes[0].grid(alpha=.2)
    labels = ("proxy7 - base3", "proxy7 - shuffled")
    keys = ("proxy7_minus_base3", "proxy7_minus_proxy7_shuffled")
    centers = [result["comparisons"][key]["mean"] for key in keys]
    bounds = [result["comparisons"][key]["q025_q50_q975"] for key in keys]
    errors = np.asarray([[center - bound[0], bound[2] - center] for center, bound in zip(centers, bounds, strict=True)]).T
    axes[1].errorbar(np.arange(2), centers, yerr=errors, marker="o", linestyle="none", capsize=5, color="#d1495b")
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(xticks=np.arange(2), xticklabels=labels, ylabel="paired mean NLL difference", title="Core-bootstrap 95% intervals")
    axes[1].grid(alpha=.2)
    figure.suptitle(f"Conditional Gaussian proxy selection: {result['selected_arm']} selected")
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("conditional Gaussian selection requires CUDA")
    selection, config, parent, _ = load_selection(args.selection_config)
    result = evaluate(args, selection, config, parent)
    output = args.output_root / "gaussian" / "P12F3_CONDITIONAL_GAUSSIAN_SELECTION.json"
    report = {
        "schema_version": "p12f3-conditional-gaussian-selection-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pass": True,
        **result,
        "selection_config": str(args.selection_config.resolve()),
        "selection_config_sha256": sha256(args.selection_config),
        "ph006_used_for_fit": False,
        "truth_files_read": ["ph000 and ph002-ph005 internal-validation density targets"],
        "ph001_opened": False,
    }
    make_plot(report, output.with_name("p12f3_conditional_gaussian_selection"))
    report["plot"] = str(output.with_name("p12f3_conditional_gaussian_selection.png").resolve())
    report["plot_sha256"] = sha256(Path(report["plot"]))
    atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
