#!/usr/bin/env python3
"""Render the preregistered visual P12-A ph001 blind-calibration summary."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.sbi.p12a_blind_evaluation_contract import SCHEMA as CONTRACT_SCHEMA
from workflows.sbi.p12a_evaluate_blind import (
    RESULT_SCHEMA,
    validate_evaluation_report,
    validate_evaluation_implementation,
)
from workflows.sbi.p12a_immutable_io import (
    publish_file_exclusive,
    write_json_exclusive,
)
from workflows.sbi.p12a_open_blind import validate_evaluation_contract


SCHEMA = "p12a-ph001-blind-plots-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _r2(report: dict, model: str) -> float:
    if model == "P12-A mean":
        return float(report["posterior_mean_r2"][0])
    if model == "U-PATCH":
        return float(report["base_unet_r2"][0])
    estimator = model.lower()
    return float(
        report["classical_deterministic"][estimator]["train_affine_ordered"]
        ["lambda1_lambda2_lambda3"][0]["r2"]
    )


def render(report: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    diagonal = np.linspace(0.0, 1.0, 201)
    axes[0, 0].fill_between(
        diagonal,
        np.maximum(0.0, diagonal - 0.05),
        np.minimum(1.0, diagonal + 0.05),
        color="0.85",
        label=r"registered $|\Delta|\leq0.05$",
    )
    for label, key, color in (
        ("ordered eigenvalues", "joint_eigenvalue_tarp", "#31688e"),
        ("eigengaps", "joint_eigengap_tarp", "#35b779"),
    ):
        curve = report["dependence"][key]
        primary = float(curve.get("maximum_deviation", np.nan))
        p90 = float(curve.get("replicate_p90_maximum_deviation", np.nan))
        diagnostic = (
            f"{label} (primary={primary:.3f}, p90={p90:.3f})"
            if np.isfinite(primary) and np.isfinite(p90)
            else label
        )
        axes[0, 0].plot(
            curve["alpha"],
            curve["expected_coverage_probability"],
            label=diagnostic,
            color=color,
        )
    axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1, label="calibrated")
    axes[0, 0].set(
        xlabel="Nominal TARP credibility",
        ylabel="Empirical coverage",
        title="Joint posterior calibration",
    )
    axes[0, 0].legend(frameon=False)

    x = np.arange(3)
    width = 0.36
    axes[0, 1].bar(x - width / 2, report["coverage68"], width, label="nominal 68%")
    axes[0, 1].bar(x + width / 2, report["coverage90"], width, label="nominal 90%")
    axes[0, 1].axhline(0.68, color="C0", ls="--", lw=1)
    axes[0, 1].axhline(0.90, color="C1", ls="--", lw=1)
    axes[0, 1].set_xticks(x, [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"])
    axes[0, 1].set(ylim=(0.5, 1.0), ylabel="Empirical interval coverage", title="Global marginal coverage")
    axes[0, 1].legend(frameon=False)

    labels, errors, colors = [], [], []
    for stratum, bins in report["conditional_coverage"]["strata"].items():
        for value, row in bins.items():
            labels.append(f"{stratum.replace('_quartile', '')}\n{value}")
            error = float(row["maximum_absolute_error"])
            errors.append(error)
            if error > 0.06:
                colors.append("#d73027")
            elif stratum == "redshift_shell" and value == "3" and error > 0.03:
                colors.append("#fee08b")
            else:
                colors.append("#1a9850")
    axes[1, 0].bar(np.arange(len(errors)), errors, color=colors)
    axes[1, 0].axhline(
        0.03, color="0.25", ls="--", lw=1, label="sparse-shell green gate"
    )
    axes[1, 0].axhline(
        0.06, color="0.25", ls=":", lw=1, label="all conditional release gate"
    )
    axes[1, 0].set_xticks(
        np.arange(len(errors)), labels, rotation=60, ha="right", fontsize=7
    )
    axes[1, 0].set(
        ylabel="Maximum |coverage - nominal|",
        title="Conditional coverage by observed stratum",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    models = ["P12-A mean", "U-PATCH", "CIC", "DTFE"]
    r2 = [_r2(report, model) for model in models]
    axes[1, 1].bar(models, r2, color=["#440154", "#31688e", "#35b779", "#fde725"])
    axes[1, 1].set(ylabel=r"Blind $R^2(\lambda_1)$", title="Matched point-estimator diagnostics")
    axes[1, 1].tick_params(axis="x", rotation=20)

    score = report.get("gaussian_minus_fmpe_joint_energy_score", {})
    score_ci = score.get("ci95", [])
    score_text = ""
    if len(score_ci) == 2 and np.all(np.isfinite(score_ci)):
        score_text = (
            "\nGaussian - FMPE joint energy score "
            f"{float(score.get('mean', np.nan)):.4g} "
            f"[95% core bootstrap {float(score_ci[0]):.4g}, {float(score_ci[1]):.4g}]"
        )
    figure.suptitle(
        f"P12-A ph001 blind evaluation — {report['release_status'].upper()}"
        f"{score_text}",
        fontsize=15,
    )
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
    try:
        figure.savefig(temporary, dpi=180)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        publish_file_exclusive(temporary, output)
    finally:
        plt.close(figure)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_plot_manifest(
    marker: dict,
    *,
    evaluation_report: Path,
    evaluation_contract: Path,
    figure: Path,
    release_status: str,
) -> None:
    if (
        marker.get("schema_version") != SCHEMA
        or marker.get("phase") != "ph001"
        or marker.get("open_count") != 1
        or marker.get("sealed_phase_opened") is not True
        or marker.get("post_open_refit_performed") is not False
        or marker.get("post_open_tuning_allowed") is not False
        or marker.get("release_status") != release_status
        or marker.get("pass") is not True
    ):
        raise RuntimeError("existing plot manifest is not frozen evaluation evidence")
    for name, path in (
        ("evaluation_report", evaluation_report),
        ("evaluation_contract", evaluation_contract),
        ("figure", figure),
    ):
        record = marker.get(name, {})
        if (
            Path(str(record.get("path", ""))).resolve() != path.resolve()
            or record.get("sha256") != sha256(path)
            or ("bytes" in record and int(record["bytes"]) != path.stat().st_size)
        ):
            raise RuntimeError(f"existing blind plot changed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.evaluation_contract.read_text())
    report = json.loads(args.evaluation_report.read_text())
    if contract.get("schema_version") != CONTRACT_SCHEMA or not contract.get("pass"):
        raise RuntimeError("blind plotter requires the frozen evaluation contract")
    validate_evaluation_contract(args.evaluation_contract)
    validate_evaluation_implementation(contract)
    if report.get("schema_version") != RESULT_SCHEMA:
        raise RuntimeError("blind plotter requires the immutable evaluation result")
    expected = contract["canonical_outputs"]
    if args.evaluation_report.resolve() != Path(expected["evaluation_report"]).resolve():
        raise PermissionError("evaluation report is not canonical")
    if args.output.resolve() != Path(expected["evaluation_figure"]).resolve():
        raise PermissionError("blind figure path differs from frozen contract")
    if args.manifest.resolve() != Path(expected["plot_manifest"]).resolve():
        raise PermissionError("blind plot-manifest path differs from frozen contract")
    if report.get("evaluation_contract", {}).get("sha256") != sha256(
        args.evaluation_contract
    ):
        raise RuntimeError("evaluation result does not bind the frozen contract")
    # Rendering is downstream of the scientific decision.  Recompute the full
    # report from the frozen posterior, proper-score sidecar and compact truth;
    # accepting a schema-valid JSON here would allow altered gates or metrics to
    # become the canonical visual evidence.
    report = validate_evaluation_report(
        args.evaluation_report,
        evaluation_contract_path=args.evaluation_contract,
    )
    if args.manifest.exists():
        if not args.output.is_file():
            raise RuntimeError("plot manifest exists without its canonical figure")
        marker = json.loads(args.manifest.read_text())
        validate_plot_manifest(
            marker,
            evaluation_report=args.evaluation_report,
            evaluation_contract=args.evaluation_contract,
            figure=args.output,
            release_status=report["release_status"],
        )
        print(json.dumps(marker, indent=2), flush=True)
        return
    attempt_id = "-".join(
        (
            os.environ.get("SLURM_JOB_ID", "local"),
            os.environ.get("SLURM_RESTART_COUNT", "0"),
            str(os.getpid()),
        )
    )
    attempt = args.output.with_name(
        f".{args.output.stem}.attempt-{attempt_id}.png"
    )
    render(report, attempt)
    try:
        if args.output.exists():
            if sha256(args.output) != sha256(attempt):
                raise RuntimeError(
                    "orphaned canonical figure differs from deterministic rerender"
                )
        else:
            publish_file_exclusive(attempt, args.output)
    finally:
        try:
            attempt.unlink()
        except FileNotFoundError:
            pass
    marker = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "evaluation_report": {
            "path": str(args.evaluation_report.resolve()),
            "sha256": sha256(args.evaluation_report),
        },
        "evaluation_contract": {
            "path": str(args.evaluation_contract.resolve()),
            "sha256": sha256(args.evaluation_contract),
        },
        "figure": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
        "release_status": report["release_status"],
        "post_open_refit_performed": False,
        "post_open_tuning_allowed": False,
        "open_count": 1,
        "sealed_phase_opened": True,
        "pass": True,
    }
    write_json_exclusive(args.manifest, marker)
    print(json.dumps(marker, indent=2), flush=True)


if __name__ == "__main__":
    main()
