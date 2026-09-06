#!/usr/bin/env python3
"""Plot the frozen D2 capacity contrast; never read targets or checkpoints.

This is a reporting-only derivative, not part of model selection or the frozen
ph006 evaluator. Error bars reproduce the registered paired one-standard-error
convention; they are not 95% intervals or independent-voxel errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def chart_rows(report):
    if (report.get("schema_version") != "p12f3-d2-internal-confirmation-v1"
            or report.get("pass") is not True
            or report.get("ph006_used_for_selection") is not False
            or report.get("ph001_opened") is not False
            or report.get("selected_arm") != "modern_base4"
            or report.get("frozen_digest") != digest(report.get("frozen_inputs", {}))):
        raise ValueError("Not the passing, frozen, training-internal base4 confirmation")
    contrast = report["paired_contrasts"]["capacity"]
    if (contrast.get("reference") != "modern_base4"
            or contrast.get("candidate") != "modern_base8"
            or contrast.get("pass") is not True
            or contrast.get("consistency", {}).get("decision_reproduced") is not True):
        raise ValueError("Capacity decision was not reproduced")
    rows = []
    for stage, cores in (("selection", 128), ("confirmation", 127)):
        result = contrast[stage]["selection"]
        paired = result["paired_energy"]
        if (paired["cores"] != cores or result["eligible"] is not False
                or result["paired_interval_convention"]
                != "mean_minus_one_standard_error_strictly_positive"):
            raise ValueError("Unexpected panel or interval convention")
        relative = paired["relative_improvement"]
        difference = paired["mean_reference_minus_candidate"]
        if not (0 < relative < 0.01 and difference > 0):
            raise ValueError("Report does not show the registered sub-materiality gain")
        # Match the stored normalization, not an unpaired comparison of panels.
        reference_mean = difference / relative
        rows.append((stage, cores, 100 * relative,
                     100 * paired["standard_error"] / reference_mean))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    rows = chart_rows(json.loads(raw))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    fig.subplots_adjust(left=0.26, bottom=0.30, top=0.76, right=0.95)
    for y, (stage, cores, gain, error) in enumerate(rows):
        ax.errorbar(gain, y, xerr=error, fmt="o", markersize=8,
                    capsize=5, color=("#4e79a7", "#159782")[y])
        ax.text(gain + error + 0.035, y, f"{gain:.3f}%", va="center", fontsize=11)
    ax.axvline(1.0, linestyle="--", color="#b24b39", linewidth=1.5)
    ax.text(1.0, 1.40, "1% required", ha="center", color="#b24b39")
    ax.set_yticks(range(len(rows)), [f"{s.title()}\n{n} internal cores" for s, n, _, _ in rows])
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.55, 1.60)
    ax.set_xlabel("Energy-score improvement of base8 over base4 (%)")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("D2: the wider model's gain is reproducible, but too small", fontsize=14, y=0.96)
    fig.text(0.5, 0.85, "Frozen capacity decision confirmed: continue base4; attention not licensed.",
             ha="center", fontsize=11)
    fig.text(0.5, 0.12, "Error bars: paired core-based ±1 standard error (not 95% intervals).", ha="center", fontsize=10)
    fig.text(0.5, 0.065, "Canary stage: 2,500 patch presentations · 32 draws · NFE50 · no ph006 evaluation", ha="center", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps({"report_sha256": hashlib.sha256(raw).hexdigest(),
                      "figure": str(args.output.resolve()), "rows": rows}))


if __name__ == "__main__":
    main()
