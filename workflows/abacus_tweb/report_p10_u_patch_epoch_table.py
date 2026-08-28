#!/usr/bin/env python3
"""Print matched ph006 macro-R2(lambda1) histories for P10 U-PATCH arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


DEFAULT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")


def read_history(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    rows: dict[int, float] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["epoch"])] = float(row["primary_macro_r2_lambda1"])
    return rows


def mean_history(paths: list[Path]) -> dict[int, float]:
    histories = [read_history(path) for path in paths]
    if not histories or any(not history for history in histories):
        return {}
    epochs = sorted(set.intersection(*(set(history) for history in histories)))
    return {
        epoch: fmean(history[epoch] for history in histories) for epoch in epochs
    }


def print_table(title: str, histories: dict[str, dict[int, float]]) -> None:
    epochs = sorted(set().union(*(history for history in histories.values())))
    print(f"\n### {title}\n")
    print("| Epoch | " + " | ".join(histories) + " |")
    print("|---:" + "|---:" * len(histories) + "|")
    for epoch in epochs:
        values = [
            f"{history[epoch]:.5f}" if epoch in history else "—"
            for history in histories.values()
        ]
        print(f"| {epoch} | " + " | ".join(values) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root

    def history(relative: str) -> dict[int, float]:
        return read_history(root / relative / "epoch_history.jsonl")

    def seed_mean(base: str, seeds: tuple[int, ...] = (42, 43)) -> dict[int, float]:
        return mean_history(
            [root / base / f"seed_{seed}" / "epoch_history.jsonl" for seed in seeds]
        )

    print_table(
        "BRIGHT and response representations",
        {
            "R0 BRIGHT": history("arm_a_training/arm_a_r0_v1/unet/seed_42"),
            "R1 compressed": history("response_training/p3br_r1_v1/unet/seed_42"),
            "R2 assignment (2-seed mean)": seed_mean(
                "response_training/p10_r2_assignment_v1/unet"
            ),
            "R3-RF random field (2-seed mean)": seed_mean(
                "response_training/p10_r3_rf_v1/unet"
            ),
        },
    )
    print_table(
        "Second-field and strict-control representations",
        {
            "BRIGHT+FAINT": history(
                "p12_and_multitracer_training/p10_bf_proxy_v1/unet_multitracer/seed_42"
            ),
            "Old FAINT Null": history(
                "p12_and_multitracer_training/p10_bf_null_v1/unet_multitracer/seed_42"
            ),
            "R3-RF-DM (2-seed mean)": seed_mean(
                "strict_control_training/p10_r3_rf_dm_seed1701_v1/unet_multitracer"
            ),
            "Cross-phase FAINT (2-seed mean)": seed_mean(
                "strict_control_training/p10_bf_xphase_forward_v1/unet_multitracer"
            ),
        },
    )


if __name__ == "__main__":
    main()
