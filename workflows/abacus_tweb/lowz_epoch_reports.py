"""Retain small, complete development reports, independent of checkpoint selection."""
import json
from pathlib import Path


def retain_report(output, report):
    phases = ("ph006", "ph012", "ph013")
    if (not report["complete_phase_coverage"] or tuple(report["validation_phases"]) != phases
            or set(report["per_phase"]) != set(phases)):
        raise ValueError("not the full corrected development panel")
    epochs = set()
    for p in phases:
        row = report["per_phase"][p]
        rt = row["runtime"]
        if not row["complete_phase_coverage"] or row["validation_phase"] != p or rt["patch_failures"]:
            raise ValueError(f"incomplete phase {p}")
        epochs.add((rt["epoch"],rt["global_step"]))
    if len(epochs) != 1:
        raise ValueError("mixed validation checkpoints")
    epoch, step = epochs.pop()
    if not (epoch > 0 and step > 0):
        raise ValueError("invalid epoch or update")
    directory = Path(output)/"validation_reports"
    directory.mkdir(exist_ok=True)
    dest = directory/f"epoch_{epoch:04d}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(dest)


def install(trainer, args):
    """Output-only: return the exact objects computed by the original scorer."""
    original = trainer.aggregate_validation_reports
    output = args.output_root/args.run_name/args.model/f"seed_{args.seed}"
    def with_report(*a, **kw):
        result = original(*a, **kw)
        retain_report(output, result[0])
        return result
    trainer.aggregate_validation_reports = with_report
