#!/usr/bin/env python3
"""Create one compact visual-audit summary for a P12-F3 rescue archive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.plot_p12f3_hierarchical_comparison import analyze_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.archive.read_text())
    if (
        manifest.get("schema_version") != "p12f-sample-archive-v1"
        or manifest.get("phase") != "ph006"
        or manifest.get("ph001_opened")
        or manifest.get("truth_files_read") != ["ph006"]
    ):
        raise RuntimeError("unsafe conditional archive for visual analysis")
    summary, _ = analyze_archive(args.archive, device=args.device)
    summary.update(
        {
            "schema_version": "p12f3-conditional-visual-summary-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "archive": str(args.archive.resolve()),
            "archive_sha256": sha256(args.archive),
            "truth_files_read": ["ph006"],
            "ph001_opened": False,
        }
    )
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
