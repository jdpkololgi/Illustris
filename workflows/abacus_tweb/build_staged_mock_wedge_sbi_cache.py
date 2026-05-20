#!/usr/bin/env python3
"""Build an SBI/Jraph regression cache for a staged SecondGen mock wedge.

Prerequisites (same node order throughout):
  1) Truth NPZ from ``build_staged_mock_wedge_truth_npz.py`` (lambda1-3, cls).
  2) Graph artifacts from ``build_abacus_graph.py`` on exported points, then
     ``abacus_graph_features_cugraph.py``.

This is a thin wrapper around ``build_abacus_sbi_cache.py`` with staged-mock defaults.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER = REPO_ROOT / "workflows" / "abacus_tweb" / "build_abacus_sbi_cache.py"

DEFAULT_TRUTH_NPZ = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/"
    "staged_mock_wedge_stage3_postcollision_rs7.npz"
)
DEFAULT_GNN_META = (
    "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
    "staged_mock_wedge_stage3_postcollision_rs7_cugraph_gnn_metadata.json"
)
DEFAULT_OUT_CACHE = (
    "/pscratch/sd/d/dkololgi/abacus/sbi_caches/"
    "staged_mock_wedge_stage3_postcollision_rs7_sbi_cache.pkl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gnn-metadata-path", type=Path, default=Path(DEFAULT_GNN_META))
    p.add_argument("--truth-npz-path", type=Path, default=Path(DEFAULT_TRUTH_NPZ))
    p.add_argument("--output-cache-path", type=Path, default=Path(DEFAULT_OUT_CACHE))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--three-targets-only",
        action="store_true",
        help="Build legacy 3-d transformed-increment targets (no 15-d derivatives).",
    )
    p.add_argument(
        "--no-transformed-eig",
        action="store_true",
        help="Use raw scaled eigenvalues instead of transformed increments.",
    )
    p.add_argument("--allow-login-node", action="store_true")
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to build_abacus_sbi_cache.py (prefix with --).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not BUILDER.exists():
        raise FileNotFoundError(f"Missing builder script: {BUILDER}")

    cmd = [
        sys.executable,
        str(BUILDER),
        "--gnn-metadata-path",
        str(args.gnn_metadata_path.expanduser().resolve()),
        "--targets-npz-path",
        str(args.truth_npz_path.expanduser().resolve()),
        "--output-cache-path",
        str(args.output_cache_path.expanduser().resolve()),
        "--seed",
        str(args.seed),
        "--no-apply-y1y5-filter",
        "--no-exclude-invalid-box-index",
    ]
    if args.three_targets_only:
        cmd.append("--three-targets-only")
    if args.no_transformed_eig:
        cmd.append("--no-transformed-eig")
    if args.allow_login_node:
        cmd.append("--allow-login-node")
    if args.extra:
        cmd.extend(args.extra)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
