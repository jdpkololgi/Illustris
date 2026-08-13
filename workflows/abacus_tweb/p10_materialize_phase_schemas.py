#!/usr/bin/env python3
"""Materialize phase-specific P3/P4 schemas with invariant physics/settings.

Only ``catalogue_id`` is permitted to differ from the tracked ph000 templates.
The base-schema hashes and a machine-readable difference allow cross-phase audits
to prove that no cell size, coordinate, support, smoothing, or fold contract drifted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--phase-root", type=Path)
    parser.add_argument("--p3-base", type=Path,
                        default=repo / "docs/evidence/p3/p3_field_schema_v1.json")
    parser.add_argument("--p4-base", type=Path,
                        default=repo / "docs/evidence/p4/p4_spatial_schema_v1.json")
    args = parser.parse_args()
    if args.phase not in {f"ph{i:03d}" for i in range(0, 7)}:
        raise RuntimeError(f"unsupported phase: {args.phase}")
    phase_root = args.phase_root or Path(
        f"/pscratch/sd/d/dkololgi/abacus/p10_multiphase/{args.phase}"
    )
    out_dir = phase_root / "contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalogue_id = f"{args.phase}_bgs_bright_full_ngc_sgc_v1"
    records = {}
    for name, base in (("p3", args.p3_base), ("p4", args.p4_base)):
        payload = json.loads(base.read_text())
        old = payload["catalogue_id"]
        payload["catalogue_id"] = catalogue_id
        out = out_dir / f"{name}_schema_v1.json"
        if out.exists():
            if json.loads(out.read_text()) != payload:
                raise RuntimeError(f"existing runtime schema drifted: {out}")
        else:
            atomic_json(out, payload)
        records[name] = {
            "base": str(base.resolve()), "base_sha256": sha256(base),
            "output": str(out.resolve()), "output_sha256": sha256(out),
            "only_allowed_change": {"catalogue_id": {"from": old, "to": catalogue_id}},
        }
    marker = out_dir / "SCHEMAS_COMPLETE.json"
    atomic_json(marker, {"schema_version": "p10-phase-schemas-v1", "phase": args.phase,
                         "catalogue_id": catalogue_id, "records": records, "pass": True})
    print(marker.read_text())


if __name__ == "__main__":
    main()
