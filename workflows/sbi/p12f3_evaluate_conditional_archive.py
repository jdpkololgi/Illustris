#!/usr/bin/env python3
"""Apply the common frozen P12-F evaluator to one conditional rescue archive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_common_evaluator import evaluate_records, load_core_record


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda");parser.add_argument("--seed",type=int,default=20260903)
    return parser.parse_args()


def main():
    args=parse_args();manifest=json.loads(args.archive.read_text())
    if manifest.get("schema_version")!="p12f-sample-archive-v1" or manifest.get("phase")!="ph006" or manifest.get("ph001_opened") or manifest.get("truth_files_read")!=["ph006"] or int(manifest.get("draws",-1))!=64: raise RuntimeError("unsafe conditional sample archive")
    panel=json.loads(Path(manifest["panel_marker"]).read_text());metadata={int(row["core_id"]):row for row in panel["selected_core_metadata"]}
    if len(manifest.get("entries",[]))!=256 or [int(row["core_id"]) for row in manifest["entries"]]!=[int(v) for v in panel["selected_core_id"]]: raise RuntimeError("conditional archive panel changed")
    records=[]
    for row in manifest["entries"]:
        path=Path(row["path"])
        if sha256(path)!=row["sha256"] or "ph001" in str(path).lower(): raise RuntimeError("conditional archive core changed")
        records.append((metadata[int(row["core_id"])],load_core_record(row,64)))
    report=evaluate_records(records,method=manifest["method"],seed=args.seed,device=args.device)
    report.update({"schema_version":"p12f3-conditional-common-evaluation-v1","created_utc":datetime.now(timezone.utc).isoformat(),"archive":str(args.archive.resolve()),"archive_sha256":sha256(args.archive),"truth_files_read":["ph006"],"ph001_opened":False})
    atomic_json(args.output,report);print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__": main()
