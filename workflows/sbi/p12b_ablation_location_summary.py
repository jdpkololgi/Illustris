#!/usr/bin/env python3
"""Lightweight aggregation of existing small internal ablation row summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    marker_path=args.root/"DIAGNOSTICS_COMPLETE.json"
    report=json.loads(marker_path.read_text())
    if not report["sampler_gate_pass"] or not report["technical_completion"]:
        raise RuntimeError("completed diagnostic required")
    baseline=Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12b_unet_representation_v1")
    ready=json.loads((baseline/"P12B_DATA_READY.json").read_text())
    cores=[a["core"] for a in ready["artifacts"] if a["role"]=="internal"]
    output=dict(schema_version="p12b-ablation-location-summary-v1",source_sha256=sha(__file__),
        diagnostic_report_sha256=sha(marker_path),ph001_access=False,ph006_access=False,arms={},input_sha256={})
    for arm in ("frozen","joint"):
        data={}
        for variant in ("original","zero","permuted"):
            rows=[]
            for core in cores:
                path=args.root/arm/variant/f"core_{core:06d}.npz"
                if path.stat().st_size>128*1024:raise RuntimeError("expected small row summary, not sample payload")
                output["input_sha256"][str(path)]=sha(path)
                with np.load(path,allow_pickle=False) as values:
                    rows.append({k:values[k] for k in ("parent","mean","truth","energy","width68","width90")})
            data[variant]={k:np.concatenate([row[k] for row in rows]) for k in rows[0]}
            expected=report["arms"][arm]["scores"][variant]["energy"]
            if not np.isclose(data[variant]["energy"].mean(),expected,rtol=0,atol=1e-12):
                raise RuntimeError("row summaries differ from completed aggregate")
            if not np.array_equal(data[variant]["parent"],data["original"]["parent"]):
                raise RuntimeError("ablation row mismatch")
        base=data["original"]
        summary={}
        for variant,values in data.items():
            delta=np.linalg.norm(values["mean"]-base["mean"],axis=1)
            summary[variant]=dict(rows=len(delta),mean_posterior_mean_displacement=float(delta.mean()),
                posterior_mean_displacement_p50_p90=np.quantile(delta,[.5,.9]).tolist(),
                posterior_mean_vector_rmse=float(np.sqrt(np.mean(np.sum((values["mean"]-values["truth"])**2,axis=1)))),
                mean_width68=values["width68"].mean(0).tolist(),
                relative_mean_width68_change=(values["width68"].mean(0)/base["width68"].mean(0)-1).tolist())
        output["arms"][arm]=summary
    with args.output.open("x") as stream:json.dump(output,stream,indent=2);stream.write("\n")
    print(json.dumps(output["arms"],indent=2))


if __name__=="__main__":main()
