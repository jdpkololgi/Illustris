"""Small, deterministic report from the frozen reference receipts (no fitting)."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def report(root, output):
    root,output=Path(root),Path(output)
    done=json.loads((root/"COMPLETE.json").read_text())
    control=json.loads((root/"controls.json").read_text())
    manifest=json.loads((root/"manifest.json").read_text())
    if done["sources"]!=manifest["sources"]:
        raise ValueError("completion provenance mismatch")
    cfg=manifest["config"]
    expected=(len(cfg["fixed_cases"])+cfg["cases"])*len(cfg["seeds"])*len(cfg["objectives"])*len(cfg["checkpoints"])*len(cfg["nfe"])
    if len(done["evaluations"])!=expected:
        raise ValueError("incomplete registered evaluation panel")
    identities={(r["fit"],r["update"],r["case"],r["nfe"]) for r in done["evaluations"]}
    if len(identities)!=expected:
        raise ValueError("duplicate evaluation receipts")
    for row in done["evaluations"]:
        path=root/row["fit"]/f'evaluation_{row["update"]}_{row["case"]}_{row["nfe"]}.json'
        if json.loads(path.read_text())!={k:v for k,v in row.items() if k!="fit"}:
            raise ValueError("receipt/summary disagreement")
    output.mkdir(parents=True,exist_ok=True)
    for name in ("COMPLETE.json","controls.json","manifest.json"):
        shutil.copy2(root/name,output/name)
    lines=["# Learned Gaussian reference results", "",
           f"Allocation {done['job']}; {done['seconds']/60:.2f} minutes in the main runner; "
           f"{done['scratch_bytes']/1024**3:.3f} GiB main outputs. "
           f"{expected} registered evaluation ensembles (512 draws each).",
           "", "These are 8^3 synthetic field diagnostics, not Abacus/DESI validation. "
           "Fixed fits use exact posterior training draws. All tolerances were frozen before fitting.",
           "", "## Exact-sampler controls", "",
           "Maximum relative covariance error across the two observation-mask templates:", "",
           "| NFE | VDM ancestral | CFM Heun |", "|---:|---:|---:|"]
    for nfe in cfg["oracle_nfe"]:
        values=[max(r["covariance_relative"] for r in control["oracle"]
                    if r["objective"]==objective and r["nfe"]==nfe) for objective in cfg["objectives"]]
        lines.append(f"| {nfe} | {values[0]:.6f} | {values[1]:.6f} |")
    lines += ["", "Numerical gate is <=0.05. A failure prevents attributing the corresponding "
              "neural result exclusively to learning. No learned NFE512 run is implied by the exact control.",
              "", "## Metric controls", "", "| Case | Control | Mean RMS | Covariance error | Variance ratio | Octant coverage | Pass |",
              "|---:|---|---:|---:|---:|---:|:---:|"]
    for r in control["metrics"]:
        lines.append(f"| {r['case']} | {r['control']} | {r['mean_rms']:.4f} | {r['covariance_relative']:.4f} | "
                     f"{r['variance_ratio']:.4f} | {r['octant_coverage']:.4f} | {r['passed']} |")
    lines += ["", "## Matched-case learning progression", "",
              "Means across cases 0/1 and the two seeds; pass counts retain all four cells. "
              "These cells are not treated as four independent cosmological universes. "
              "Covariance refers to the fixed 16-dimensional probe covariance.", "",
              "| Objective | Regime | Update | NFE | Mean RMS | Covariance error | Variance ratio | Coverage | Pass cells |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    groups=defaultdict(list)
    for r in done["evaluations"]:
        if r["case"] in cfg["fixed_cases"]:
            mode="amortised" if "amortised" in r["fit"] else "fixed"
            groups[r["objective"],mode,r["update"],r["nfe"]].append(r)
    summaries=[]
    for (objective,mode,update,nfe),rows in sorted(groups.items()):
        vals={k:statistics.mean(r[k] for r in rows) for k in
              ("mean_rms","covariance_relative","variance_ratio","octant_coverage")}
        passed=sum(r["passed"] for r in rows)
        lines.append(f"| {objective} | {mode} | {update} | {nfe} | {vals['mean_rms']:.4f} | "
                     f"{vals['covariance_relative']:.4f} | {vals['variance_ratio']:.4f} | "
                     f"{vals['octant_coverage']:.4f} | {passed}/{len(rows)} |")
        summaries.append(dict(objective=objective,mode=mode,update=update,nfe=nfe,
                              passed=passed,count=len(rows),**vals))
    lines += ["", "## Final per-seed/per-case evidence at NFE256", "",
              "| Fit | Case | Mean RMS | Covariance error | Variance ratio | Coverage | Failing gates |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for r in done["evaluations"]:
        if r["update"]==cfg["updates"] and r["nfe"]==max(cfg["nfe"]):
            failed=", ".join(k for k,v in r["checks"].items() if not v) or "none"
            lines.append(f"| {r['fit']} | {r['case']} | {r['mean_rms']:.4f} | {r['covariance_relative']:.4f} | "
                         f"{r['variance_ratio']:.4f} | {r['octant_coverage']:.4f} | {failed} |")
    lines += ["", "Coverage here is the exact Gaussian posterior probability inside the generated "
              "5%-95% octant intervals. It is not TARP, a test on thousands of independent truths, "
              "or proof of correct full 512-dimensional dependence. The complete receipts also "
              "retain all shell power ratios and voxel variance errors.", "",
              "No lognormal/Poisson or Abacus training was run. Numerical correctness, finite-budget "
              "neural accuracy and real-survey robustness remain separate judgments.", ""]
    (output/"README.md").write_text("\n".join(lines))
    (output/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name!="SHA256.json"}
    (output/"SHA256.json").write_text(json.dumps(checksums,indent=2)+"\n")
    print(json.dumps(summaries,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root"); parser.add_argument("output")
    args=parser.parse_args()
    report(args.root,args.output)
