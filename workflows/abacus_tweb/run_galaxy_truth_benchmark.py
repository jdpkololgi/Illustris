#!/usr/bin/env python3
"""Bounded first-phase repair benchmark; invoked inside one approved CPU step."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import fitsio

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from workflows.abacus_tweb.p10_phase_assets import load_registry
from workflows.abacus_tweb.p10_build_observed_truth import default_lss
from workflows.abacus_tweb.product_lifecycle import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    args = p.parse_args()
    if not (os.environ.get("SLURM_STEP_ID") and socket.gethostname().startswith("nid")):
        p.error("Requires approved CPU compute step")
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "BENCHMARK_STATUS.json"
    if status_path.exists():
        raise FileExistsError("Use a fresh run directory; no automatic retry")
    status = dict(job_id=os.environ["SLURM_JOB_ID"], node=socket.gethostname(),
                  source_root=str(REPO), steps=[], completed=False)
    def save():
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2)+"\n")
        tmp.replace(status_path)
    def run(name, script, *arguments):
        command = [sys.executable, "-u", str(REPO/"workflows/abacus_tweb"/script),
                   *map(str, arguments)]
        record = dict(name=name, command=command, log=str(root/f"{name}.log"), started=time.time())
        status["steps"].append(record); save()
        with Path(record["log"]).open("x") as log:
            result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, seconds=time.time()-record["started"]); save()
        print(name, record["exit_code"], round(record["seconds"], 1), flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {record['log']}")
    save()
    try:
        run("tests", "run_galaxy_position_tests.py")
        registry_path = REPO/"configs/p10_phase_registry_v1.json"
        registry = load_registry(registry_path)
        inventory = {}
        for phase in ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006"):
            phase_root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
            manifest = phase_root/"p1_canonical/manifest.json"
            canonical = json.loads(manifest.read_text())
            paths = {"legacy_canonical": Path(canonical["parent"]), "lss": default_lss(registry, phase),
                     "raw_cutsky": Path(registry["path_templates"]["cutsky"].format(phase=phase))}
            inventory[phase] = {"legacy_p1_manifest": str(manifest), "manifest_sha256": digest(manifest), "catalogues": {}}
            for name, path in paths.items():
                with fitsio.FITS(path) as f:
                    inventory[phase]["catalogues"][name] = dict(
                        path=str(path), rows=f[1].get_nrows(), columns=f[1].get_colnames(),
                        bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        (root/"phase_catalogue_inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
        run("ph006_coordinate_audit", "validate_galaxy_position_mapping.py",
            "--phases", "ph006", "--output", root/"ph006_coordinate_audit")
        certificate = root/"qualification_ph006_v2.json"
        run("ph006_certificate", "certify_galaxy_position_scope.py",
            "--audit", root/"ph006_coordinate_audit/report.json",
            "--samples", root/"ph006_coordinate_audit", "--output", certificate)
        run("ph006_annotation_smoke", "smoke_galaxy_position_annotation.py",
            "--certificate", certificate, "--samples", root/"ph006_coordinate_audit",
            "--output", root/"ph006_smoke")
        phase = "ph002"
        phase_root = root/"replacement"/phase
        catalogues = phase_root/"catalogues"
        run("ph002_catalogues", "migrate_p10_galaxy_position_truth.py",
            "--phase", phase, "--qualification",
            REPO/"workflows/abacus_tweb/galaxy_position_qualification_allpop_v2_20261019.json",
            "--output-root", catalogues, "--execute")
        observed = catalogues/f"{phase}_observed_galaxy_realspace_v2_tweb.fits"
        run("ph002_p1", "p10_build_phase_index.py", "--phase", phase,
            "--observed", observed, "--out-dir", phase_root/"p1_canonical")
        run("ph002_replacement_audit", "audit_galaxy_truth_replacement.py", "--phase", phase,
            "--legacy-root", "/pscratch/sd/d/dkololgi/abacus/p10_multiphase",
            "--replacement-root", root/"replacement", "--observed", observed)
        status["completed"] = True
    except Exception as exc:
        status["error"] = str(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
