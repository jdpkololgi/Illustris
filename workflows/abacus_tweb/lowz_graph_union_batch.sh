#!/bin/bash
#SBATCH --job-name=lowz-point-unions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --time=02:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/lowz-point-unions-%j.log
set -euo pipefail
r=/pscratch/sd/d/dkololgi/abacus
repo=/global/u2/d/dkololgi/TNG/Illustris
out="$r/lowz_graph_unions_${SLURM_JOB_ID:?}"
mkdir "$out" "$out/src"
rev=$(git -C "$repo" rev-parse --verify "${1:?committed revision}^{commit}")
git -C "$repo" archive "$rev" shared workflows configs | tar -x -C "$out/src"
cd "$out/src"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
srun --exact -N1 -n1 -c64 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u - "$out" "$rev" <<'PY'
import json, subprocess, sys
from pathlib import Path
from workflows.abacus_tweb.lowz_point_loader import ROOT, read, roles, require
from workflows.abacus_tweb.lowz_graph_union_pilot import digest
out = Path(sys.argv[1])
r = out.parent
index = r / "lowz_points_61662493"
products = {}
for phase in ("ph002", *(p for p in roles(ROOT) if p != "ph002")):
    directory = r / "lowz_graph_union_61662908" if phase == "ph002" else out / phase
    if phase != "ph002":
        subprocess.run([sys.executable, "-u", "-m", "workflows.abacus_tweb.lowz_graph_union_pilot",
                        str(index), str(directory), "--phase", phase], check=True)
    rec = read(directory / "CORRECTED_UNION_READY.json")
    require(rec["phase"] == phase and rec["output_sha256"] == digest(directory / "p2b_union_manifest.json"),
            f"{phase}: invalid union receipt")
    products[phase] = dict(directory=str(directory), manifest_sha256=rec["output_sha256"])
    print(phase, "verified", flush=True)
require(len(products) == 23 and set(products) == set(roles(ROOT)), "incomplete union set")
(out / "ALL_UNIONS_READY.json").write_text(json.dumps(dict(source=sys.argv[2], phases=products, passed=True)))
PY
