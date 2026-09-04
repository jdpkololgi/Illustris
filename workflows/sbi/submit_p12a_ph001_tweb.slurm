#!/bin/bash
#SBATCH --job-name=p12a_p1_tweb
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --licenses=scratch
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/p12a_ph001_tweb_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/p12a_ph001_tweb_%j.err

# Exact registered 2048^3 CACTUS layout.  Sixteen ranks keep the largest
# shift/mpi4py pickle transpose below the 2 GiB count boundary.
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=${P12A_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
TRUTH_ROOT=${P12A_TRUTH_ROOT:-/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1}
cd "$REPO"

timeout 120 "$PY" -c 'import mpi4py,numpy; from shift import mpiutils; print("P12A_PH001_TWEB_RUNTIME_OK")'
"$PY" -u -m workflows.sbi.p12a_authorized_truth guard \
  --stage tweb --truth-root "$TRUTH_ROOT"
if "$PY" -u -m workflows.sbi.p12a_authorized_truth status \
  --stage tweb --truth-root "$TRUTH_ROOT"; then
  echo "Authorized T-web stage is already complete and unchanged."
  exit 0
else
  status=$?
  [[ "$status" -eq 3 ]] || exit "$status"
fi

density=$("$PY" -m workflows.sbi.p12a_authorized_truth artifact \
  --stage density --name density --truth-root "$TRUTH_ROOT")
manifest=$("$PY" -m workflows.sbi.p12a_authorized_truth artifact \
  --stage density --name p10_density_manifest --truth-root "$TRUTH_ROOT")
tweb_dir="$TRUTH_ROOT/attempts/tweb/$SLURM_JOB_ID"
mkdir -p "$tweb_dir"
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores \
  "$PY" -u workflows/abacus_tweb/p10_run_tweb.py \
  --phase ph001 \
  --density "$density" \
  --density-manifest "$manifest" \
  --output-dir "$tweb_dir"
"$PY" -u -m workflows.sbi.p12a_authorized_truth complete-tweb \
  --truth-root "$TRUTH_ROOT" --tweb-dir "$tweb_dir"
