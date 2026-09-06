#!/bin/bash
#SBATCH --job-name=p12a_p1_rho
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --licenses=scratch,cfs
#SBATCH --export=NONE
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/p12a_ph001_density_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/p12a_ph001_density_%j.err

set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/usr/bin:/bin

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
TRUTH_ROOT=/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1
cd "$REPO"

timeout 120 "$PY" -c 'import abacusnbody,numpy; print("P12A_PH001_DENSITY_RUNTIME_OK")'
"$PY" -u -m workflows.sbi.p12a_authorized_truth guard \
  --stage density --truth-root "$TRUTH_ROOT"
if "$PY" -u -m workflows.sbi.p12a_authorized_truth status \
  --stage density --truth-root "$TRUTH_ROOT"; then
  echo "Authorized density stage is already complete and unchanged."
  exit 0
else
  status=$?
  [[ "$status" -eq 3 ]] || exit "$status"
fi

attempt="$TRUTH_ROOT/attempts/density/$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
mkdir -p "$attempt"
particle_marker=$("$PY" -m workflows.sbi.p12a_authorized_truth artifact \
  --stage particle_b --name p10_b_stage_marker --truth-root "$TRUTH_ROOT")
particle_staging_root=$(dirname "$(dirname "$particle_marker")")
density="$attempt/AbacusSummit_base_c000_ph001_z0.200_ngrid2048_ab10_tsc_counts.npy"
manifest="$attempt/AbacusSummit_base_c000_ph001_z0.200_ngrid2048_ab10_tsc_counts.manifest.json"
if [[ -e "$density" || -e "$manifest" ]]; then
  [[ -f "$density" && -f "$manifest" ]] || {
    echo "Refusing incomplete density attempt: $attempt" >&2
    exit 2
  }
  echo "Adopting completed density attempt after deep validation."
else
  "$PY" -u workflows/abacus_tweb/p10_build_density_field.py \
    --phase ph001 \
    --staging-root "$particle_staging_root" \
    --ngrid 2048 \
    --threads 64 \
    --output "$density" \
    --manifest "$manifest"
fi
"$PY" -u -m workflows.sbi.p12a_authorized_truth complete-density \
  --truth-root "$TRUTH_ROOT" --density "$density" --manifest "$manifest"
