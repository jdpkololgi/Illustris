#!/bin/bash
#SBATCH --job-name=p12a_blind
#SBATCH --account=desi_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=p12a_blind_%j.out
#SBATCH --error=p12a_blind_%j.err

set -euo pipefail
: "${P12A_CANDIDATE:?set the frozen P12-A candidate marker}"
: "${P12A_PLAN:?set the frozen four-core shard plan}"
: "${P12A_CONTEXT:?set the truth-free ph001 context archive}"
: "${P12A_QUALITY_THRESHOLDS:?set the frozen quality thresholds}"
: "${P12A_CHECKPOINT:?set the frozen FMPE checkpoint}"
: "${P12A_OUTPUT_ROOT:?set a new blind-output directory}"

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=16
PY=${P12A_PYTHON:-/global/cfs/cdirs/desi/users/dkololgi/conda/envs/cosmic_env_recovery_v4_20260901/bin/python}
REPO=/global/homes/d/dkololgi/TNG/Illustris
cd "$REPO"

timeout 90 "$PY" -c 'import torch,numpy; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
mkdir -p "$P12A_OUTPUT_ROOT"
srun --nodes=1 --ntasks=4 --cpus-per-task=16 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --export=ALL \
  "$PY" -u -m workflows.sbi.p12a_blind_array_worker \
  --candidate "$P12A_CANDIDATE" \
  --plan "$P12A_PLAN" \
  --context "$P12A_CONTEXT" \
  --quality-thresholds "$P12A_QUALITY_THRESHOLDS" \
  --checkpoint "$P12A_CHECKPOINT" \
  --output-root "$P12A_OUTPUT_ROOT" \
  --draws 512 \
  --seed 42 \
  --sample-chunk 2048
