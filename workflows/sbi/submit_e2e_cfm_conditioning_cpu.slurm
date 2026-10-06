#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3/evaluation_v1/logs/cpu_%A_%a.out
#SBATCH --error=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3/evaluation_v1/logs/cpu_%A_%a.err
set -euo pipefail
TASK_KIND=${1:?score or summary}
TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}
export CFM_SCORE_THREADS=8
EVAL_SOURCE=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3/evaluation_v1/source
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --export=ALL \
  bash "$EVAL_SOURCE/workflows/sbi/run_e2e_cfm_conditioning_extension.sh" \
  "$EVAL_SOURCE/workflows/sbi/e2e_cfm_conditioning_evaluation.py" worker --kind "$TASK_KIND" --index "$TASK_INDEX"
