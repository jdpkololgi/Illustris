#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3/evaluation_v1/logs/score_pack_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3/evaluation_v1/logs/score_pack_%j.err
set -euo pipefail
export CFM_SCORE_THREADS=8
EVAL_SOURCE=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3/evaluation_v1/source
# Resource-only packing: all numerical work remains in the immutable snapshot.
# Stop at the first unsuccessful task; never retry or overwrite a partial score.
for CFM_SETTING in 1 2 3 4 5 6; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --export=ALL \
    bash "$EVAL_SOURCE/workflows/sbi/run_e2e_cfm_conditioning_extension.sh" \
    "$EVAL_SOURCE/workflows/sbi/e2e_cfm_conditioning_evaluation.py" \
    worker --kind score --index "$CFM_SETTING"
done
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --export=ALL \
  bash "$EVAL_SOURCE/workflows/sbi/run_e2e_cfm_conditioning_extension.sh" \
  "$EVAL_SOURCE/workflows/sbi/e2e_cfm_conditioning_evaluation.py" \
  worker --kind summary --index 0
