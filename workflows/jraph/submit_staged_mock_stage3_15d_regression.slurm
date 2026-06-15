#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 08:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --job-name=jraph_s3_15d_reg
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/jraph_staged_mock_stage3_15d_reg_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/jraph_staged_mock_stage3_15d_reg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dkololgi@lbl.gov

set -euo pipefail

REPO="/global/homes/d/dkololgi/TNG/Illustris"
CACHE="/pscratch/sd/d/dkololgi/abacus/sbi_caches/staged_mock_wedge_stage3_postcollision_rs7_sbi_cache_15d.pkl"
OUT="/pscratch/sd/d/dkololgi/abacus/jraph_runs/staged_mock_wedge_stage3_postcollision_rs7_15d"
PY="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python"

mkdir -p /pscratch/sd/d/dkololgi/logs "${OUT}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

cd "${REPO}"
echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Cache: ${CACHE}"
echo "Output: ${OUT}"

# 15-d regression: block-weighted per-channel MSE is built in (loss_w15); do not pass --heteroscedastic_15d.
srun --gpus-per-task=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  "${PY}" workflows/jraph/jraph_pipeline.py \
  --cache_path "${CACHE}" \
  --prediction_mode regression \
  --no_transformed_eig \
  --output_dir "${OUT}" \
  --seed 42 \
  --epochs 10000

echo "Finished at $(date)"
