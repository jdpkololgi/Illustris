#!/bin/bash
# Production NPE (FlowJAX) training on the path1 fiberassign wedge subvolume.
# Full single node: 4x A100. Script shards train/val masks across devices (pmap).
# Pipeline validated interactively on 2026-06-16 (1-GPU smoke: 50 epochs, clean
# train/val NLL descent). 12 h walltime gives generous margin for the full
# 7000-epoch run plus compile and post-training posterior sampling/eval.
# gpu_regular caps at 2 days.
#
# Submit with:  sbatch workflows/sbi/submit_sbi_flowjax_wedge.slurm
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --job-name=flowjax_npe_path1_wedge_3d
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/flowjax_npe_path1_wedge_3d_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/flowjax_npe_path1_wedge_3d_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

set -euo pipefail
mkdir -p /pscratch/sd/d/dkololgi/logs

# --- environment hygiene (HPC user-site contamination guard) ---
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosmic_env

# --- JAX runtime flags ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# --- 3-d (softplus-increment) wedge cache for FlowJAX ---
# Resolves to processed_jraph_data_mc1e+09_v2_scaled_3_transformed_eig.pkl
export TNG_SBI_CACHE_DIR=/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d

OUT=/pscratch/sd/d/dkololgi/abacus/sbi_runs/path1_fiberassign_wedge_ra120_160_flowjax_3d

REPO=/global/u2/d/dkololgi/TNG/Illustris
PY="$(command -v python)"
echo "Host: $(hostname)  |  python: $PY"
"$PY" -c "import sys; print('exe:', sys.executable)"

srun -n 1 "$PY" -u "$REPO/workflows/sbi/jraph_sbi_flowjax.py" \
  --epochs 7000 \
  --seed 42 \
  --output_dir "$OUT"
