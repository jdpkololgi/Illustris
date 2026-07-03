#!/bin/bash
# G3 (rung b') — retrain the linear-SI NPE on the Delaunay ∪ radius(10 Mpc/h) UNION
# graph (roadmap v2, connectivity-axis gate). Identical architecture, features,
# targets, seed and splits as the SI baseline; ONLY the message-passing edge set
# changes (740k -> 1.99M undirected pairs). Compare vs
# path1_wedge_flowjax_3d_Bcorrected_linear_si on the shared test split.
#
# Submit with:  sbatch workflows/sbi/submit_sbi_flowjax_uniongraph.slurm
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --job-name=flowjax_g3_uniongraph
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/flowjax_g3_uniongraph_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/flowjax_g3_uniongraph_%j.err
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

# --- union-graph SI cache (linear increments) ---
export TNG_SBI_CACHE_DIR=/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si_uniongraph

OUT=/pscratch/sd/d/dkololgi/abacus/sbi_runs/path1_wedge_flowjax_3d_linear_si_uniongraph
mkdir -p "$OUT"

REPO=/global/u2/d/dkololgi/TNG/Illustris
PY="$(command -v python)"
echo "Host: $(hostname)  |  python: $PY"
"$PY" -c "import sys; print('exe:', sys.executable)"

srun -n 1 "$PY" -u "$REPO/workflows/sbi/jraph_sbi_flowjax.py" \
  --epochs 7000 \
  --seed 42 \
  --increment_mode linear \
  --checkpoint_every 250 \
  --resume \
  --output_dir "$OUT"
