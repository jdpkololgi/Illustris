#!/bin/bash
# G3 (union graph) — interactive 4-GPU training in tmux, bypassing the batch queue.
# Same OUT as the HELD sbatch fallback (job 55441429): checkpoints are shared, and
# the hold guarantees no simultaneous writer. Re-run this script in a fresh 4h
# window to resume from the atomic checkpoint.
set -euo pipefail

export OUT=/pscratch/sd/d/dkololgi/abacus/sbi_runs/path1_wedge_flowjax_3d_linear_si_uniongraph
export TNG_SBI_CACHE_DIR=/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si_uniongraph
LOG=/pscratch/sd/d/dkololgi/logs/flowjax_g3_union_interactive_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$OUT" /pscratch/sd/d/dkololgi/logs
echo "=== G3 union interactive: salloc 4xA100 4h at $(date); log $LOG ==="

salloc --nodes=1 --gpus-per-node=4 --cpus-per-task=32 --constraint=gpu \
       --qos=interactive --time=04:00:00 --account=desi_g \
  srun -n 1 bash -lc '
    set -euo pipefail
    unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cosmic_env
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
    cd /global/u2/d/dkololgi/TNG/Illustris
    python -c "import jax; print(\"jax devices:\", jax.devices())"
    python -u workflows/sbi/jraph_sbi_flowjax.py --epochs 7000 --seed 42 \
      --increment_mode linear --checkpoint_every 250 --resume --output_dir "$OUT"
  ' 2>&1 | tee -a "$LOG"
echo "=== G3 union interactive EXITED $(date) ==="