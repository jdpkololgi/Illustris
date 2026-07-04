#!/bin/bash
# G4-PROPER P1a — graph-construction control: the EXISTING attentional GraphNet +
# curated features + FlowJAX, with ONLY the edge set swapped to radius-only
# (14.78 Mpc = 10 Mpc/h). --epochs 3750 targets the exact matched-budget anchor:
# union control@3749 = 0.8041, Delaunay baseline (fully trained) = 0.7750.
# The pipeline runs its posterior-mean eval automatically after the loop.
set -euo pipefail

export OUT=/pscratch/sd/d/dkololgi/abacus/sbi_runs/path1_wedge_flowjax_3d_linear_si_radiusgraph
export TNG_SBI_CACHE_DIR=/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si_radiusgraph
LOG=/pscratch/sd/d/dkololgi/logs/g4_p1a_radius_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$OUT" /pscratch/sd/d/dkololgi/logs
echo "=== G4 P1a radius-only control: salloc 4xA100 4h at $(date); log $LOG ==="

salloc --nodes=1 --gpus-per-node=4 --cpus-per-task=32 --constraint="gpu&hbm80g" \
       --qos=interactive --time=04:00:00 --account=desi_g \
  srun -n 1 bash -lc '
    set -euo pipefail
    unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cosmic_env
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
    cd /global/u2/d/dkololgi/TNG/Illustris
    python -c "import jax; print(\"jax devices:\", jax.devices())"
    python -u workflows/sbi/jraph_sbi_flowjax.py --epochs 3750 --seed 42 \
      --increment_mode linear --checkpoint_every 250 --resume --output_dir "$OUT"
  ' 2>&1 | tee -a "$LOG"
echo "=== G4 P1a EXITED $(date) ==="
