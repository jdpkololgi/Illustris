#!/bin/bash
# Interactive 4-GPU FlowJAX NPE on the path1 wedge, designed to run inside tmux so
# an SSH drop never stalls it. Grabs a 4 h gpu_interactive allocation and trains.
#
# Resumable: jraph_sbi_flowjax.py now writes an atomic checkpoint every
# --checkpoint_every epochs and --resume continues from it. Re-run this same
# script in a fresh 4 h allocation to pick up where the last one stopped; the
# first run finds no checkpoint and starts from scratch. Checkpoint lives at
# $OUT/flowjax_sbi_checkpoint_seed_42.pkl.
set -euo pipefail

export OUT=/pscratch/sd/d/dkololgi/abacus/sbi_runs/path1_fiberassign_wedge_ra120_160_flowjax_3d_interactive
export TNG_SBI_CACHE_DIR=/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d
LOG=/pscratch/sd/d/dkololgi/logs/flowjax_npe_path1_wedge_3d_interactive_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$OUT" /pscratch/sd/d/dkololgi/logs
echo "$LOG" > /tmp/flowjax_interactive_log.txt   # so the watcher knows the path

echo "=== launching salloc (4xA100, gpu_interactive, 4h) at $(date) ==="
echo "=== log: $LOG ==="

# salloc runs this command on the login/head node; srun dispatches to the compute node.
# OUT and TNG_SBI_CACHE_DIR are exported above and propagate via srun --export=ALL (default).
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
      --checkpoint_every 250 --resume --output_dir "$OUT"
  ' 2>&1 | tee "$LOG"

echo "=== TRAINING WRAPPER EXITED at $(date) (rc=$?) ===" | tee -a "$LOG"
