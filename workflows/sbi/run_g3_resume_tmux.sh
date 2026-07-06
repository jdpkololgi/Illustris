#!/bin/bash
# Finish G3 union training to 7000 epochs from the last checkpoint (epoch 3749),
# 4x A100-80GB, interactive QOS, inside tmux (SSH-independent). Because the
# interactive wall cap is ~4h (~2700 epochs at ~5.2 s/epoch) and 3749->7000 needs
# ~3250, this LOOPS: each pass re-allocates and resumes from the atomic
# checkpoint; the pipeline writes its results file only when it reaches 7000 +
# eval, which is the loop's stop condition.
#   tmux new-session -d -s g3_resume 'bash ~/TNG/Illustris/workflows/sbi/run_g3_resume_tmux.sh'
# NOTE: the fallback sbatch 55441429 MUST stay held (no double-writer on the
# shared checkpoint dir).
set -uo pipefail

OUT=/pscratch/sd/d/dkololgi/abacus/sbi_runs/path1_wedge_flowjax_3d_linear_si_uniongraph
export TNG_SBI_CACHE_DIR=/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si_uniongraph
CHAINLOG=/pscratch/sd/d/dkololgi/logs/g3_resume_$(date +%Y%m%d_%H%M%S).log
MAX_ITERS=4
log() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAINLOG"; }

log "G3 resume-to-7000 loop started on $(hostname); OUT=$OUT"
for i in $(seq 1 $MAX_ITERS); do
  if ls "$OUT"/flowjax_sbi_results_seed_42_*.txt >/dev/null 2>&1; then
    log "results file present -> G3 training COMPLETE."; break
  fi
  log "pass $i/$MAX_ITERS: allocating 4x A100-80GB (4h) and resuming..."
  salloc --nodes=1 --gpus-per-node=4 --cpus-per-task=32 --constraint="gpu&hbm80g" \
         --qos=interactive --time=04:00:00 --account=desi_g \
    srun -n 1 bash -lc '
      set -euo pipefail
      unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
      source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cosmic_env
      export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
      cd /global/u2/d/dkololgi/TNG/Illustris
      python -c "import jax; print(\"jax devices:\", jax.devices())"
      python -u workflows/sbi/jraph_sbi_flowjax.py --epochs 7000 --seed 42 \
        --increment_mode linear --checkpoint_every 250 --resume --output_dir "'"$OUT"'"
    ' 2>&1 | tee -a "$CHAINLOG"
  log "pass $i returned (wall cap or completion); checking checkpoint/results next pass."
  sleep 15
done
if ls "$OUT"/flowjax_sbi_results_seed_42_*.txt >/dev/null 2>&1; then
  log "DONE. results: $(ls -t $OUT/flowjax_sbi_results_seed_42_*.txt | head -1)"
else
  log "STOPPED after $MAX_ITERS passes without results — inspect $CHAINLOG."
fi
