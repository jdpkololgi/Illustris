#!/usr/bin/env bash
# Resume one frozen MT4/MT5 recovery run across interactive allocation walls.
# Usage: run_p8_multitracer_recovery_supervisor.sh unet_multitracer 0 [run_name]

set -eo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
set -u

MODEL=${1:?model required}
ROTATION=${2:?rotation required}
RUN_NAME=${3:-mt4_proxy_v1}
case "$MODEL" in
  unet_multitracer|graph_multitracer) ;;
  *) echo "unsupported multi-tracer model: $MODEL" >&2; exit 2 ;;
esac

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1
OUT_ROOT=$ROOT/models/recovery
OUT=$OUT_ROOT/$RUN_NAME/$MODEL/rotation_$ROTATION/seed_42
LOG=/pscratch/sd/d/dkololgi/logs/p8_${RUN_NAME}_${MODEL}_rot${ROTATION}_supervisor.log
EXPECTED_REVISION=$(git -C "$REPO" rev-parse HEAD)
MAX_ATTEMPTS=8

mkdir -p "$(dirname "$LOG")" "$OUT"
{
  echo "supervisor_start=$(date -u +%FT%TZ)"
  echo "repo=$REPO"
  echo "expected_revision=$EXPECTED_REVISION"
  echo "model=$MODEL rotation=$ROTATION run_name=$RUN_NAME"
} >> "$LOG"

for ATTEMPT in $(seq 1 "$MAX_ATTEMPTS"); do
  if [[ -f "$OUT/recovery_summary.json" ]]; then
    echo "complete_before_attempt=$ATTEMPT" >> "$LOG"
    exit 0
  fi
  CURRENT_REVISION=$(git -C "$REPO" rev-parse HEAD)
  if [[ "$CURRENT_REVISION" != "$EXPECTED_REVISION" ]]; then
    echo "revision_changed=$CURRENT_REVISION expected=$EXPECTED_REVISION" >> "$LOG"
    exit 3
  fi

  # Enforce the user's global two-interactive-allocation ceiling.  The first
  # run may share the machine with one independent interactive experiment.
  while true; do
    ACTIVE=$(squeue -h -u "$USER" -o '%j' 2>/dev/null | awk '$1 == "interactive" {n++} END {print n+0}')
    if [[ "$ACTIVE" -lt 2 ]]; then
      break
    fi
    echo "waiting_for_allocation_slot active=$ACTIVE at=$(date -u +%FT%TZ)" >> "$LOG"
    sleep 30
  done

  RESUME=()
  if [[ -f "$OUT/recovery_checkpoint.pt" ]]; then
    RESUME=(--resume)
  fi
  echo "attempt=$ATTEMPT resume=${#RESUME[@]} at=$(date -u +%FT%TZ)" >> "$LOG"
  set +e
  salloc \
    --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --constraint="gpu&hbm80g" --gpus=1 \
    --qos=interactive --time=04:00:00 --account=desi_g --immediate=600 \
    srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --cpu-bind=cores \
    "$PY" -u -m workflows.abacus_tweb.p8_train_patch_recovery \
      --model "$MODEL" --rotation "$ROTATION" --seed 42 \
      --epochs 20 --min-epochs 5 --patience 3 --min-delta 0.005 \
      --lr 0.002 --disable-early-stopping \
      --loss-log-every 25 --checkpoint-every 250 \
      --run-name "$RUN_NAME" --output-root "$OUT_ROOT" \
      "${RESUME[@]}" >> "$LOG" 2>&1
  STATUS=$?
  set -e
  echo "attempt=$ATTEMPT exit_status=$STATUS at=$(date -u +%FT%TZ)" >> "$LOG"

  if [[ -f "$OUT/recovery_summary.json" ]]; then
    echo "supervisor_complete=$(date -u +%FT%TZ)" >> "$LOG"
    exit 0
  fi
  if [[ ! -f "$OUT/recovery_checkpoint.pt" ]]; then
    echo "no_checkpoint_after_failed_attempt" >> "$LOG"
    exit "${STATUS:-4}"
  fi
  sleep 15
done

echo "maximum_attempts_exhausted=$MAX_ATTEMPTS" >> "$LOG"
exit 5
