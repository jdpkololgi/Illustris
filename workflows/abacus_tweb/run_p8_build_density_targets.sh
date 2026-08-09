#!/usr/bin/env bash
# Persistent interactive CPU supervisor for P8.9 target-field construction.

set -eo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
OUT=/pscratch/sd/d/dkololgi/abacus/p8_density_phys_v1/targets/target_manifest.json
LOG=/pscratch/sd/d/dkololgi/logs/p8_build_density_targets_supervisor.log
EXPECTED_REVISION=${EXPECTED_REVISION:-$(git -C "$REPO" rev-parse HEAD)}
ACTUAL_REVISION=$(git -C "$REPO" rev-parse HEAD)

mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"
{
  echo "supervisor_start=$(date -u +%FT%TZ)"
  echo "repo=$REPO"
  echo "expected_revision=$EXPECTED_REVISION"
  echo "actual_revision=$ACTUAL_REVISION"
  echo "output=$OUT"
} >> "$LOG"

if [[ "$ACTUAL_REVISION" != "$EXPECTED_REVISION" ]]; then
  echo "revision_mismatch=true" >> "$LOG"
  exit 4
fi
if [[ -f "$OUT" ]]; then
  echo "output_exists_before_launch=true" >> "$LOG"
  exit 0
fi

ACTIVE=$(squeue -h -u "$USER" -o '%j' 2>/dev/null | awk '$1 == "interactive" {n++} END {print n+0}')
if [[ "$ACTIVE" -ge 2 ]]; then
  echo "allocation_limit_reached=$ACTIVE" >> "$LOG"
  exit 2
fi

set +e
salloc \
  --nodes=1 --ntasks=1 --cpus-per-task=64 \
  --constraint=cpu --qos=interactive --time=04:00:00 --account=desi \
  srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
  env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
  "$PY" -u -m workflows.abacus_tweb.p8_build_density_targets \
  >> "$LOG" 2>&1
STATUS=$?
set -e

echo "exit_status=$STATUS at=$(date -u +%FT%TZ)" >> "$LOG"
if [[ "$STATUS" -ne 0 ]]; then
  exit "$STATUS"
fi
if [[ ! -f "$OUT" ]]; then
  echo "missing_output_after_success=true" >> "$LOG"
  exit 3
fi
echo "supervisor_complete=$(date -u +%FT%TZ)" >> "$LOG"
