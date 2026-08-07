#!/usr/bin/env bash
# Allocation-owned corrected Proxy pipeline: repair products, then global graph.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run inside an interactive salloc allocation" >&2
  exit 2
fi

repo="/global/homes/d/dkololgi/TNG/Illustris"
cd "$repo"

exec bash workflows/abacus_tweb/run_p8_multitracer_cpu_targetbit_worker.sh
