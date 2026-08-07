#!/usr/bin/env bash
# Allocation-owned corrected Proxy pipeline: repair products, then global graph.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run inside an interactive salloc allocation" >&2
  exit 2
fi

repo="/global/homes/d/dkololgi/TNG/Illustris"
cd "$repo"

bash workflows/abacus_tweb/run_p8_multitracer_proxy_repair_worker.sh "$SLURM_JOB_ID"
bash workflows/abacus_tweb/run_p8_multitracer_graph_cpu_worker.sh
