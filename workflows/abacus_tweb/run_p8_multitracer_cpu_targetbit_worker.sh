#!/usr/bin/env bash
# Allocation-owned corrected Proxy build: catalogue/fields/selection then graph.
set -euo pipefail

repo="/global/homes/d/dkololgi/TNG/Illustris"
root="/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1"

cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This worker must run as the command owned by an interactive allocation" >&2
  exit 2
fi

bash workflows/abacus_tweb/run_p8_multitracer_proxy_repair_worker.sh \
  "$SLURM_JOB_ID"

if [[ ! -f "$root/MT_PHOTSYS_MARGINAL_PROXY_PRODUCTS_READY" ]]; then
  echo "corrected Proxy products did not pass their gate" >&2
  exit 1
fi

bash workflows/abacus_tweb/run_p8_multitracer_graph_cpu_worker.sh
