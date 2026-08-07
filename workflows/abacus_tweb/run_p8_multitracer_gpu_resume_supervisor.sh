#!/usr/bin/env bash
# Wait for the resumed global graph, then acquire the second allocation.
set -euo pipefail

root="/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1"
worker="/global/homes/d/dkololgi/TNG/Illustris/workflows/abacus_tweb/run_p8_multitracer_gpu_worker.sh"

while [[ ! -f "$root/MT_PHOTSYS_MARGINAL_CPU_PIPELINE_READY_FOR_RAPIDS" ]]; do
  sleep 30
done

exec salloc --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --constraint="gpu&hbm80g" --gpus=1 --qos=interactive \
  --time=02:00:00 --account=desi_g --immediate=600 \
  bash "$worker"
