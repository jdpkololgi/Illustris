#!/usr/bin/env bash
# Wait for the resumed global graph, then acquire the second allocation.
set -euo pipefail

root="/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1"
worker="/global/homes/d/dkololgi/TNG/Illustris/workflows/abacus_tweb/run_p8_multitracer_gpu_worker.sh"

while [[ ! -f "$root/MT_CPU_PIPELINE_READY_FOR_RAPIDS" ]]; do
  sleep 30
done

exec salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=32 \
  --constraint=gpu --qos=interactive --time=04:00:00 --account=desi \
  bash "$worker"
