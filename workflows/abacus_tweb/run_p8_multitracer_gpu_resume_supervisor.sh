#!/usr/bin/env bash
# Wait for the resumed global graph, then acquire the second allocation.
set -euo pipefail

root="/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1"
worker="/global/homes/d/dkololgi/TNG/Illustris/workflows/abacus_tweb/run_p8_multitracer_gpu_worker.sh"
handoff="$root/MT_PHOTSYS_MARGINAL_CPU_PIPELINE_READY_FOR_RAPIDS"
validation="$root/graph/bf_proxy_response_v1_photsys_marginal/global/global_graph_validation.json"

while [[ ! -f "$handoff" ]]; do
  sleep 30
done

# A file's presence alone is not a sufficient dependency gate. Require both the
# explicit handoff token and the validator's machine-readable PASS.
python="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python"
if ! grep -qx 'validation=PASS' "$handoff"; then
  echo "graph handoff lacks validation=PASS: $handoff" >&2
  exit 1
fi
"$python" -c '
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
assert report["pass"] is True
assert all(report["gates"].values())
' "$validation"

exec salloc --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --constraint="gpu&hbm80g" --gpus=1 --qos=interactive \
  --time=02:00:00 --account=desi_g --immediate=600 \
  bash "$worker"
