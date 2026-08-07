#!/usr/bin/env bash
# Resume the Proxy global Delaunay graph in a self-owned interactive allocation.
set -euo pipefail

repo="/global/homes/d/dkololgi/TNG/Illustris"
root="/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1"
logs="/pscratch/sd/d/dkololgi/logs"
python="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python"
product="bf_proxy_response_v1"
graph_dir="$root/graph/${product}_photsys_marginal/global"

cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1

for required in \
  "$root/catalogues/$product/CATALOGUE_COMPLETE" \
  "$root/fields/$product/FIELD_OVERLAY_COMPLETE" \
  "$root/selection/$product/MULTITRACER_SELECTION_COMPLETE"; do
  if [[ ! -f "$required" ]]; then
    echo "missing passed dependency marker: $required" >&2
    exit 1
  fi
done

"$python" -c '
import json
from pathlib import Path
path = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/catalogues/bf_proxy_response_v1/manifest.json")
manifest = json.loads(path.read_text())
audit = manifest["response"]["application_audit"]
assert manifest["pass"]
assert manifest["response"]["calibration_basis"] == "DESI LOA PHOTSYS"
assert audit["ambiguous_rows"] == 0
assert audit["overall_fallback_rows"] == manifest["counts"]["FAINT"]["rows"]
assert "mock has no PHOTSYS" in audit["mapping"]
'

mkdir -p "$graph_dir"
srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=256 \
  "$python" -u -m workflows.abacus_tweb.build_abacus_graph \
  --points-path "$root/catalogues/$product/points.npy" \
  --catalog-path "" \
  --no-apply-y1y5-filter \
  --no-exclude-invalid-box-index \
  --mode delaunay \
  --split-hemispheres \
  --output-dir "$graph_dir" \
  --output-prefix bf_proxy_delaunay \
  2>&1 | tee "$logs/p8_multitracer_delaunay_${SLURM_JOB_ID}.log"

for required in \
  "$graph_dir/bf_proxy_delaunay_metadata.json" \
  "$graph_dir/bf_proxy_delaunay_ngc_pairs.npy" \
  "$graph_dir/bf_proxy_delaunay_sgc_pairs.npy"; do
  if [[ ! -f "$required" ]]; then
    echo "global graph command returned without artifact: $required" >&2
    exit 1
  fi
done

printf 'job_id=%s\ncommit=%s\n' \
  "$SLURM_JOB_ID" "$(git rev-parse HEAD)" > "$root/MT_PHOTSYS_MARGINAL_CPU_PIPELINE_READY_FOR_RAPIDS"
