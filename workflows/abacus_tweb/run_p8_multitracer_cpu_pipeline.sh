#!/usr/bin/env bash
# Resume-safe interactive CPU pipeline for P8 Bright+Faint products.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SLURM_JOB_ID" >&2
  exit 2
fi

job_id="$1"
repo="/global/homes/d/dkololgi/TNG/Illustris"
root="/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1"
logs="/pscratch/sd/d/dkololgi/logs"
python="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python"
oracle_marker="$root/catalogues/bf_oracle_assigned_v1/CATALOGUE_COMPLETE"
proxy_marker="$root/catalogues/bf_proxy_response_v1/CATALOGUE_COMPLETE"

cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1

while [[ ! -f "$oracle_marker" || ! -f "$proxy_marker" ]]; do
  if ! squeue -h -j "$job_id" | grep -q .; then
    echo "allocation $job_id ended before catalogue completion" >&2
    exit 1
  fi
  sleep 30
done

srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 --cpus-per-task=128 \
  "$python" -m workflows.abacus_tweb.p8_build_multitracer_fields --force \
  2>&1 | tee "$logs/p8_multitracer_fields_${job_id}.log"

srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 --cpus-per-task=128 \
  "$python" -m workflows.abacus_tweb.p8_refit_multitracer_selection --force \
  2>&1 | tee "$logs/p8_multitracer_selection_${job_id}.log"

printf 'job_id=%s\n' "$job_id" > "$root/MT_FIELDS_SELECTION_READY"
for product in bf_oracle_assigned_v1 bf_proxy_response_v1; do
  field_marker="$root/fields/$product/FIELD_OVERLAY_COMPLETE"
  selection_marker="$root/selection/$product/MULTITRACER_SELECTION_COMPLETE"
  if [[ ! -f "$field_marker" ]]; then
    echo "missing passed field marker: $field_marker" >&2
    exit 1
  fi
  if [[ ! -f "$selection_marker" ]]; then
    echo "missing passed selection marker: $selection_marker" >&2
    exit 1
  fi
done

graph_dir="$root/graph/bf_proxy_response_v1/global"
mkdir -p "$graph_dir"
srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 --cpus-per-task=256 \
  "$python" -m workflows.abacus_tweb.build_abacus_graph \
  --points-path "$root/catalogues/bf_proxy_response_v1/points.npy" \
  --catalog-path "" \
  --no-apply-y1y5-filter \
  --no-exclude-invalid-box-index \
  --mode delaunay \
  --split-hemispheres \
  --output-dir "$graph_dir" \
  --output-prefix bf_proxy_delaunay \
  2>&1 | tee "$logs/p8_multitracer_delaunay_${job_id}.log"

printf 'job_id=%s\n' "$job_id" > "$root/MT_CPU_PIPELINE_READY_FOR_RAPIDS"
