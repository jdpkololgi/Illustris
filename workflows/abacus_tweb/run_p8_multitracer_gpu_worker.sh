#!/usr/bin/env bash
# Runs inside a four-GPU interactive allocation; all compute is launched with srun.
set -euo pipefail

repo="/global/homes/d/dkololgi/TNG/Illustris"
root="/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1"
logs="/pscratch/sd/d/dkololgi/logs"
cosmic="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python"
rapids="/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn/bin/python"
product="bf_proxy_response_v1"
graph_dir="$root/graph/${product}_targetbit/global"
radius_dir="$root/graph/${product}_targetbit/radius"
adapter_dir="$root/graph/${product}_targetbit/adapter"

cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1

run_u() {
  local product_name="$1"
  local run_name="$2"
  srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=24 \
    --gpus-per-task=1 --cpu-bind=cores \
    env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
    "$cosmic" -u -m workflows.abacus_tweb.p8_train_multitracer_unet_patch \
    --product "$product_name" --rotation 0 --seed 42 \
    --run-name "$run_name" --steps 100 --eval-every 100 --loss-log-every 10
}

u_oracle_pid=""
u_proxy_pid=""
oracle_u_marker="$root/models/u_patch/bf_oracle_assigned_v1/rotation_0/seed_42/canary_steps100/MULTITRACER_U_PATCH_SCREEN_COMPLETE"
proxy_run="canary_targetbit_response_steps100"
proxy_u_marker="$root/models/u_patch/bf_proxy_response_v1/rotation_0/seed_42/$proxy_run/MULTITRACER_U_PATCH_SCREEN_COMPLETE"
if [[ -f "$oracle_u_marker" ]]; then
  echo "Reusing passed Oracle U-PATCH canary: $oracle_u_marker"
else
  { run_u bf_oracle_assigned_v1 canary_steps100 2>&1 \
      | tee "$logs/p8_mt_u_oracle_rot0_canary_${SLURM_JOB_ID}.log"; } &
  u_oracle_pid=$!
fi
if [[ -f "$proxy_u_marker" ]]; then
  echo "Reusing passed Proxy U-PATCH canary: $proxy_u_marker"
else
  { run_u bf_proxy_response_v1 "$proxy_run" 2>&1 \
      | tee "$logs/p8_mt_u_proxy_targetbit_rot0_canary_${SLURM_JOB_ID}.log"; } &
  u_proxy_pid=$!
fi

while [[ ! -f "$root/MT_TARGETBIT_CPU_PIPELINE_READY_FOR_RAPIDS" ]]; do
  cpu_job="$(sed -n 's/^job_id=//p' "$root/MT_FIELDS_SELECTION_READY")"
  if [[ -n "$cpu_job" ]] && ! squeue -h -j "$cpu_job" | grep -q .; then
    echo "CPU allocation $cpu_job ended before the global graph completed" >&2
    exit 1
  fi
  sleep 30
done

srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --gpus-per-task=1 --cpu-bind=cores \
  env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
  "$rapids" -c 'import cupy, cudf, cugraph; print(cupy.cuda.runtime.getDeviceCount())'

srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --gpus-per-task=1 --cpu-bind=cores \
  env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
  "$rapids" -u -m workflows.abacus_tweb.abacus_graph_features_cugraph \
  --metadata-path "$graph_dir/bf_proxy_delaunay_metadata.json" \
  --points-path "$root/catalogues/$product/points.npy" \
  --artifacts-dir "$graph_dir" --prefix bf_proxy_delaunay \
  --output-dir "$graph_dir" --output-prefix bf_proxy_delaunay_cugraph \
  2>&1 | tee "$logs/p8_mt_cugraph_${SLURM_JOB_ID}.log"

mkdir -p "$radius_dir"
srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
  env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
  "$cosmic" -u -m workflows.abacus_tweb.p2b_build_full_radius_union \
  --graph-dir "$graph_dir" --prefix bf_proxy_delaunay \
  --canonical-index "$root/catalogues/$product/catalogue_index.npz" \
  --out-dir "$radius_dir" --radius-mpc 14.78 \
  2>&1 | tee "$logs/p8_mt_radius_union_${SLURM_JOB_ID}.log"

mkdir -p "$adapter_dir"
srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
  env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
  "$cosmic" -u -m workflows.abacus_tweb.p8_build_multitracer_graph_adapter \
  --graph-dir "$graph_dir" --prefix bf_proxy_delaunay \
  --catalogue-index "$root/catalogues/$product/catalogue_index.npz" \
  --p2-manifest "$radius_dir/p2b_union_manifest.json" \
  --out-dir "$adapter_dir" \
  2>&1 | tee "$logs/p8_mt_graph_adapter_${SLURM_JOB_ID}.log"

srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
  env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
  "$cosmic" -u -m workflows.abacus_tweb.p8_prepare_multitracer_graph_features \
  --product "$product" --rotation 0 \
  2>&1 | tee "$logs/p8_mt_graph_features_rot0_${SLURM_JOB_ID}.log"

{ srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=24 \
    --gpus-per-task=1 --cpu-bind=cores \
    env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
    "$cosmic" -u -m workflows.abacus_tweb.p8_train_multitracer_graph_patch \
    --product "$product" --rotation 0 --seed 42 \
    --run-name "$proxy_run" --steps 100 --eval-every 100 --loss-log-every 10 \
    2>&1 | tee "$logs/p8_mt_g_proxy_rot0_canary_${SLURM_JOB_ID}.log"; } &
g_proxy_pid=$!

set +e
u_oracle_status=0
u_proxy_status=0
if [[ -n "$u_oracle_pid" ]]; then wait "$u_oracle_pid"; u_oracle_status=$?; fi
if [[ -n "$u_proxy_pid" ]]; then wait "$u_proxy_pid"; u_proxy_status=$?; fi
wait "$g_proxy_pid"; g_proxy_status=$?
set -e
if (( u_oracle_status != 0 || u_proxy_status != 0 || g_proxy_status != 0 )); then
  printf 'u_oracle=%d u_proxy=%d g_proxy=%d\n' \
    "$u_oracle_status" "$u_proxy_status" "$g_proxy_status" >&2
  exit 1
fi

printf 'allocation=%s\ncommit=%s\n' \
  "$SLURM_JOB_ID" "$(git rev-parse HEAD)" > "$root/MT_TARGETBIT_MODEL_CANARIES_COMPLETE"
