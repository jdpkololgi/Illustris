#!/usr/bin/env bash
# Execute from a separately approved 4-node CPU interactive allocation shell.
# One representative new R7 T-Web phase; no submission or automatic retry here.
set -euo pipefail
: "${SLURM_JOB_ID:?Approved four-node allocation required}"
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
root=/pscratch/sd/d/dkololgi/abacus/p12a_r7_20phase_20261027_v1
run="$root/ph014/preparation_${SLURM_JOB_ID}"
tweb="$root/ph014/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7"
# Preserve any previous full/partial attempt; do not mix rank products.
if [[ -e "$tweb" ]]; then
  printf '%s\n' "Existing target directory requires inspection: $tweb" >&2
  exit 1
fi
mkdir -p "$run"
exec > >(tee "$run/run.log") 2>&1
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --cpu-bind=cores --export=ALL \
  "$py" workflows/abacus_tweb/p10_expansion_preflight.py \
  --registry configs/p10_phase_registry_v3.json --adapt-density ph014 \
  --output "$run/ph014_density_adapter.json"
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --export=ALL \
  "$py" -u workflows/abacus_tweb/p10_run_tweb.py \
  --registry configs/p10_phase_registry_v3.json --phase ph014 \
  --density /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1/matter/ph014/counts_tsc_2048.npy \
  --density-manifest "$run/ph014_density_adapter.json" --output-dir "$tweb"
printf '%s\n' "T-Web build returned successfully; inspect $tweb/TWEB_COMPLETE.json before downstream adoption."
