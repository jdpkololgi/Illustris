#!/bin/bash
set -uo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=32
cd "$1" || exit 1
python -m workflows.sbi.e2e_coupled_prepare_ops register --job-id "$SLURM_JOB_ID" --hours 4 || exit 1
run_root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1
# No Cartesian processing starts until the fixed distance convention passes
# three independent training phases. Catalogue/particle source authority stays v1.
srun --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_coordinate_verify \
  > "$run_root/coordinates_${SLURM_JOB_ID}.log" 2>&1 || exit 1
srun --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_cpu_worker --kind pairing \
  > "$run_root/pairing_${SLURM_JOB_ID}.log" 2>&1 &
pair_pid=$!
srun --exact --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=128G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_cpu_worker --kind matter \
  > "$run_root/matter_${SLURM_JOB_ID}.log" 2>&1 &
matter_pid=$!
srun --exact --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_cpu_worker --kind observations \
  > "$run_root/observations_${SLURM_JOB_ID}.log" 2>&1 &
obs_pid=$!
wait "$pair_pid"; pair_rc=$?
wait "$matter_pid"; matter_rc=$?
wait "$obs_pid"; obs_rc=$?
python -m workflows.sbi.e2e_coupled_prepare_ops accounting
for rc in "$pair_rc" "$matter_rc" "$obs_rc"; do
  if [[ "$rc" != 0 && "$rc" != 75 ]]; then exit "$rc"; fi
done
if [[ "$pair_rc" == 75 || "$matter_rc" == 75 || "$obs_rc" == 75 ]]; then exit 75; fi
