#!/bin/bash
set -uo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=16
cd "$1" || exit 1
python -m workflows.sbi.e2e_coupled_prepare_ops register --job-id "$SLURM_JOB_ID" --hours 4 || exit 1
run_root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1
srun --exact -N1 -n1 -c64 --mem=288G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind targets \
  > "$run_root/targets_${SLURM_JOB_ID}.log" 2>&1 &
target_pid=$!
srun --exact -N1 -n1 -c8 --mem=16G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind conditions \
  > "$run_root/conditions_${SLURM_JOB_ID}.log" 2>&1 &
condition_pid=$!
srun --exact -N1 -n1 -c16 --mem=48G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind observations --phases ph000 ph002 ph003 \
  > "$run_root/legacy_observations_${SLURM_JOB_ID}.log" 2>&1 &
observation_pid=$!
srun --exact -N1 -n1 -c8 --mem=48G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind legacy_adopt --phases ph002 ph003 \
  > "$run_root/legacy_adopt_${SLURM_JOB_ID}.log" 2>&1 &
adopt_pid=$!
wait "$target_pid"; target_rc=$?
wait "$condition_pid"; condition_rc=$?
wait "$observation_pid"; observation_rc=$?
wait "$adopt_pid"; adopt_rc=$?
python -m workflows.sbi.e2e_coupled_prepare_ops accounting
for rc in "$target_rc" "$condition_rc" "$observation_rc" "$adopt_rc"; do
  if [[ "$rc" != 0 && "$rc" != 75 ]]; then exit "$rc"; fi
done
if [[ "$target_rc" == 75 || "$condition_rc" == 75 || "$observation_rc" == 75 || "$adopt_rc" == 75 ]]; then exit 75; fi
