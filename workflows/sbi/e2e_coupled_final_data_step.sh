#!/bin/bash
# Historical, never-launched pre-OOM wrapper.58552205 now fails its clean guard.
# Retained for provenance; the reviewed fresh-process recovery supersedes it.
# One bounded all-panel data completion pass after58552205 is terminal.
# No scientific fit, new allocation request or automatic successor here.
set -uo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=16
cd "$1" || exit 1
python -m workflows.sbi.e2e_coupled_prepare_ops register --job-id "$SLURM_JOB_ID" --hours 4 || exit 1
python -c 'from workflows.sbi.e2e_coupled_post_allocation import require_planned_terminal; require_planned_terminal("58552205")' || exit 1
python -m workflows.sbi.e2e_coupled_prepare_ops step-budget --job-id "$SLURM_JOB_ID" --memory-gib 472 --cpus 128 || exit 1
run_root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1
# Prior product-node termination guarantees its tail publisher is no longer
# active. The serialized tail also rechecks the named original audit/stat steps.
# Completed products are revalidated/skipped; native000/002/003 are already
# qualified and deliberately excluded from the non-legacy native builder.
pids=()
srun --exact -N1 -n1 -c64 --mem=288G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind targets --seconds 13800 \
  > "$run_root/targets_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c32 --mem=128G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_native_pool --threads 16 --seconds 13800 \
  --phases ph007 ph008 ph009 ph010 ph011 ph012 ph013 ph014 ph015 ph016 ph017 ph018 ph019 ph020 ph021 ph022 ph023 ph024 \
  > "$run_root/native_final_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c16 --mem=24G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind observations --seconds 13800 \
  > "$run_root/observations_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c8 --mem=16G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind conditions --seconds 13800 \
  > "$run_root/conditions_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c4 --mem=8G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_qualification_tail --seconds 13800 \
  --audit-step 58550091.3 --normalization-step 58550091.4 \
  > "$run_root/qualification_tail_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c4 --mem=8G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_interface_worker --seconds 13800 --publish-data-release \
  > "$run_root/interface_queue_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
result=0
for worker_pid in "${pids[@]}"; do
  wait "$worker_pid"; rc=$?
  if [[ "$rc" != 0 && "$rc" != 75 ]]; then result=1
  elif [[ "$rc" == 75 && "$result" == 0 ]]; then result=75
  fi
done
python -m workflows.sbi.e2e_coupled_prepare_ops accounting
exit "$result"
