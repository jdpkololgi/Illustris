#!/bin/bash
# One approved four-hour continuation. No fits, GPU work, or successor request.
set -uo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=16
cd "$1" || exit 1
python -m workflows.sbi.e2e_coupled_prepare_ops register --job-id "$SLURM_JOB_ID" --hours 4 || exit 1
python -m workflows.sbi.e2e_coupled_prepare_ops step-budget --job-id "$SLURM_JOB_ID" --memory-gib 464 --cpus 128 || exit 1
run_root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1
# 128 logical CPUs and464GiB total step reservations fit the CPU node. Each
# native lane has128GiB; its measured peaks approach the former96GiB allowance.
# Existing targets/other native lanes stay in the second allocation.
pids=()
srun --exact -N1 -n1 -c32 --mem=128G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_native_pool --threads 16 --seconds 13800 \
  --phases ph009 ph010 ph011 ph012 ph013 \
  > "$run_root/native_low_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c32 --mem=128G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_native_pool --threads 16 --seconds 13800 \
  --phases ph021 ph020 ph022 \
  > "$run_root/native_mid_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c32 --mem=128G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_native_pool --threads 16 --seconds 13800 \
  --phases ph024 ph023 ph014 ph015 ph016 \
  > "$run_root/native_high_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c16 --mem=48G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind observations --seconds 13800 \
  > "$run_root/observations_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c8 --mem=16G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind conditions --seconds 13800 \
  > "$run_root/conditions_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c4 --mem=8G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_audit_worker --seconds 13800 \
  > "$run_root/product_audit_queue_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c4 --mem=8G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_normalization_worker --seconds 13800 \
  > "$run_root/normalization_queue_${SLURM_JOB_ID}.log" 2>&1 &
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
