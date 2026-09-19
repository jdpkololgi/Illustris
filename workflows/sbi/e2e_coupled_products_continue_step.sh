#!/bin/bash
# Reviewed product-node continuation only. No allocation request or model fit.
set -uo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=16
cd "$1" || exit 1
python -m workflows.sbi.e2e_coupled_prepare_ops register --job-id "$SLURM_JOB_ID" --hours 4 || exit 1
python -m workflows.sbi.e2e_coupled_prepare_ops step-budget --job-id "$SLURM_JOB_ID" --memory-gib 472 --cpus 128 || exit 1
run_root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1
# 128logical CPUs/472GiB fit the measured487802MiB allocation. Observation
# workers peak below8GiB so24GiB retains ample headroom; native needs128GiB.
# The qualification tail waits for the verified bulk-node audit/statistics
# steps before taking over, with no overlapping publishers. Native013/016
# share existing per-phase locks with the bulk node's later queue entries.
# This wrapper owns ALL its steps;
# it does not return early and terminate independently attached workers.
pids=()
srun --exact -N1 -n1 -c64 --mem=288G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind targets --seconds 13800 \
  > "$run_root/targets_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c32 --mem=128G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_native_pool --threads 16 --seconds 13800 \
  --phases ph019 ph018 ph017 ph013 ph016 \
  > "$run_root/native_pool_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c16 --mem=24G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind observations --seconds 13800 \
  --phases ph019 ph018 ph017 ph016 ph015 ph014 ph013 ph012 ph024 ph023 ph022 ph021 ph020 \
  > "$run_root/observations_high_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c8 --mem=16G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_post_worker --kind conditions --seconds 13800 \
  > "$run_root/conditions_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c4 --mem=8G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_interface_worker --seconds 13800 \
  > "$run_root/interface_queue_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c4 --mem=8G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_qualification_tail --seconds 13800 \
  --audit-step 58550091.3 --normalization-step 58550091.4 \
  > "$run_root/qualification_tail_${SLURM_JOB_ID}.log" 2>&1 &
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
