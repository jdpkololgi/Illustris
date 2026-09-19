#!/bin/bash
# Single two-hour preparation recovery: native first, then fresh FFT processes.
# No allocation requests, successor, numerical-kernel changes or scientific fits.
set -uo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=16
cd "$1" || exit 1
python -m workflows.sbi.e2e_coupled_prepare_ops register --job-id "$SLURM_JOB_ID" --hours 2 || exit 1
python -c 'from workflows.sbi.e2e_coupled_target_recovery import verify_predecessor; verify_predecessor(allow_running=True)' || exit 1
python -m workflows.sbi.e2e_coupled_prepare_ops step-budget --job-id "$SLURM_JOB_ID" --memory-gib 416 --cpus 72 || exit 1
run_root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1
deadline=$((SECONDS + 6600))
# No high-memory FFT while the native gridding lane is active.
srun --exact -N1 -n1 -c32 --mem=128G --cpu-bind=cores --time=00:45:00 \
  python -u -m workflows.sbi.e2e_coupled_native_pool --threads 16 --seconds 2400 \
  --phases ph007 ph008 ph009 ph010 ph011 ph012 ph013 ph014 ph015 ph016 ph017 ph018 ph019 ph020 ph021 ph022 ph023 ph024 \
  > "$run_root/native_recovery_${SLURM_JOB_ID}.log" 2>&1
rc=$?
if [[ "$rc" != 0 ]]; then exit "$rc"; fi
remaining=$((deadline - SECONDS))
if (( remaining < 3600 )); then exit 75; fi
pids=()
srun --exact -N1 -n1 -c64 --mem=400G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_fresh_targets --seconds "$remaining" \
  > "$run_root/targets_recovery_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
# Target writer .0 is already terminal. Other old workers may finish concurrently
# on their own node; per-phase native locks prevent duplicate gridding. New
# publication lanes wait until the entire old allocation has terminated.
python -c 'from workflows.sbi.e2e_coupled_target_recovery import wait_for_writers; wait_for_writers()' || exit 1
remaining=$((deadline - SECONDS))
if (( remaining < 1500 )); then
  wait "${pids[0]}"
  exit 75
fi
srun --exact -N1 -n1 -c4 --mem=8G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_audit_worker --seconds "$remaining" \
  > "$run_root/audit_recovery_${SLURM_JOB_ID}.log" 2>&1 &
pids+=("$!")
srun --exact -N1 -n1 -c4 --mem=8G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_interface_worker --seconds "$remaining" --publish-data-release \
  > "$run_root/interface_recovery_${SLURM_JOB_ID}.log" 2>&1 &
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
