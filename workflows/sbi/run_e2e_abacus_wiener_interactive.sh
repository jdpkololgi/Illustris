#!/bin/bash
# One authorized fixed run; no retries or scientific decisions in the launcher.
set -euo pipefail
run_root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/abacus_wiener_20260924_v2
cd "$run_root/source"
set +e
salloc --job-name=e2e_wiener_v2 --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --constraint=cpu --qos=interactive --account=desi --time=02:00:00 \
  --licenses=scratch --immediate=600 \
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 \
  bash workflows/sbi/submit_e2e_abacus_wiener.slurm > "$run_root/interactive.log" 2>&1
result=$?
printf 'EXIT_CODE=%s\n' "$result" >> "$run_root/interactive.log"
exit "$result"
