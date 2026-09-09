#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use the authorized interactive allocation}"
cd /global/u2/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
 --output="/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/domain_gate_20260909_${SLURM_JOB_ID}.log" \
 /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -m workflows.sbi.e2e_field_domain_gate
