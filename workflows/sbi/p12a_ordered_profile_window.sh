#!/bin/bash
set -eu
cd /global/u2/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PROFILE_OUT=/pscratch/sd/d/dkololgi/abacus/p12a_ordered_profile_${SLURM_JOB_ID}
mkdir "$PROFILE_OUT"
exec srun -N1 -n1 -c32 --exact --gpus=1 --cpu-bind=cores timeout 3300 /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -m workflows.sbi.p12a_ordered_profile >"$PROFILE_OUT/profile.log" 2>&1
