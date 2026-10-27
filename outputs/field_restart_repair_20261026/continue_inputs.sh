#!/bin/bash
set -uo pipefail
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PATH="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH"
timeout "${1:-1200}" python -m workflows.sbi.p12a_prepare_visible_responses --workers 4
rc=$?
printf 'CONTINUATION_EXIT_CODE inputs=%s\n' "$rc"
exit "$rc"
