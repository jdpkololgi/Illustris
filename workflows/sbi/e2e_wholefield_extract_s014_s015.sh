#!/bin/bash
# Fixed verification/extraction of eight completed tasks; no generation/retry.
set -euo pipefail
segment=${1:?segment required}
[[ $segment =~ ^s[0-9]+$ ]] || exit 64
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_v1
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd "$root/source"
"$py" -m workflows.sbi.e2e_wholefield_fullcap_run start --segment "$segment"
"$py" "$root/extraction_v1/extract.py" run --phase ph016 --cap NGC --seed 17 --first 28
for first in 0 4 8 12 16 20 24; do
    "$py" "$root/extraction_v1/extract.py" run --phase ph016 --cap NGC --seed 29 --first "$first"
done
printf 'EXTRACTION_ALL_EIGHT_COMPLETE\n'
