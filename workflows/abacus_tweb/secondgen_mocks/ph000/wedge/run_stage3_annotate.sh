#!/bin/bash
set -euxo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
module purge || true
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd /global/homes/d/dkololgi/TNG/Illustris
$PY workflows/abacus_tweb/annotate_cutsky_with_tweb_eigs.py \
  --cutsky /pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/staged_mock_stage3_science_for_annotate.fits \
  --output-dir /pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_7/ \
  --output-name staged_mock_stage3_postcollision_rs7_with_tweb_eigs_rs7_ngrid2048_thr0p2.fits \
  --overwrite
