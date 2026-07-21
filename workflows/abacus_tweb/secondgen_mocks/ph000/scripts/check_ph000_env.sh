#!/usr/bin/env bash
# Quick package check for ph000 DESI-aligned pipeline.
set -euo pipefail

COSMIC_PY="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python"
PH000="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000"

echo "=== cosmic_env (use this Python for mkCat + build_mock_bgs_maglim) ==="
unset PYTHONPATH PYTHONHOME
"$COSMIC_PY" -c "
import LSS; from LSS.main import cattools
import fitsio, numpy, desitarget
print('LSS:', LSS.__file__)
print('OK: LSS, cattools, fitsio, desitarget')
"

echo ""
echo "=== desi_environment (fiberassign binary + spec catalog paths) ==="
# shellcheck disable=SC1091
source /global/common/software/desi/desi_environment.sh main
which fiberassign
fiberassign --version 2>&1 | head -1
python -c "import fiberassign, desimodel, desitarget; print('OK: fiberassign, desimodel, desitarget')"
echo "Note: LSS is NOT in desi_environment by default. Use COSMIC_PY for mkCat."

echo ""
echo "=== ph000 stage-2 inputs ==="
for f in \
  "$PH000/stage_2/SecondGenMocks/AbacusSummitBGS_v2/forFA0_nomask.fits" \
  "$PH000/stage_2/SecondGenMocks/AbacusSummitBGS_v2/mock0/pota-BRIGHT.fits" \
  "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/tiles-BRIGHT.fits"
do
  if [[ -f "$f" ]]; then echo "OK  $f"; else echo "MISSING $f"; fi
done
