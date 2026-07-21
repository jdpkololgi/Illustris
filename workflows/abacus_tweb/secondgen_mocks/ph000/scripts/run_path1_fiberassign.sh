#!/usr/bin/env bash
# Path 1 — Full fiberassign via LSS fa_multipass.py (Abacus BGS bright).
#
# Prerequisites:
#   forFA0.fits exists (run scripts/run_path1_prepare.sh first)
#   salloc with many CPUs (recommend --cpus-per-task=128, -t 24:00:00 or regular queue)
#   source desi_environment; LSS/py on PYTHONPATH
#
# Output: $FA_OUT/faruns/farun-pass*/fba-*.fits
# Then run scripts/install_fba_to_univ000.sh to layout for mkCat usepota=n

set -euo pipefail
PH000="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000"
cd "$PH000"

FORFA="${FORFA:-${PH000}/stage_2/SecondGenMocks/AbacusSummitBGS_v2/forFA0.fits}"
if [[ ! -f "$FORFA" ]]; then
  FORFA="${PH000}/stage_2/SecondGenMocks/AbacusSummitBGS_v2/forFA0_nomask.fits"
fi
if [[ ! -f "$FORFA" ]]; then
  echo "ERROR: forFA not found. Run: bash scripts/run_path1_prepare.sh" >&2
  exit 1
fi

FA_OUT="${FA_OUT:-${PH000}/stage_3/path1_fiberassign_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$FA_OUT"
TILES_OBS="/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/tiles-BRIGHT.fits"
NUMPROC="${NUMPROC:-64}"

source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="/pscratch/sd/d/dkololgi/LSS/py:${PYTHONPATH:-}"

echo "FORFA=$FORFA"
echo "FA_OUT=$FA_OUT"
echo "NUMPROC=$NUMPROC"

python /pscratch/sd/d/dkololgi/LSS/scripts/mock_tools/fa_multipass.py \
  --infn "$FORFA" \
  --outdir "$FA_OUT" \
  --program bright \
  --survey main \
  --tilesfn "$TILES_OBS" \
  --npass 4 \
  --numproc "$NUMPROC" \
  --steps tiles,sky,targ,fa \
  2>&1 | tee "${FA_OUT}/fa_multipass.log"

echo "Done. Next: FA_OUT=$FA_OUT bash scripts/install_fba_to_univ000.sh"
