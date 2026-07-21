#!/usr/bin/env bash
# Path 1 — mkCat: assignwdup (usepota=n) then joindspec + fulld + mkclusdat.
set -euo pipefail
PH000="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000"
cd "$PH000"
unset PYTHONPATH PYTHONHOME
export SCRATCH=/pscratch/sd/d/dkololgi
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT with Univ000/fa/MAIN from install_fba_to_univ000.sh}"
TARG_DIR="${PH000}/stage_2/SecondGenMocks/AbacusSummitBGS_v2"

if [[ ! -d "${RUN_ROOT}/Univ000/fa/MAIN" ]] || [[ -z "$(find "${RUN_ROOT}/Univ000/fa/MAIN" -name 'fba-*.fits' 2>/dev/null | head -1)" ]]; then
  echo "ERROR: No fba-*.fits under ${RUN_ROOT}/Univ000/fa/MAIN. Run install_fba_to_univ000.sh first." >&2
  exit 1
fi
export LSS_TILES_FITS="${LSS_TILES_FITS:-${RUN_ROOT}/tiles-BRIGHT-with-fba.fits}"
if [[ ! -f "$LSS_TILES_FITS" ]]; then
  echo "ERROR: Missing $LSS_TILES_FITS (created by install_fba_to_univ000.sh)." >&2
  exit 1
fi

# --- 1) COMBD + assignwdup (usepota=n) ---
$PY scripts/upstream_mkCat_SecondGen_amtl.py \
  --tracer BGS_BRIGHT --mockver ab_secondgen --mocknum 0 \
  --base_output "${RUN_ROOT}/" --outmd scratch \
  --targDir "${TARG_DIR}/" \
  --simName SecondGenMocks/AbacusSummit_v4_1 --survey DA2 \
  --specdata loa-v1 --dataversion v2 \
  --combd y --usepota n \
  --joindspec n --fulld n --add_gtl n --mkclusdat n \
  --compmd not_altmtl \
  2>&1 | tee "${RUN_ROOT}/mkCat_combd_assignwdup.log"

test -f "${RUN_ROOT}/fba0/datcomb_brightassignwdup.fits" || {
  echo "ERROR: assignwdup not created" >&2
  exit 1
}

# --- 2) joindspec + fulld + clustering ---
# Do NOT source desi_environment here: it prepends desiconda numpy to PYTHONPATH and
# breaks cosmic_env's $PY (ImportError: numpy._core._multiarray_umath).
unset PYTHONPATH PYTHONHOME
export PYTHONPATH="/pscratch/sd/d/dkololgi/LSS/py"
export DESI_ROOT_READONLY="${DESI_ROOT_READONLY:-/dvs_ro/cfs/cdirs/desi}"

$PY scripts/upstream_mkCat_SecondGen_amtl.py \
  --tracer BGS_BRIGHT --mockver ab_secondgen --mocknum 0 \
  --base_output "${RUN_ROOT}/" --outmd scratch \
  --targDir "${TARG_DIR}/" \
  --pota "${TARG_DIR}/mock0/pota-BRIGHT.fits" \
  --simName SecondGenMocks/AbacusSummit_v4_1 --survey DA2 \
  --specdata loa-v1 --dataversion v2 \
  --combd n --usepota y \
  --joindspec y --fulld y --add_gtl y --mkclusdat y \
  --compmd not_altmtl \
  2>&1 | tee "${RUN_ROOT}/mkCat_fulld.log"

echo "Full catalog: ${RUN_ROOT}/loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto.dat.fits"
