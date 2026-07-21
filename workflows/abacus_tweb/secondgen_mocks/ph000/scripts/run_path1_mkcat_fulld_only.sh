#!/usr/bin/env bash
# Path 1 — mkCat stage 2 only (after datcomb_brightassignwdup.fits exists).
set -euo pipefail
PH000="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000"
cd "$PH000"
unset PYTHONPATH PYTHONHOME
export SCRATCH=/pscratch/sd/d/dkololgi
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
TARG_DIR="${PH000}/stage_2/SecondGenMocks/AbacusSummitBGS_v2"

test -f "${RUN_ROOT}/fba0/datcomb_brightassignwdup.fits" || {
  echo "ERROR: Run run_path1_mkcat.sh stage 1 first (assignwdup missing)" >&2
  exit 1
}

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
