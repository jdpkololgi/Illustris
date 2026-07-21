#!/usr/bin/env bash
# Path 1 — mkCat clustering step only (requires BGS_BRIGHT_full_noveto.dat.fits from fulld).
set -euo pipefail
PH000="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000"
cd "$PH000"
unset PYTHONPATH PYTHONHOME
export SCRATCH=/pscratch/sd/d/dkololgi
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
TARG_DIR="${PH000}/stage_2/SecondGenMocks/AbacusSummitBGS_v2"
FULL_NOVETO="${RUN_ROOT}/loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto.dat.fits"

test -f "${FULL_NOVETO}" || {
  echo "ERROR: Missing ${FULL_NOVETO}; run run_path1_mkcat_fulld_only.sh first" >&2
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
  --joindspec n --fulld n --add_gtl n --mkclusdat y \
  --compmd not_altmtl \
  2>&1 | tee "${RUN_ROOT}/mkCat_mkclusdat.log"

echo "Clustering catalog: ${RUN_ROOT}/loa-v1/mock0/LSScats/BGS_BRIGHT_clustering.dat.fits"
