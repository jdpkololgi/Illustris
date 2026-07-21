#!/usr/bin/env bash
# Run LOA-aligned components B → C → D → E for ph000 (mock0).
#
# B: prepare_mocks with --apply_mask y (DESI imaging footprint)
# C: getpota + mkCat COMBD (fibre potentials + collision strip)
# D: inject_loa_spec_from_zall (LOA marginal ZWARN/DELTACHI2/SPECTYPE)
# E: build_mock_bgs_maglim_catalog (same cuts as DESI GraphWeb catalog)
#
# Prerequisites:
#   salloc on a compute node (prepare/getpota are heavy)
#   bash scripts/check_ph000_env.sh
#
# Usage:
#   bash scripts/run_loa_BCDE.sh              # all steps
#   bash scripts/run_loa_BCDE.sh --from C    # skip B (reuse forFA if already masked)
#   bash scripts/run_loa_BCDE.sh --from D    # only inject + maglim

set -euo pipefail

PH000="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000"
cd "$PH000"

unset PYTHONPATH PYTHONHOME
export SCRATCH=/pscratch/sd/d/dkololgi
export DESI_ROOT_READONLY="${DESI_ROOT_READONLY:-/dvs_ro/cfs/cdirs/desi}"
PY="${PY:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}"
mkdir -p "$SCRATCH/rantiles"

RUN_ROOT="${RUN_ROOT:-${PH000}/stage_3/loa_BCDE_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"
echo "RUN_ROOT=$RUN_ROOT" | tee "$RUN_ROOT/run_root.txt"

STAGE2="${PH000}/stage_2"
TARG_DIR="${STAGE2}/SecondGenMocks/AbacusSummitBGS_v2"
CUTSKY_LINK="${STAGE2}/cutsky_link/BGS/v0.1/z0.200"
CUTSKY_NAME="cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits"
FORFA="${TARG_DIR}/forFA0.fits"
POTA="${TARG_DIR}/mock0/pota-BRIGHT.fits"
DATCOMB="${RUN_ROOT}/fba0/datcomb_brightwdup.fits"
INJECTED="${RUN_ROOT}/fba0/datcomb_brightwdup_loa_spec.fits"
MAGLIM="${RUN_ROOT}/mock_bgs_maglim_bright_galaxy_zwarn0_dchi2ge25.fits"

FROM_STEP="B"
if [[ "${1:-}" == "--from" ]]; then
  FROM_STEP="${2:-B}"
fi

run_B() {
  echo "=== B: prepare_mocks with apply_mask ==="
  mkdir -p "${CUTSKY_LINK}"
  ln -sf "/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/${CUTSKY_NAME}" \
    "${CUTSKY_LINK}/${CUTSKY_NAME}"

  rm -f "${TARG_DIR}/forFA0.fits" "${TARG_DIR}/forFA0_nomask.fits"

  # prepare uses LSS + desitarget imaging masks; needs desi_environment for bitmask I/O
  source /global/common/software/desi/desi_environment.sh main
  export PYTHONPATH="/pscratch/sd/d/dkololgi/LSS/py:${PYTHONPATH:-}"
  python scripts/upstream_prepare_mocks_Y3_bright.py \
    --mockver ab_secondgen_cosmosim \
    --mockpath "${STAGE2}/cutsky_link/" \
    --realmin 0 --realmax 1 \
    --prog bright --rbandcut 19.5 \
    --apply_mask y --downsampling n \
    --base_output "${STAGE2}/" \
    --isProduction n \
    2>&1 | tee "$RUN_ROOT/prepare_B.log"
  # forFA0.fits (masked) or forFA0_nomask.fits if mask fails — check log
  if [[ -f "${TARG_DIR}/forFA0.fits" ]]; then
    FORFA="${TARG_DIR}/forFA0.fits"
  else
    FORFA="${TARG_DIR}/forFA0_nomask.fits"
  fi
  echo "FORFA=$FORFA" >> "$RUN_ROOT/run_root.txt"
}

run_C() {
  echo "=== C: getpota + mkCat COMBD ==="
  source /global/common/software/desi/desi_environment.sh main
  export PYTHONPATH="/pscratch/sd/d/dkololgi/LSS/py:${PYTHONPATH:-}"
  python scripts/upstream_getpotaDA2_mock.py \
    --mock ab2ndgen --mock_version BGS_v2 --prog BRIGHT \
    --realization 0 \
    --base_output "${STAGE2}/" \
    2>&1 | tee "$RUN_ROOT/getpota_C.log"

  mkdir -p "${RUN_ROOT}/fba0"
  $PY scripts/upstream_mkCat_SecondGen_amtl.py \
    --tracer BGS_BRIGHT --mockver ab_secondgen --mocknum 0 \
    --base_output "${RUN_ROOT}/" --outmd scratch \
    --targDir "${TARG_DIR}/" --pota "${POTA}" \
    --simName SecondGenMocks/AbacusSummit_v4_1 --survey DA2 \
    --specdata loa-v1 --dataversion v2 \
    --combd y --usepota y \
    --joindspec n --fulld n --add_gtl n --mkclusdat n \
    --compmd not_altmtl \
    2>&1 | tee "$RUN_ROOT/mkCat_combd_C.log"
}

run_D() {
  echo "=== D: LOA-calibrated spec injection ==="
  $PY scripts/inject_loa_spec_from_zall.py \
    --input-fits "$DATCOMB" \
    --out-fits "$INJECTED" \
    --overwrite \
    2>&1 | tee "$RUN_ROOT/inject_D.log"
}

run_E() {
  echo "=== E: DESI parity mag-lim catalog ==="
  $PY scripts/build_mock_bgs_maglim_catalog.py \
    --input-fits "$INJECTED" \
    --out-path "$MAGLIM" \
    --overwrite \
    2>&1 | tee "$RUN_ROOT/maglim_E.log"
  echo "GraphWeb wedge export should use: $MAGLIM"
}

case "$FROM_STEP" in
  B) run_B ;&
  C) run_C ;&
  D) run_D ;&
  E) run_E ;;
  *)
    echo "Unknown --from step: $FROM_STEP (use B, C, D, or E)" >&2
    exit 1
    ;;
esac

echo "Finished. Catalogue for wedge/GNN: $MAGLIM"
