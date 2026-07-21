#!/usr/bin/env bash
# Path 1 — Step B only: rebuild forFA0 with DESI imaging mask (required before fiberassign).
set -euo pipefail
PH000="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000"
cd "$PH000"
STAGE2="${PH000}/stage_2"
TARG_DIR="${STAGE2}/SecondGenMocks/AbacusSummitBGS_v2"
CUTSKY_LINK="${STAGE2}/cutsky_link/BGS/v0.1/z0.200"
CUTSKY_NAME="cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits"
RUN_ROOT="${RUN_ROOT:-${PH000}/stage_3/path1_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"

mkdir -p "${CUTSKY_LINK}"
ln -sf "/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/${CUTSKY_NAME}" \
  "${CUTSKY_LINK}/${CUTSKY_NAME}"

source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="/pscratch/sd/d/dkololgi/LSS/py:${PYTHONPATH:-}"

# Imaging mask: parallel brick reads from CFS (I/O bound). Default to node CPUs.
export IMAGING_NOBS_NPROC="${IMAGING_NOBS_NPROC:-${SLURM_CPUS_PER_TASK:-64}}"

python scripts/upstream_prepare_mocks_Y3_bright.py \
  --mockver ab_secondgen_cosmosim \
  --mockpath "${STAGE2}/cutsky_link/" \
  --realmin 0 --realmax 1 \
  --prog bright --rbandcut 19.5 \
  --apply_mask y --downsampling n \
  --base_output "${STAGE2}/" \
  --isProduction n \
  2>&1 | tee "${RUN_ROOT}/prepare.log"

ls -lh "${TARG_DIR}"/forFA0*.fits
echo "forFA ready under ${TARG_DIR}"
