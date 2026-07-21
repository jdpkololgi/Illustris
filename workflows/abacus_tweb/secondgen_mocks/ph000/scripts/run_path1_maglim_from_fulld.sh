#!/usr/bin/env bash
# Path 1 — LOA spec injection + DESI-parity mag-lim catalog (after fulld).
set -euo pipefail
PH000="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000"
cd "$PH000"
unset PYTHONPATH PYTHONHOME
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"

FULL="${RUN_ROOT}/loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto.dat.fits"
INJECTED="${RUN_ROOT}/loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto_loa_spec.fits"
MAGLIM="${RUN_ROOT}/mock_bgs_maglim.fits"

test -f "${FULL}" || { echo "ERROR: Missing ${FULL}" >&2; exit 1; }

$PY scripts/inject_loa_spec_from_zall.py \
  --input-fits "${FULL}" \
  --out-fits "${INJECTED}" \
  --overwrite

$PY scripts/build_mock_bgs_maglim_catalog.py \
  --input-fits "${INJECTED}" \
  --out-path "${MAGLIM}" \
  --overwrite

echo "Mag-lim catalog: ${MAGLIM}"
