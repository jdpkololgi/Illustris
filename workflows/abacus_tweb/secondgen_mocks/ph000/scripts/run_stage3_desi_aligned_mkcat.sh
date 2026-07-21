#!/usr/bin/env bash
# Documented mkCat invocation for DESI-aligned stage-3 (ph000 / mock0).
#
# Does NOT run mkCat by default (COMBD+fulld can take hours). Use --dry-run-help
# to verify the upstream script is importable after loading desi_environment.
#
# Prerequisites:
#   source /global/common/software/desi/desi_environment.sh main
#   python -c "import LSS"   # must succeed before a real run
#   salloc …                 # interactive node recommended (see STAGE3_DESI_ALIGNMENT.md)
#
# Blocker: joindspec=y requires stage_3/fba0/datcomb_bright_assignwdup.fits, which is
# NOT produced when usepota=y alone. See STAGE3_DESI_ALIGNMENT.md § Blockers.

set -euo pipefail

PH000="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000"
MKCAT="${PH000}/scripts/upstream_mkCat_SecondGen_amtl.py"

TARG_DIR="${PH000}/stage_2/SecondGenMocks/AbacusSummitBGS_v2"
POTA="${TARG_DIR}/mock0/pota-BRIGHT.fits"
BASE_OUTPUT="${PH000}/stage_3"

# Matches logged 2026-05-15 run (COMBD-only)
CURRENT_FLAGS=(
  --tracer BGS_BRIGHT
  --mockver ab_secondgen
  --mocknum 0
  --base_output "${BASE_OUTPUT}/"
  --targDir "${TARG_DIR}/"
  --pota "${POTA}"
  --simName SecondGenMocks/AbacusSummit_v4_1
  --survey DA2
  --specdata loa-v1
  --dataversion v2
  --combd y
  --usepota y
  --joindspec n
  --fulld n
  --add_gtl n
  --mkclusdat n
)

# Target flags for DESI LOA parity (requires assignwdup + desi_environment LSS stack)
TARGET_FLAGS=(
  --tracer BGS_BRIGHT
  --mockver ab_secondgen
  --mocknum 0
  --base_output "${BASE_OUTPUT}/"
  --targDir "${TARG_DIR}/"
  --pota "${POTA}"
  --simName SecondGenMocks/AbacusSummit_v4_1
  --survey DA2
  --specdata loa-v1
  --dataversion v2
  --combd y
  --usepota y
  --joindspec y
  --fulld y
  --add_gtl y
  --mkclusdat y
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run-help | --print-target-cmd | --print-current-cmd]

  --dry-run-help       source desi_environment and run mkCat --help (quick check)
  --print-target-cmd   echo recommended DESI-aligned mkCat command (no execution)
  --print-current-cmd  echo last known COMBD-only command (no execution)

See: ${PH000}/STAGE3_DESI_ALIGNMENT.md
EOF
}

print_cmd() {
  local -n _flags=$1
  echo "# After: source /global/common/software/desi/desi_environment.sh main"
  printf 'python %s \\\n' "${MKCAT}"
  local i=0
  while [[ $i -lt ${#_flags[@]} ]]; do
    if [[ $((i + 2)) -lt ${#_flags[@]} ]]; then
      printf '  %s %s \\\n' "${_flags[$i]}" "${_flags[$((i + 1))]}"
    elif [[ $((i + 1)) -lt ${#_flags[@]} ]]; then
      printf '  %s %s\n' "${_flags[$i]}" "${_flags[$((i + 1))]}"
    else
      printf '  %s\n' "${_flags[$i]}"
    fi
    i=$((i + 2))
  done
}

case "${1:-}" in
  --dry-run-help)
    # shellcheck disable=SC1091
    source /global/common/software/desi/desi_environment.sh main
    python "${MKCAT}" --help
    ;;
  --print-target-cmd)
    print_cmd TARGET_FLAGS
    ;;
  --print-current-cmd)
    print_cmd CURRENT_FLAGS
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    echo "Unknown option: $1" >&2
    usage >&2
    exit 1
    ;;
esac
