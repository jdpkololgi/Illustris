#!/usr/bin/env bash
set -euo pipefail

echo "### Quota checks (best-effort)"
echo "Note: NERSC provides quota tooling like 'myquota' in many environments."

if command -v myquota >/dev/null 2>&1; then
  echo
  echo "Running: myquota"
  myquota || true
else
  echo
  echo "myquota not found on PATH. Skipping."
  echo "If on NERSC, try: module load myquota (or ask NERSC docs for current quota tool)."
fi

echo
echo "### Inode/space quick view (HOME + PSCRATCH if set)"
df -h "$HOME" || true
if [[ -n "${PSCRATCH:-}" ]]; then
  df -h "$PSCRATCH" || true
fi

