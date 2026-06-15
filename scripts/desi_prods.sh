#!/usr/bin/env bash
set -euo pipefail

echo "### DESI productions (best-effort)"
echo "This script is intentionally conservative: it tries to point you at canonical areas,"
echo "but does not assume a specific production naming map (it drifts)."

# Common project roots at NERSC (adjust if your site uses different paths)
roots=(
  "/global/cfs/cdirs/desi"
  "/global/cfs/cdirs/desi/spectro/redux"
  "/global/cfs/cdirs/desi/users"
)

for r in "${roots[@]}"; do
  if [[ -d "$r" ]]; then
    echo
    echo "Found: $r"
    if [[ "$r" == *"/redux" ]]; then
      echo "Top-level redux entries (sorted by mtime):"
      ls -1t "$r" 2>/dev/null | head -n 20 || true
    else
      ls -la "$r" 2>/dev/null | head -n 40 || true
    fi
  fi
done

echo
echo "Tip: If you know the target data product, search within a narrow subtree rather than 'find' over all of CFS."

