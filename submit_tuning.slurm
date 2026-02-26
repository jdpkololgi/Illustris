#!/bin/bash
set -euo pipefail

# DEPRECATION SHIM WARNING
echo "DEPRECATION WARNING: submit_tuning.slurm is a compatibility shim and may be removed in a future cycle. Use workflows entrypoint `workflows/jraph/submit_tuning.slurm` instead." >&2

# Compatibility wrapper for migrated launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/workflows/jraph/submit_tuning.slurm" "$@"
