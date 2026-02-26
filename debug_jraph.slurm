#!/bin/bash
set -euo pipefail

# Compatibility wrapper for migrated launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/workflows/jraph/debug_jraph.slurm" "$@"
