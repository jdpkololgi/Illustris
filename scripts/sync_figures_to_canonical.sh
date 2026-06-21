#!/bin/bash
# Mirror finished figures from a run-specific output dir into the canonical,
# conference-agnostic figure root (shared.config_paths.CANONICAL_FIGURE_ROOT),
# under a run-named subfolder. Run dirs stay the source of truth during active
# experimentation (checkpoints/logs/plots live together); this just keeps a
# browsable, centralised index to pick figures from later, instead of figures
# being scattered across /pscratch/.../sbi_runs/<variant>/plots/.
#
# Usage: scripts/sync_figures_to_canonical.sh <src_dir> <run_name>
set -euo pipefail

SRC="${1:?usage: sync_figures_to_canonical.sh <src_dir> <run_name>}"
RUN_NAME="${2:?usage: sync_figures_to_canonical.sh <src_dir> <run_name>}"
CANONICAL_ROOT="${TNG_CANONICAL_FIGURE_ROOT:-/pscratch/sd/d/dkololgi/tng_illustris/figures}"
DEST="${CANONICAL_ROOT}/${RUN_NAME}"

mkdir -p "$DEST"
rsync -av --include='*.png' --include='*.pdf' --include='*.html' --include='*.svg' \
  --exclude='*' "$SRC"/ "$DEST"/
echo "Synced figures: $SRC -> $DEST"
