#!/usr/bin/env bash
# Run only after explicit user approval of this bounded CPU allocation.
set -euo pipefail
repo=/global/u2/d/dkololgi/TNG/Illustris
analysis_output=/pscratch/sd/d/dkololgi/abacus/tidal_statistics_20261006
if [[ -e "$analysis_output" ]]; then
  echo "Output already exists; choose a fresh run directory before launching." >&2
  exit 1
fi
mkdir -p "$analysis_output"
cp "$repo/workflows/tidal_statistics/abacus_eigenvalue_statistics.py" "$analysis_output/analysis.py"
sha256sum "$analysis_output/analysis.py" > "$analysis_output/source.sha256"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
salloc --nodes=1 --ntasks=1 --cpus-per-task=64 --constraint=cpu \
  --qos=interactive --account=desi --time=01:00:00 --licenses=scratch --immediate=600 \
  --job-name=tidal_pdfs_pk \
  srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
  /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u \
  "$analysis_output/analysis.py" --output "$analysis_output" --ngrid 256 512 --workers 16 \
  > "$analysis_output/run.log" 2>&1
