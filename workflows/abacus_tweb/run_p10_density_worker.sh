#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PHASE" >&2
  exit 2
fi

phase=$1
case "$phase" in
  ph002|ph003|ph004|ph005|ph006) ;;
  *) echo "refusing unregistered/development-ineligible phase: $phase" >&2; exit 2 ;;
esac

repo=/global/homes/d/dkololgi/TNG/Illustris
root=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
python=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
stem="AbacusSummit_base_c000_${phase}_z0.200_ngrid2048_ab10_tsc_counts"
output="$root/$phase/targets/density/$stem.npy"
manifest="$root/$phase/targets/density/$stem.manifest.json"

if [[ -s "$output" && -s "$manifest" ]]; then
  echo "[$phase] density already complete: $manifest"
  exit 0
fi
if [[ -e "$output" || -e "$manifest" ]]; then
  echo "[$phase] ambiguous partial density artifact; refusing: $output / $manifest" >&2
  exit 2
fi

mkdir -p "$root/logs"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
exec /usr/bin/time -v "$python" -u \
  "$repo/workflows/abacus_tweb/p10_build_density_field.py" \
  --phase "$phase" --ngrid 2048 --threads 64 \
  --output "$output" --manifest "$manifest"
