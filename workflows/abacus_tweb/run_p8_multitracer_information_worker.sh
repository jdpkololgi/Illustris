#!/usr/bin/env bash
# Run the truth-free MT2 Bright+Faint information audit inside a CPU allocation.
set -euo pipefail

repo="/global/homes/d/dkololgi/TNG/Illustris"
root="/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1"
logs="/pscratch/sd/d/dkololgi/logs"
python="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python"

cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run inside an interactive CPU allocation" >&2
  exit 2
fi

srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=256 \
  env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
  "$python" -u -m workflows.abacus_tweb.p8_multitracer_information_audit \
  --product bf_proxy_response_v1 --rotation 0 --workers 128 \
  2>&1 | tee "$logs/p8_mt_information_proxy_rot0_${SLURM_JOB_ID}.log"

report="$root/diagnostics/bf_proxy_response_v1/information_audit.json"
marker="$root/diagnostics/bf_proxy_response_v1/MT2_INFORMATION_AUDIT_COMPLETE"
for required in "$report" "$marker"; do
  if [[ ! -s "$required" ]]; then
    echo "information audit returned without passed artifact: $required" >&2
    exit 1
  fi
done
