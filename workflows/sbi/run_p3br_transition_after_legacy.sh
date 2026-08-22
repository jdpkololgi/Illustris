#!/usr/bin/env bash
# Wait for the legacy FAINT/P12 programme, then hand off to P3b-R production.
# This waiting process requests no resources.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
MT_RUN=${ROOT}/p12_and_multitracer_training
SUMMARY=${ROOT}/p12_oof_summaries
LOG_ROOT=${ROOT}/p3br_r1_p12_logs
NEXT=${REPO}/workflows/sbi/run_p3br_r1_p12_4gpu_interactive.sh
mkdir -p "${LOG_ROOT}"
LOG=${LOG_ROOT}/legacy_transition.log

legacy_products_complete() {
  [[ -f "${MT_RUN}/p10_bf_proxy_v1/unet_multitracer/seed_42/EPOCH15_FROZEN.json" ]] &&
  [[ -f "${MT_RUN}/p10_bf_null_v1/unet_multitracer/seed_42/EPOCH15_FROZEN.json" ]] &&
  [[ -f "${SUMMARY}/ph000/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY}/ph002/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY}/ph003/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY}/ph004/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY}/ph005/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY}/ph006/OOF_SUMMARY_COMPLETE.json" ]]
}

echo "$(date -u +%FT%TZ) transition_wait_start pid=$$" >> "${LOG}"
while ! legacy_products_complete; do
  sleep 60
done
echo "$(date -u +%FT%TZ) legacy_products_complete" >> "${LOG}"

# The old supervisor can write its final marker before its enclosing allocation
# has fully left Slurm.  Never overlap the integrated allocator with that job.
while [[ -n $(squeue -h -u "${USER}" -n p10mtp12 -o '%A') ]]; do
  sleep 30
done
echo "$(date -u +%FT%TZ) legacy_allocation_absent handoff" >> "${LOG}"
exec /bin/bash "${NEXT}"
