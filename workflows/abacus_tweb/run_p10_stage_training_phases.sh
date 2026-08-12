#!/usr/bin/env bash
set -euo pipefail

# Sequentially restore and verify the unsealed HPSS particle-B training phases.
# ph001 remains sealed and ph006 is online on CFS, so neither belongs here.

REPO_ROOT="${TNG_ILLUSTRIS_PROJECT_DIR:-/global/homes/d/dkololgi/TNG/Illustris}"
PYTHON="${P10_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}"
OUTPUT_ROOT="${P10_OUTPUT_ROOT:-/pscratch/sd/d/dkololgi/abacus/p10_multiphase}"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1

for phase in ph003 ph004 ph005; do
    result="${OUTPUT_ROOT}/${phase}_b_stage_result.json"
    log="${LOG_ROOT}/${phase}_b_stage.log"
    echo "[$(date -u +%FT%TZ)] starting ${phase}" >> "${LOG_ROOT}/training_phase_staging_supervisor.log"
    "${PYTHON}" -u workflows/abacus_tweb/p10_stage_particle_b.py \
        --phase "${phase}" \
        --mode stage \
        --out "${result}" \
        > "${log}" 2>&1
    echo "[$(date -u +%FT%TZ)] completed ${phase}: ${result}" >> "${LOG_ROOT}/training_phase_staging_supervisor.log"
done

echo "[$(date -u +%FT%TZ)] all training-phase B restores complete" >> "${LOG_ROOT}/training_phase_staging_supervisor.log"
