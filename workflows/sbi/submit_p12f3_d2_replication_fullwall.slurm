#!/usr/bin/env bash
#SBATCH -A desi_g
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -C gpu
#SBATCH --gpus-per-task=1
#SBATCH --licenses=scratch
#SBATCH -t 06:00:00
#SBATCH -J p12f3_d2_replicate
#SBATCH -o /pscratch/sd/d/dkololgi/abacus/p12f3_d2_%x_%j.out
#SBATCH -e /pscratch/sd/d/dkololgi/abacus/p12f3_d2_%x_%j.err

# Frozen replication from initialization: six hours accommodates the extra
# 2,500 presentations absent from the primary continuation. Scientific budget,
# checkpoint choice and all guards stay unchanged; this wrapper accepts only
# the registered second-seed action and performs no automatic retry.
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REPO=/global/u2/d/dkololgi/TNG/Illustris_d2_467f442
OUTPUT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_d2_diffusion_v1/official_467f442_seed42_v1
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd "$REPO"
[[ "$(git rev-parse HEAD)" == 467f442c5c54864658fdfaf948335d6e11a647fe ]]
[[ -z "$(git status --porcelain)" ]]
[[ "$(sha256sum configs/p12f3_d2_diffusion_v1.json | cut -d ' ' -f 1)" == 3143ce1dfdb9546d3eb40413feab91bafa11b70718a2e4bb0ecb451080793533 ]]
EXTRA=()
SEED=42
case "${D2_ACTION:?D2_ACTION is required}" in
  "replicate modern_base4 $OUTPUT/D2_SECOND_SEED_LICENSE.json")
    SEED=314159
    EXTRA=(--seed-role replication --second-seed-license "$OUTPUT/D2_SECOND_SEED_LICENSE.json") ;;
  *) echo 'Unregistered training action' >&2; exit 2 ;;
esac
RESUME=()
[[ -f "$OUTPUT/training/modern_base4/seed${SEED}_v1/checkpoint.pt" ]] && RESUME=(--resume)
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" --gpus-per-task=1 \
  "$PY" -u -m workflows.sbi.p12f3_d2_train \
  --config "$REPO/configs/p12f3_d2_diffusion_v1.json" --output-root "$OUTPUT" \
  --contract "$OUTPUT/D2_CONTRACT_FROZEN.json" --arm modern_base4 --stage science \
  --selection-marker "$OUTPUT/D2_FINAL_SELECTION.json" \
  --confirmation-marker "$OUTPUT/D2_INTERNAL_CONFIRMATION.json" \
  --device cuda --max-wall-seconds 21000 "${RESUME[@]}" "${EXTRA[@]}"

