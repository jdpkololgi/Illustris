#!/usr/bin/env bash
#SBATCH -A desi_g
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -C gpu&hbm80g
#SBATCH --gpus-per-task=1
#SBATCH --licenses=scratch
#SBATCH -t 02:00:00
#SBATCH -J d2_native_support
#SBATCH -o /pscratch/sd/d/dkololgi/abacus/d2_native_%x_%j.out
#SBATCH -e /pscratch/sd/d/dkololgi/abacus/d2_native_%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
BASE=/global/u2/d/dkololgi/TNG/Illustris_d2_467f442
CODE=/global/homes/d/dkololgi/TNG/Illustris/workflows/sbi
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_d2_diffusion_v1/official_467f442_seed42_v1
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
AMENDMENT="$ROOT/D2_NATIVE_SUPPORT_AMENDMENT.json"
SMOKE="$ROOT/recovery_20260906/native_support_smoke_v1/NATIVE_SUPPORT_SMOKE_PASS.json"
cd "$BASE"
[[ "$(git rev-parse HEAD)" == 467f442c5c54864658fdfaf948335d6e11a647fe ]]
[[ -z "$(git status --porcelain)" ]]
ACTION="${1:?requires smoke, evaluate or decide}"
ROLE="${2:-primary}"
[[ "$ROLE" == primary || "$ROLE" == replication ]]
SEED=42
[[ "$ROLE" != replication ]] || SEED=314159
EVAL="$ROOT/evaluation/seed${SEED}_v1"
run_python() {
  srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 "$PY" "$@"
}
if [[ "$ACTION" != smoke ]]; then
  "$PY" "$CODE/p12f3_d2_native_support.py" --amendment "$AMENDMENT" --require-smoke "$SMOKE"
fi
case "$ACTION" in
  smoke)
    run_python "$CODE/p12f3_d2_evaluate_native_support.py" \
      --config "$BASE/configs/p12f3_d2_diffusion_v1.json" --output-root "$ROOT" \
      --archive "$EVAL/d2_modern_base4_nfe50/P12F_SAMPLE_ARCHIVE.json" \
      --support-amendment "$AMENDMENT" --support-smoke \
      --output-dir "${SMOKE%/*}" --device cuda
    ;;
  evaluate)
    NFE="${3:?requires 50 or 100}"
    [[ "$NFE" == 50 || "$NFE" == 100 ]]
    SAMPLER="${4:-deterministic}"
    [[ "$SAMPLER" == deterministic || "$SAMPLER" == stochastic ]]
    SUB="d2_modern_base4_nfe${NFE}"
    if [[ "$SAMPLER" == stochastic ]]; then
      [[ "$ROLE" == primary && "$NFE" == 100 ]]
      SUB="${SUB}_eta1_diagnostic"
    fi
    run_python "$CODE/p12f3_d2_evaluate_native_support.py" \
      --config "$BASE/configs/p12f3_d2_diffusion_v1.json" --output-root "$ROOT" \
      --archive "$EVAL/${SUB}/P12F_SAMPLE_ARCHIVE.json" \
      --support-amendment "$AMENDMENT" \
      --output-dir "$EVAL/${SUB}/reports_native_support_v1" --device cuda
    ;;
  decide)
    FIFTY="$EVAL/d2_modern_base4_nfe50/reports_native_support_v1/D2_PH006_EVALUATION.json"
    HUNDRED="$EVAL/d2_modern_base4_nfe100/reports_native_support_v1/D2_PH006_EVALUATION.json"
    "$PY" "$CODE/p12f3_d2_native_support.py" --amendment "$AMENDMENT" --verify-evaluations "$FIFTY" "$HUNDRED"
    run_python -m workflows.sbi.p12f3_d2_decide \
      --config "$BASE/configs/p12f3_d2_diffusion_v1.json" --output-root "$ROOT" \
      --stage seed --seed-role "$ROLE" --nfe50-evaluation "$FIFTY" --nfe100-evaluation "$HUNDRED"
    ;;
  *) echo "unknown action" >&2; exit 2 ;;
esac
