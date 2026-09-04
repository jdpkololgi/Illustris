#!/usr/bin/env bash
# Run one bounded P12-F3-D2 stage from an existing one-GPU interactive allocation.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing substantial D2 work outside a Slurm allocation" >&2
  exit 2
fi

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

REPO="${D2_SOURCE_ROOT:-/global/homes/d/dkololgi/TNG/Illustris}"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || {
  echo "D2_SOURCE_ROOT is not a Git worktree" >&2
  exit 3
}
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
CONFIG="$REPO/configs/p12f3_d2_diffusion_v1.json"
OUTPUT="${D2_OUTPUT_ROOT:-/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_d2_diffusion_v1}"
CONTRACT="$OUTPUT/D2_CONTRACT_FROZEN.json"
CAPACITY="$OUTPUT/D2_CAPACITY_SELECTION.json"
FINAL="$OUTPUT/D2_FINAL_SELECTION.json"
CONFIRMATION="$OUTPUT/D2_INTERNAL_CONFIRMATION.json"
SEED42_DECISION="$OUTPUT/D2_SEED42_PH006_DECISION.json"
SECOND_LICENSE="$OUTPUT/D2_SECOND_SEED_LICENSE.json"
MATCHED_REFERENCES="$OUTPUT/D2_MATCHED_REFERENCE_REPORTS.json"

cd "$REPO"
ACTION="${1:-}"
case "$ACTION" in
  test)
    "$PY" -m unittest tests.phase4.test_p12f3_d2 -v
    ;;
  contract)
    "$PY" -m workflows.sbi.p12f3_d2_contract \
      --config "$CONFIG" --output-root "$OUTPUT"
    ;;
  matched-references)
    "$PY" -m workflows.sbi.p12f3_d2_build_references \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --device cuda
    ;;
  transform-roundtrip)
    "$PY" -m workflows.sbi.p12f3_d2_roundtrip \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --device cuda
    ;;
  gpu-smoke)
    "$PY" -m workflows.sbi.p12f3_d2_gpu_smoke \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --device cuda
    ;;
  a0|a1)
    if [[ "$ACTION" == a0 ]]; then ARM=modern_base4; else ARM=modern_base8; fi
    RESUME=()
    [[ -f "$OUTPUT/training/$ARM/seed42_v1/checkpoint.pt" ]] && RESUME=(--resume)
    "$PY" -m workflows.sbi.p12f3_d2_train \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --arm "$ARM" --stage canary --device cuda --max-wall-seconds 6500 \
      "${RESUME[@]}"
    ;;
  select-capacity)
    "$PY" -m workflows.sbi.p12f3_d2_select \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --stage capacity
    ;;
  a2)
    ARM=modern_base8_attention
    RESUME=()
    [[ -f "$OUTPUT/training/$ARM/seed42_v1/checkpoint.pt" ]] && RESUME=(--resume)
    "$PY" -m workflows.sbi.p12f3_d2_train \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --arm "$ARM" --stage canary --capacity-selection-marker "$CAPACITY" \
      --device cuda --max-wall-seconds 6500 "${RESUME[@]}"
    ;;
  select-final)
    "$PY" -m workflows.sbi.p12f3_d2_select \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --stage final
    ;;
  confirm)
    "$PY" -m workflows.sbi.p12f3_d2_confirm \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --selection-marker "$FINAL" --device cuda
    ;;
  science)
    ARM="${2:?science action requires the frozen selected arm as argument}"
    RESUME=()
    [[ -f "$OUTPUT/training/$ARM/seed42_v1/checkpoint.pt" ]] && RESUME=(--resume)
    "$PY" -m workflows.sbi.p12f3_d2_train \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --arm "$ARM" --stage science --selection-marker "$FINAL" \
      --confirmation-marker "$CONFIRMATION" \
      --device cuda --max-wall-seconds 6500 "${RESUME[@]}"
    ;;
  replicate)
    ARM="${2:?replicate action requires the frozen selected arm as argument}"
    LICENSE="${3:?replicate action requires D2_SECOND_SEED_LICENSE.json}"
    RESUME=()
    [[ -f "$OUTPUT/training/$ARM/seed314159_v1/checkpoint.pt" ]] && RESUME=(--resume)
    "$PY" -m workflows.sbi.p12f3_d2_train \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --arm "$ARM" --stage science --seed-role replication \
      --selection-marker "$FINAL" --confirmation-marker "$CONFIRMATION" \
      --second-seed-license "$LICENSE" \
      --device cuda --max-wall-seconds 6500 "${RESUME[@]}"
    ;;
  export)
    ROLE="${2:?export requires primary or replication}"
    [[ "$ROLE" == primary || "$ROLE" == replication ]] || {
      echo "export role must be primary or replication" >&2
      exit 2
    }
    NFE="${3:?export requires NFE 50 or 100}"
    ARM="${4:?export requires frozen selected arm}"
    SEED=42
    EXTRA=()
    if [[ "$ROLE" == replication ]]; then
      SEED=314159
      EXTRA=(--second-seed-license "$SECOND_LICENSE")
    fi
    ARCHIVE="$OUTPUT/evaluation/seed${SEED}_v1/d2_${ARM}_nfe${NFE}/P12F_SAMPLE_ARCHIVE.json"
    RESUME=()
    [[ -e "${ARCHIVE%/*}/SAMPLE_ARCHIVE_PROGRESS.json" ]] && RESUME=(--resume)
    "$PY" -m workflows.sbi.p12f3_d2_export \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --selection-marker "$FINAL" --confirmation-marker "$CONFIRMATION" \
      --seed-role "$ROLE" --network-evaluations "$NFE" --device cuda \
      "${EXTRA[@]}" "${RESUME[@]}"
    ;;
  evaluate)
    ROLE="${2:?evaluate requires primary or replication}"
    [[ "$ROLE" == primary || "$ROLE" == replication ]] || {
      echo "evaluate role must be primary or replication" >&2
      exit 2
    }
    NFE="${3:?evaluate requires NFE 50 or 100}"
    ARM="${4:?evaluate requires frozen selected arm}"
    if [[ "$ROLE" == primary ]]; then SEED=42; else SEED=314159; fi
    ROOT="$OUTPUT/evaluation/seed${SEED}_v1/d2_${ARM}_nfe${NFE}"
    "$PY" -m workflows.sbi.p12f3_d2_evaluate \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --archive "$ROOT/P12F_SAMPLE_ARCHIVE.json" --output-dir "$ROOT/reports" \
      --matched-reference-marker "$MATCHED_REFERENCES" --device cuda
    ;;
  decide-seed)
    ROLE="${2:?decide-seed requires primary or replication}"
    [[ "$ROLE" == primary || "$ROLE" == replication ]] || {
      echo "decide role must be primary or replication" >&2
      exit 2
    }
    ARM="${3:?decide-seed requires frozen selected arm}"
    if [[ "$ROLE" == primary ]]; then SEED=42; else SEED=314159; fi
    "$PY" -m workflows.sbi.p12f3_d2_decide \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --stage seed --seed-role "$ROLE" \
      --matched-reference-marker "$MATCHED_REFERENCES" \
      --nfe50-evaluation "$OUTPUT/evaluation/seed${SEED}_v1/d2_${ARM}_nfe50/reports/D2_PH006_EVALUATION.json" \
      --nfe100-evaluation "$OUTPUT/evaluation/seed${SEED}_v1/d2_${ARM}_nfe100/reports/D2_PH006_EVALUATION.json"
    ;;
  stochastic-control)
    ARM="${2:?stochastic-control requires frozen selected arm}"
    LICENSE="$OUTPUT/D2_STOCHASTIC_CONTROL_LICENSE.json"
    ROOT="$OUTPUT/evaluation/seed42_v1/d2_${ARM}_nfe100_eta1_diagnostic"
    RESUME=()
    [[ -e "$ROOT/SAMPLE_ARCHIVE_PROGRESS.json" ]] && RESUME=(--resume)
    "$PY" -m workflows.sbi.p12f3_d2_export \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --selection-marker "$FINAL" --confirmation-marker "$CONFIRMATION" \
      --seed-role primary --network-evaluations 100 --sampler stochastic \
      --stochastic-control-license "$LICENSE" --device cuda "${RESUME[@]}"
    "$PY" -m workflows.sbi.p12f3_d2_evaluate \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --archive "$ROOT/P12F_SAMPLE_ARCHIVE.json" --output-dir "$ROOT/reports" \
      --matched-reference-marker "$MATCHED_REFERENCES" --device cuda
    ;;
  decide-combined)
    "$PY" -m workflows.sbi.p12f3_d2_decide \
      --config "$CONFIG" --output-root "$OUTPUT" --contract "$CONTRACT" \
      --stage combined \
      --primary-decision "$OUTPUT/D2_SEED42_PH006_DECISION.json" \
      --replication-decision "$OUTPUT/D2_SEED314159_PH006_DECISION.json"
    ;;
  *)
    echo "usage: $0 {test|contract|matched-references|transform-roundtrip|gpu-smoke|a0|a1|select-capacity|a2|select-final|confirm|science ARM|replicate ARM LICENSE|export ROLE NFE ARM|evaluate ROLE NFE ARM|decide-seed ROLE ARM|stochastic-control ARM|decide-combined}" >&2
    exit 2
    ;;
esac
