#!/bin/bash
# Read-only --check before salloc; --run executes once inside the approved srun.
# This script never requests an allocation or resubmits a failed stage.
set -euo pipefail
r=/pscratch/sd/d/dkololgi/abacus
base="$r/p12a_lowz_20261029_v1/training_contract_v1"
campaign="$r/lowz_graph_adapters_61668072"
view="$campaign/corrected_graph_view"
ready="$base/TRAINING_LOADER_READY.json"
receipt="$campaign/GRAPH_LOADER_CHECKED.json"
mode=${1:?use --check or --run}
case "$mode" in --check|--run) ;; *) exit 2 ;; esac
if [[ "$mode" == --check ]]; then
    status=$(sacct -nXP -j 61668072 -o JobIDRaw,State,ExitCode)
    if [[ "$status" != '61668072|COMPLETED|0:0' ]]; then
        echo "CPU prerequisite is not complete: $status" >&2; exit 1
    fi
fi
jq -e --slurpfile ready "$ready" --arg base "$base" --arg view "$view" '
  ($ready[0].roles) as $r | (($r.training + $r.validation_and_selection)|sort) as $visible |
  $ready[0].pass and ($visible|unique|length)==23 and
  $r.validation_and_selection==["ph006","ph012","ph013"] and
  ($r.sealed_blind_test|sort)==["ph018","ph019"] and
  (($visible|index("ph018"))==null) and (($visible|index("ph019"))==null) and
  .passed and .base==$base and .overlay==$view and (.patches|keys)==$visible and
  (.parent_delaunay_context_boundary|keys)==$visible and
  all(.patches[]; (map(.cap)|sort)==[0,1] and all(.[]; .owned>0 and .nodes>=.owned and .edges>=0))
' "$receipt" >/dev/null
transform="$view/transforms/graph/graph_transform.json"
jq -e --slurpfile ready "$ready" '
  $ready[0].roles as $r | .pass and .fit_phases==$r.training and
  (.applied|keys)==(($r.training+$r.validation_and_selection)|sort)
' "$transform" >/dev/null
printf '%s  %s\n' "$(jq -er .graph_view_sha256 "$receipt")" "$view/GRAPH_VIEW_READY.json" \
    "$(jq -er .transform_sha256 "$receipt")" "$transform" | sha256sum -c -
if [[ "$mode" == --check ]]; then exit 0; fi
: "${SLURM_STEP_ID:?requires GPU job step}" "${SLURM_JOB_ID:?}"
repo=/global/u2/d/dkololgi/TNG/Illustris
rev=$(git -C "$repo" rev-parse --verify "${2:?pinned source}^{commit}")
out="$r/lowz_gpatch_gpu_${SLURM_JOB_ID}"
mkdir "$out"
exec >"$out/probe.log" 2>&1
git clone --quiet --shared --no-checkout "$repo" "$out/src"
cd "$out/src"
git sparse-checkout init --cone
git sparse-checkout set configs shared workflows
git -c advice.detachedHead=false checkout -q --detach "$rev"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
echo "commit=$rev job=$SLURM_JOB_ID host=$(hostname)"
exec /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u \
    -m workflows.abacus_tweb.lowz_gpatch_preflight "$campaign" "$out/measurement"
