#!/bin/bash
# A single approved compute step. No nested allocation, scheduler retries or training.
set -euo pipefail
: "${SLURM_STEP_ID:?}" "${SLURM_JOB_ID:?}"
mode=${1:?}; rev=${2:?}
case "$mode" in patch|repeat) ;; *) exit 2;; esac
r=/pscratch/sd/d/dkololgi/abacus
d="$r/lowz_${mode}_diagnostic_$SLURM_JOB_ID"
mkdir "$d" "$d/src"
exec >"$d/diagnostic.log" 2>&1
repo=/global/u2/d/dkololgi/TNG/Illustris
git -C "$repo" archive "$rev" shared workflows configs | tar -x -C "$d/src"
cd "$d/src"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$d/cache" CUDA_CACHE_PATH="$d/cache/cuda" NUMBA_CACHE_DIR="$d/cache/numba"
mkdir -p "$NUMBA_CACHE_DIR"
echo "mode=$mode source=$rev node=$(hostname) job=$SLURM_JOB_ID"
if [[ $mode == patch ]]; then
    /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -m workflows.abacus_tweb.lowz_graph_patch_index \
        ph002 "$r/lowz_graph_union_61662908" "$d/ph002"
    jq -e '.passed and .phase=="ph002" and (.checks|length)==2' "$d/ph002/PATCH_GEOMETRY_READY.json"
else
    py="$r/lowz_litept_preflight_61662989/venv/bin/python"
    export CUMM_CUDA_ARCH_LIST=8.0
    for arm in native deterministic_attention; do
        "$py" -u -m workflows.abacus_tweb.lowz_litept_repeat_probe "$arm" | tee "$d/$arm.log"
    done
    "$py" - "$d" "$rev" <<'PY'
import json,sys
from pathlib import Path
d=Path(sys.argv[1]); res={}
for arm in ("native","deterministic_attention"):
    records=[json.loads(s) for s in (d/(arm+".log")).read_text().splitlines() if s.startswith("{")]
    assert len(records)==len({r["core"] for r in records})==4 and all(r["mode"]==arm for r in records)
    res[arm]=records
assert [(r["core"],r["rows"]) for r in res["native"]]==[(r["core"],r["rows"]) for r in res["deterministic_attention"]]
(d/"REPEAT_DIAGNOSTIC_COMPLETE.json").write_text(json.dumps(dict(source=sys.argv[2],results=res)))
PY
fi
