#!/bin/bash
set -euo pipefail
r=/pscratch/sd/d/dkololgi/abacus
repo=/global/u2/d/dkololgi/TNG/Illustris
stage=${2:?staging path required}
if [[ $1 == prepare ]]; then
    cp "$repo/workflows/abacus_tweb/point_panel_diagnostic.py" "$stage/diagnostic.py"
else
    [[ $1 == run ]]
    out=$r/point_panel_diag_${SLURM_JOB_ID:?}
    mkdir "$out"
fi
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 CUMM_CUDA_ARCH_LIST=8.0
export XDG_CACHE_HOME="$stage/cache" CUDA_CACHE_PATH="$stage/cache/cuda"
for arm in graph litept; do
    rev=449c60da50e1d567aa8f007400c3fa1ee4391d35
    py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
    if [[ $arm == litept ]]; then rev=fc0b407371dba9d164b29f103755fdbc6dc86b81; py=$r/lowz_litept_preflight_61662989/venv/bin/python; fi
    if [[ $1 == prepare ]]; then
        git clone -q --no-checkout --shared "$repo" "$stage/src_$arm"
        git -C "$stage/src_$arm" sparse-checkout set --cone shared workflows configs
        git -C "$stage/src_$arm" checkout -q --detach "$rev"
    fi
    cd "$stage/src_$arm"
    test "$(git rev-parse HEAD)" = "$rev"
    git diff --quiet HEAD
    if [[ $1 == run ]]; then
        srun -N1 -n1 -c32 --exact --gpus=1 --cpu-bind=cores "$py" -u "$stage/diagnostic.py" "$arm" "$out"
    fi
done
