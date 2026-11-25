#!/bin/bash
# User-authorized bounded production progress: two independent GPU workers.
set -euo pipefail
revision=${1:?}; first=${2:?}; second=${3:?}
case "$first:$second" in ph004:ph005|ph007:ph008|ph009:ph010|ph011:ph014) ;; *) exit 2;; esac
base=/pscratch/sd/d/dkololgi/abacus
repo=/global/u2/d/dkololgi/TNG/Illustris
manifest=$base/p12a_lowz_20261029_v1/LOWZ_TRAINING_PRODUCTS_COMPLETE.json
echo "24764983f5424593240469c68cee01222174d2af1e261efc8ef05b79083c45ee  $manifest" | sha256sum -c -
stage=$base/p12_oof_pair_source_${SLURM_JOB_ID:?}
git clone -q --no-checkout --shared "$repo" "$stage"
cd "$stage"
git sparse-checkout set --cone workflows shared configs
git checkout -q --detach "$revision"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE LD_PRELOAD
worker() {
    local phase=$1 key=omit_$1 root out rc=0
    root=$(jq -er --arg key "$key" '.contracts[$key].path' "$manifest")
    echo "$(jq -er --arg key "$key" '.contracts[$key].sha256' "$manifest")  $root/TRAINING_LOADER_READY.json" | sha256sum -c -
    out=$base/p12a_lowz_oof_${phase}_${SLURM_JOB_ID}
    test ! -e "$out"
    srun --exclusive --exact -N1 -n1 -c32 --gpus=1 --mem=56G --cpu-bind=cores \
      /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python \
      -um workflows.abacus_tweb.p10_train_arm_a --contract-root "$root" \
      --output-root "$out" --run-name oof --model unet --seed 42 \
      --ordered-prefetch --epochs 4 --min-epochs 4 --disable-early-stopping \
      --checkpoint-every 1000 --max-runtime-seconds 12600 --validation-reserve-seconds 1800 \
      > "$base/p12-oof-${phase}-${SLURM_JOB_ID}.log" 2>&1 || rc=$?
    printf '%s exit=%s\n' "$phase" "$rc"
    # Trainer exit 75 is an intentional checkpointed budget stop.
    [[ $rc == 0 || $rc == 75 ]]
}
worker "$first" & one=$!
worker "$second" & two=$!
status=0
wait "$one" || status=1
wait "$two" || status=1
exit "$status"
