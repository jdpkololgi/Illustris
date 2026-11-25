#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/oof-campaign-%A_%a.log
set -euo pipefail
base=/pscratch/sd/d/dkololgi/abacus
repo=/global/u2/d/dkololgi/TNG/Illustris
phase=$(printf 'ph%03d' "${SLURM_ARRAY_TASK_ID:?}")
stage=$base/oof_campaign_src_${SLURM_JOB_ID:?}
git clone -q --no-checkout --shared "$repo" "$stage"
cd "$stage"
git sparse-checkout set --cone workflows shared configs
git checkout -q --detach "${1:?revision}"
manifest=$base/p12a_lowz_20261029_v1/LOWZ_TRAINING_PRODUCTS_COMPLETE.json
echo "24764983f5424593240469c68cee01222174d2af1e261efc8ef05b79083c45ee  $manifest" | sha256sum -c -
root=$(jq -er --arg key "omit_$phase" '.contracts[$key].path' "$manifest")
echo "$(jq -er --arg key "omit_$phase" '.contracts[$key].sha256' "$manifest")  $root/TRAINING_LOADER_READY.json" | sha256sum -c -
name=oof
case $phase in
 ph000) out=$base/lowz_oof_probe_61925898; name=probe;;
 ph001|ph002|ph003|ph004|ph005|ph007|ph008|ph009|ph010|ph011|ph014)
    matches=("$base"/p12a_lowz_oof_${phase}_[0-9]*)
    [[ ${#matches[@]} == 1 && -d ${matches[0]} ]]
    out=${matches[0]};;
 *) out=$base/oof_campaign_20261125/$phase;;
esac
run=$out/$name/unet/seed_42
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
python_step=(srun -N1 -n1 -c32 --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -um)
publish_summary() {
    local dest=$base/oof_campaign_20261125/summaries
    local attempt=$dest/.attempts/${SLURM_JOB_ID}
    local receipt=OOF_SUMMARY_COMPLETE.json
    test -s "$run/ARM_A_TRAINING_COMPLETE.json"
    if [[ -e $dest/$phase ]]; then
        jq -e --arg p "$phase" '.pass and .phase==$p and .terminal_epochs==4' "$dest/$phase/$receipt" >/dev/null
        return
    fi
    # A partial attempt stays private. Array paths in the receipt remain valid.
    mkdir -p "$attempt"
    "${python_step[@]}" workflows.sbi.p12_export_unet_summaries --contract-root "$root" --phase "$phase" --checkpoint "$run/arm_a_checkpoint.pt" --context-halo 48 --terminal-epochs 4 --output-root "$attempt"
    jq -e --arg p "$phase" '.pass and .phase==$p and .rows>0 and .terminal_epochs==4' "$attempt/$phase/$receipt" >/dev/null
    ln -sT "$attempt/$phase" "$dest/$phase"
}
if [[ $2 == export ]]; then publish_summary; exit 0; fi
if [[ -s $run/ARM_A_TRAINING_COMPLETE.json ]]; then publish_summary; exit 0; fi
resume=()
if [[ $2 != fresh ]]; then
    test -s "$run/arm_a_checkpoint.pt"
    test -s "$run/ALLOCATION_PAUSED.json"
    resume=(--auto-resume)
else
    test ! -e "$out"
fi
rc=0
"${python_step[@]}" workflows.abacus_tweb.p10_train_arm_a --contract-root "$root" --output-root "$out" --run-name "$name" --model unet --seed 42 --ordered-prefetch --epochs 4 --min-epochs 4 --disable-early-stopping --checkpoint-every 1000 --max-runtime-seconds 84600 --validation-reserve-seconds 3600 "${resume[@]}" || rc=$?
if [[ $rc == 75 ]]; then test -s "$run/ALLOCATION_PAUSED.json"; exit 0; fi
if [[ $rc != 0 ]]; then exit "$rc"; fi
test -s "$run/ARM_A_TRAINING_COMPLETE.json"
# Retain the existing six-hour phase export budget when estimating spare time.
if (( ${SLURM_JOB_END_TIME:-0} - $(date +%s) >= 21600 )); then publish_summary; fi
