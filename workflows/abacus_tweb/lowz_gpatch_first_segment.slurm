#!/bin/bash
#SBATCH --job-name=gpatch-lowz-s42
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=gpu&a100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/gpatch-lowz-s42-%j.log
set -euo pipefail
r=/pscratch/sd/d/dkololgi/abacus
repo=/global/u2/d/dkololgi/TNG/Illustris
rev=449c60da50e1d567aa8f007400c3fa1ee4391d35
campaign="$r/lowz_graph_adapters_61668072"
view="$campaign/corrected_graph_view"
measured="$r/lowz_gpatch_gpu_61683377/measurement"
gate="$measured/GPATCH_GPU_GATE.json"
git -C "$repo" cat-file -e "$rev^{commit}"
printf '%s  %s\n' \
    24730f7ce6684d148188e8398cf4964d4d78b82383155745a52841f8b1b76727 "$gate" \
    dae840dbbe3a8644cc13c731e7d5c9f7c77ab2238290e49a5a69d82afb44e6b0 "$measured/HOST_CONE_REAL_PATCH_GATE.json" \
    9fd12e51d1af7a5dcda1cee5d9ad60b91bf1f403bc1c5c96e17b5df5e221f0ed "$campaign/GRAPH_LOADER_CHECKED.json" \
    b00a8e39c1ca794e31777538c28659496291dd7c8f4483adc1727e8242bcd787 "$view/GRAPH_VIEW_READY.json" \
    1aa9bd10bb034a835638dca378fba61a7e23aace985041d7c024d449e421d71f "$view/transforms/graph/graph_transform.json" \
    63b3c61cd09d2ec6fe462df83cc274f40e4eb23747b4f297acc0d22da241641c "$view/TRAINING_LOADER_READY.json" \
    463a8a916d6f8da16c593b6f9f704ff2a42465455f1fff181d85b584d8d79596 "$view/transforms/target_scaler.json" | sha256sum -c -
jq -e '.passed and .host_extractor=="host_cone" and .source_step==256 and .seed==42 and .training_cores_per_epoch==346364 and
       (.validation_samples|length)==24 and ([.validation_samples[].rows]|add)==217200' "$gate" >/dev/null
if [[ $# == 1 && ${1:-} == --check ]]; then exit 0; fi
if [[ $# != 0 ]]; then echo 'Only --check or no arguments supported' >&2; exit 2; fi
: "${SLURM_JOB_ID:?requires batch allocation}"
out="$r/lowz_gpatch_s42_${SLURM_JOB_ID}"
mkdir "$out"
git clone --quiet --shared --no-checkout "$repo" "$out/src"
cd "$out/src"
git sparse-checkout init --cone
git sparse-checkout set configs shared workflows
git -c advice.detachedHead=false checkout -q --detach "$rev"
test "$(git rev-parse HEAD)" = "$rev"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
name=lowz_gpatch_host_cone_s42
echo "trainer=$rev job=$SLURM_JOB_ID fresh seed=42 host_cone=true gate=$gate"
rc=0
srun --exact -N1 -n1 -c32 --gpus=1 --cpu-bind=cores "$py" -u -c \
    'from workflows.abacus_tweb.lowz_gpatch_train import main; main(host_cone=True)' \
    --model graph --seed 42 --epochs 4 --min-epochs 4 --disable-early-stopping --lr 0.002 \
    --validation-group-cores 4 --checkpoint-every 256 --loss-log-every 16 \
    --max-runtime-seconds 84600 --validation-reserve-seconds 7200 \
    --contract-root "$view" --output-root "$out" --run-name "$name" || rc=$?
d="$out/$name/graph/seed_42"
if (( rc == 75 )); then
    test -s "$d/arm_a_checkpoint.pt"
    jq -e '.schema_version=="p10-arm-a-allocation-pause-v1" and .resume_command_required
           and .epoch_length==346364 and .global_step>0 and .cursor>=0 and .cursor<=.epoch_length
           and .global_step==((.epoch-1)*.epoch_length+.cursor)' "$d/ALLOCATION_PAUSED.json" >/dev/null
    echo "Budget stop: durable checkpoint; no automatic resubmission."
elif (( rc == 0 )); then
    test -s "$d/arm_a_checkpoint.pt"
    jq -e '.epochs_completed==4 and .global_steps==1385456' "$d/ARM_A_TRAINING_COMPLETE.json" >/dev/null
else
    exit "$rc"
fi
