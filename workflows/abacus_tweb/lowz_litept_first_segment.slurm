#!/bin/bash
#SBATCH --job-name=litept-lowz-s42
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
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/litept-lowz-s42-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?}"
r=/pscratch/sd/d/dkololgi/abacus
out="$r/lowz_litept_s42_${SLURM_JOB_ID}"
contract="$r/p12a_lowz_20261029_v1/training_contract_v1"
repo=/global/u2/d/dkololgi/TNG/Illustris
rev=fc0b407371dba9d164b29f103755fdbc6dc86b81
py="$r/lowz_litept_preflight_61662989/venv/bin/python"
mkdir "$out"
git clone --quiet --shared --no-checkout "$repo" "$out/src"
cd "$out/src"
git sparse-checkout init --cone
git sparse-checkout set configs shared workflows
git -c advice.detachedHead=false checkout -q --detach "$rev"
test -f configs/lowz_retraining_binding_20261118.json
printf '%s  %s\n' \
    63b3c61cd09d2ec6fe462df83cc274f40e4eb23747b4f297acc0d22da241641c "$contract/TRAINING_LOADER_READY.json" \
    9ff78c9b61aa1c18e937d5144e81bb6def91294540528e944aa9de74061e16e4 "$r/lowz_points_61662493/INDEX_READY.json" | sha256sum -c -
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 CUMM_CUDA_ARCH_LIST=8.0
export XDG_CACHE_HOME="$out/cache" CUDA_CACHE_PATH="$out/cache/cuda"
boot='import runpy,sys;sys.path.append("/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/lib/python3.11/site-packages");runpy.run_module("workflows.abacus_tweb.lowz_litept_train",run_name="__main__")'
echo "trainer=$rev job=$SLURM_JOB_ID fresh seed=42"
rc=0
srun -N1 -n1 -c32 --gpus=1 --cpu-bind=cores "$py" -u -c "$boot" \
    --model unet --seed 42 --epochs 4 --min-epochs 4 --ordered-prefetch \
    --disable-early-stopping --checkpoint-every 1000 --loss-log-every 25 \
    --max-runtime-seconds 84600 --validation-reserve-seconds 7200 \
    --contract-root "$contract" --output-root "$out" --run-name litept_s_native || rc=$?
dest="$out/litept_s_native/unet/seed_42"
if [[ $rc == 75 ]]; then
    # Only a durable, intentional budget stop counts as a successful segment.
    test -s "$dest/arm_a_checkpoint.pt"
    jq -e '.resume_command_required and .global_step > 0' "$dest/ALLOCATION_PAUSED.json"
    exit 0
fi
exit "$rc"
