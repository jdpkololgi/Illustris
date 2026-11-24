#!/bin/bash
# Modes: train (array 1-3), development (three sequential phase exports).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --time=06:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/p12-parallel-%A_%a.log
set -euo pipefail
mode=${1:?mode required}
revision=${2:?committed revision required}
base=/pscratch/sd/d/dkololgi/abacus
repo=/global/u2/d/dkololgi/TNG/Illustris
manifest=$base/p12a_lowz_20261029_v1/LOWZ_TRAINING_PRODUCTS_COMPLETE.json
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE LD_PRELOAD
echo "24764983f5424593240469c68cee01222174d2af1e261efc8ef05b79083c45ee  $manifest" | sha256sum -c -
stage=$base/p12_parallel_source_${SLURM_JOB_ID:?}
git clone -q --no-checkout --shared "$repo" "$stage"
cd "$stage"
git sparse-checkout set --cone workflows shared configs
git checkout -q --detach "$revision"
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
if [[ $mode == train ]]; then
    case ${SLURM_ARRAY_TASK_ID:?} in
        1) phase=ph001;; 2) phase=ph002;; 3) phase=ph003;;
        *) exit 2;;
    esac
    key=omit_$phase
    root=$(jq -er --arg key "$key" '.contracts[$key].path' "$manifest")
    echo "$(jq -er --arg key "$key" '.contracts[$key].sha256' "$manifest")  $root/TRAINING_LOADER_READY.json" | sha256sum -c -
    out=$base/p12a_lowz_oof_${phase}_${SLURM_JOB_ID}
    test ! -e "$out"
    exec srun -N1 -n1 -c32 --gpus=1 --cpu-bind=cores "$py" \
      -um workflows.abacus_tweb.p10_train_arm_a --contract-root "$root" \
      --output-root "$out" --run-name oof --model unet --seed 42 \
      --ordered-prefetch --epochs 4 --min-epochs 4 --disable-early-stopping \
      --checkpoint-every 1000 --max-runtime-seconds 19800 --validation-reserve-seconds 3600
elif [[ $mode == development ]]; then
    root=$(jq -er '.contracts.full.path' "$manifest")
    echo "$(jq -er '.contracts.full.sha256' "$manifest")  $root/TRAINING_LOADER_READY.json" | sha256sum -c -
    ck=$base/p12a_lowz_full_seed42_61548250/full/unet/seed_42/arm_a_checkpoint.pt
    out=$base/p12a_lowz_development_summaries_${SLURM_JOB_ID}
    test ! -e "$out"
    for phase in ph006 ph012 ph013; do
        srun -N1 -n1 -c32 --gpus=1 --cpu-bind=cores "$py" \
          -um workflows.sbi.p12_export_unet_summaries --contract-root "$root" \
          --checkpoint "$ck" --phase "$phase" --output-root "$out" \
          --context-halo 48 --terminal-epochs 4
    done
else
    exit 2
fi
