#!/bin/bash
# First production continuation of the leakage-safe ph000-omitted encoder.
# Submit with an explicit committed source revision as argument 1.
#SBATCH --job-name=p12-oof-ph000
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
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/p12-oof-ph000-%j.log
set -euo pipefail
repo=/global/u2/d/dkololgi/TNG/Illustris
base=/pscratch/sd/d/dkololgi/abacus
out=$base/lowz_oof_probe_61925898
stage=$base/lowz_oof_ph000_source_${SLURM_JOB_ID:?}
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE LD_PRELOAD
echo "3cdadd1006c5a69835bedeee4cf2ba712b6a5ad13111b407041eb3329612f3a8  $out/probe/unet/seed_42/arm_a_checkpoint.pt" | sha256sum -c -
echo "7660daf251010fe0b8d87dd59dffd82be9f214e45017d6b39d05b3cd35cb6033  $out/SMOKE_ONLY_ph000.npz" | sha256sum -c -
git clone -q --no-checkout --shared "$repo" "$stage"
cd "$stage"
git sparse-checkout set --cone workflows shared configs
git checkout -q --detach "${1:?source revision required}"
exec srun -N1 -n1 -c32 --gpus=1 --cpu-bind=cores \
 /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python \
 -um workflows.abacus_tweb.p10_train_arm_a \
 --contract-root "$base/p12a_lowz_20261029_v1/p12_crossfit_contracts/omit_ph000" \
 --output-root "$out" --run-name probe --model unet --seed 42 \
 --ordered-prefetch --epochs 4 --min-epochs 4 --disable-early-stopping \
 --auto-resume --checkpoint-every 1000 --max-runtime-seconds 19800 \
 --validation-reserve-seconds 3600
