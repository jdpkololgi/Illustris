#!/bin/bash
# Export-only smoke retry: preserve the existing training checkpoint.
set -euo pipefail
repo=/global/u2/d/dkololgi/TNG/Illustris
base=/pscratch/sd/d/dkololgi/abacus
out=$base/lowz_oof_probe_61925898
stage=$base/lowz_oof_export_${SLURM_JOB_ID:?}
root=$base/p12a_lowz_20261029_v1/p12_crossfit_contracts/omit_ph000
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE LD_PRELOAD
test ! -e "$out/SMOKE_ONLY_ph000.npz"
git clone -q --no-checkout --shared "$repo" "$stage"
cd "$stage"
git sparse-checkout set --cone workflows shared configs
git checkout -q --detach "${1:?source revision required}"
sha256sum "$out/probe/unet/seed_42/arm_a_checkpoint.pt"
srun -N1 -n1 -c32 --gpus=1 --exact --cpu-bind=cores \
 /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python \
 -um workflows.sbi.lowz_oof_probe "$root" "$out"
sha256sum "$out/probe/unet/seed_42/arm_a_checkpoint.pt" "$out/SMOKE_ONLY_ph000.npz"
