#!/bin/bash
set -euo pipefail
r=/pscratch/sd/d/dkololgi/abacus
repo=/global/u2/d/dkololgi/TNG/Illustris
out=$r/lowz_oof_probe_${SLURM_JOB_ID:?}
root=$r/p12a_lowz_20261029_v1/p12_crossfit_contracts/omit_ph000
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE LD_PRELOAD
mkdir "$out"
git clone -q --no-checkout --shared "$repo" "$out/src"
cd "$out/src"
git sparse-checkout set --cone workflows shared configs
git checkout -q --detach "${1:?}"
cp "$repo/workflows/sbi/lowz_oof_probe.py" workflows/sbi/
manifest=$r/p12a_lowz_20261029_v1/LOWZ_TRAINING_PRODUCTS_COMPLETE.json
echo "24764983f5424593240469c68cee01222174d2af1e261efc8ef05b79083c45ee  $manifest" | sha256sum -c -
echo "$(jq -r .contracts.omit_ph000.sha256 "$manifest")  $root/TRAINING_LOADER_READY.json" | sha256sum -c -
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
srun -N1 -n1 -c32 --gpus=1 --exact --cpu-bind=cores "$py" -um workflows.abacus_tweb.p10_train_arm_a --contract-root "$root" --output-root "$out" --run-name probe --model unet --seed 42 --ordered-prefetch --epochs 4 --min-epochs 4 --disable-early-stopping --stop-after-updates 256 --checkpoint-every 256
srun -N1 -n1 -c32 --gpus=1 --exact --cpu-bind=cores "$py" -um workflows.sbi.lowz_oof_probe "$root" "$out"
