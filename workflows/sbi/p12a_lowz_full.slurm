#!/bin/bash
#SBATCH --job-name=p12a-lowz-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/p12a-lowz-full-%j.log
set -euo pipefail
cd /global/u2/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
binding=configs/lowz_retraining_binding_20261118.json
digest() { sha256sum workflows/abacus_tweb/{p10*,p12a*,p6*}.py shared/*.py "$binding" workflows/sbi/p12a_lowz_full.slurm | sha256sum | cut -d' ' -f1; }
if [[ ${1:-} == --digest ]]; then digest; exit; fi
test "$(git rev-parse HEAD)" = "$P12_REV"
test "$(digest)" = "$P12_DIGEST"
manifest=$(jq -er .p12a_encoder.manifest "$binding")
test "$(sha256sum "$manifest" | cut -d' ' -f1)" = "$(jq -er .p12a_encoder.sha256 "$binding")"
contract=$(jq -er .contracts.full.path "$manifest")
test "$(sha256sum "$contract/TRAINING_LOADER_READY.json" | cut -d' ' -f1)" = "$(jq -er .contracts.full.sha256 "$manifest")"
out=/pscratch/sd/d/dkololgi/abacus/p12a_lowz_full_seed42_${SLURM_JOB_ID}
mkdir "$out"
sha256sum "$binding" "$manifest" "$contract/TRAINING_LOADER_READY.json" > "$out/input_hashes.txt"
printf '%s\n' "$P12_REV" "$P12_DIGEST" > "$out/source_revision_digest.txt"
exec srun -N1 -n1 -c32 --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -m workflows.abacus_tweb.p10_train_arm_a --contract-root "$contract" --output-root "$out" --run-name full --model unet --seed 42 --ordered-prefetch --disable-early-stopping --epochs 4 --min-epochs 4 --max-runtime-seconds 85200 --validation-reserve-seconds 7200 --checkpoint-every 1000
