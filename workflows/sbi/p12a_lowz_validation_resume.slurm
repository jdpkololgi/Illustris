#!/bin/bash
#SBATCH --job-name=p12a-val-resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&a100
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --time=46:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/p12a-lowz-resume-%j.log
set -euo pipefail
cd /global/u2/d/dkololgi/TNG/Illustris
out=/pscratch/sd/d/dkololgi/abacus/p12a_lowz_full_seed42_61548250
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
test "$(sha256sum "$REPAIR_LIST" | cut -d' ' -f1)" = "$REPAIR_DIGEST"
sha256sum -c "$REPAIR_LIST" --status
sha256sum -c "$out/input_hashes.txt" --status
contract=$(jq -er .arguments.contract_root "$out/full/unet/seed_42/run_manifest.json")
step=(srun -N1 -n1 -c32 --exact --gpus=1 --cpu-bind=cores)
"${step[@]}" "$py" -u -m workflows.sbi.p12a_resume_epoch1_61548250
exec "${step[@]}" "$py" -u -m workflows.abacus_tweb.p10_train_arm_a --contract-root "$contract" \
    --output-root "$out" --run-name full --model unet --seed 42 --ordered-prefetch \
    --disable-early-stopping --epochs 4 --min-epochs 4 --checkpoint-every 1000 \
    --max-runtime-seconds 163800 --validation-reserve-seconds 7200 --auto-resume
