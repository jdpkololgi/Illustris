#!/bin/bash
#SBATCH --job-name=p12a-window-e3
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
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/encoder-resume-%j.log
set -euo pipefail
r=/pscratch/sd/d/dkololgi/abacus
out=$r/encoder_prod_window_attention_61664810
d=$out/sweep_window_attention_production/unet/seed_42
cd "$out/src"
test "$(git rev-parse HEAD)" = 71307f51349fd71263d5db8daa01e1d40ce7c010
jq -r '.source_contract|to_entries[]|.value+"  "+.key' "$d/run_manifest.json" | sha256sum --check --status
echo "65953678e09fdcf65673ceb1cc037a3241c79edc81f56b626496cb7fe76d14c4  $d/arm_a_checkpoint.pt" | sha256sum --check --status
tail -1 "$d/epoch_history.jsonl" | jq -e '.epoch==2 and .global_step==692728 and .complete_epoch_coverage' >/dev/null
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
"$py" -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert (c["resume"]["epoch"],c["resume"]["cursor"],c["global_step"],c["scheduler_state"]["last_epoch"])==(3,0,692728,692728)' "$d/arm_a_checkpoint.pt"
if [[ ${1:-} == --verify-only ]]; then exit 0; fi
exec srun -N1 -n1 -c32 --exact --gpus=1 --cpu-bind=cores "$py" -u -m workflows.abacus_tweb.p12a_encoder_sweep window_attention --ordered-prefetch --epochs 4 --min-epochs 4 --contract-root "$r/p12a_lowz_20261029_v1/training_contract_v1" --model unet --seed 42 --disable-early-stopping --auto-resume --run-name sweep_window_attention_production --output-root "$out" --checkpoint-every 1000 --max-runtime-seconds 84600 --validation-reserve-seconds 6500
