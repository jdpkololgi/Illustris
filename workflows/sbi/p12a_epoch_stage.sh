#!/bin/bash
#SBATCH --job-name=p12a-epoch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&a100
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --time=14:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/p12a-epoch-%j.log
set -euo pipefail
cd /global/u2/d/dkololgi/TNG/Illustris
out=/pscratch/sd/d/dkololgi/abacus/p12a_lowz_full_seed42_61548250
d="$out/full/unet/seed_42"
epoch=${1:?}
case "$epoch" in 1|2|3|4) ;; *) exit 2 ;; esac
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
test "$(sha256sum "$REPAIR_LIST" | cut -d' ' -f1)" = "$REPAIR_DIGEST"
sha256sum -c "$REPAIR_LIST" --status
sha256sum -c "$out/input_hashes.txt" --status
contract=$(jq -er .arguments.contract_root "$d/run_manifest.json")
cmd=(srun -N1 -n1 -c32 --exact --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -m)
if [[ $epoch == 1 ]]; then
    exec >"$out/validation-$SLURM_JOB_ID.log" 2>&1
    "${cmd[@]}" workflows.sbi.p12a_resume_epoch1_61548250
else
    previous=$((epoch-1))
    jq -e --argjson p "$previous" '.epoch==$p and .global_step==346364*$p and .complete_epoch_coverage==true and .repeat_cores==0 and (.primary_macro_r2_lambda1|isfinite)' < <(tail -1 "$d/epoch_history.jsonl") >/dev/null
    test -e "$d/STAGE_${previous}_PASSED"
fi
seconds=49800
if [[ $epoch == 1 ]]; then seconds=13800; fi
"${cmd[@]}" workflows.abacus_tweb.p10_train_arm_a --contract-root "$contract" --output-root "$out" --run-name full --model unet --seed 42 --ordered-prefetch --disable-early-stopping --epochs 4 --min-epochs 4 --checkpoint-every 1000 --max-runtime-seconds "$seconds" --validation-reserve-seconds 5400 --auto-resume --stage-epoch "$epoch"
jq -es --argjson e "$epoch" 'length==$e and last.epoch==$e and last.global_step==346364*$e and last.complete_epoch_coverage and last.repeat_cores==0 and (last.primary_macro_r2_lambda1|isfinite)' "$d/epoch_history.jsonl" >/dev/null
if [[ $epoch == 1 ]]; then
    jq -e '.complete_phase_coverage and (.per_phase|keys)==["ph006","ph012","ph013"] and all(.per_phase[]; .complete_phase_coverage and .runtime.patch_failures==0 and .per_shell["0p01_0p25"].n>0)' "$d/best_validation_report.json" >/dev/null
fi
if [[ $epoch == 4 ]]; then test -s "$d/ARM_A_TRAINING_COMPLETE.json"; fi
(set -C; echo "$SLURM_JOB_ID" > "$d/STAGE_${epoch}_PASSED")
