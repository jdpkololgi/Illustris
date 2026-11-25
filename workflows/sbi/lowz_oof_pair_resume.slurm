#!/bin/bash
#SBATCH --job-name=p12-oof-resume
#SBATCH --array=4,5,7,8
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
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/oof-resume-%A_%a.log
set -euo pipefail
case ${SLURM_ARRAY_TASK_ID:?} in
  4) phase=ph004; original=61932924; digest=e227105742b20719069092dcd23d1045442fe650636952d484df4e9d43068751;;
  5) phase=ph005; original=61932924; digest=372af26c07d2632c346e6595348ebfb57117a54ffc51c4705f80d2ff98a33d05;;
  7) phase=ph007; original=61932921; digest=ecf0078044b679ead31493ccde928872585f40b1b09c12275de28e9ee54b58dd;;
  8) phase=ph008; original=61932921; digest=10829043fe6ab44680d89a7089d972eb293a4690088f53d47bb696b7e8be43c2;;
  *) exit 2;;
esac
base=/pscratch/sd/d/dkololgi/abacus
out=$base/p12a_lowz_oof_${phase}_${original}
marker=$out/oof/unet/seed_42/arm_a_checkpoint.pt
echo "$digest  $marker" | sha256sum -c -
cd "$base/p12_oof_pair_source_${original}"
test "$(git rev-parse --short=7 HEAD)" = 02bbd35
git diff --quiet HEAD
manifest=$base/p12a_lowz_20261029_v1/LOWZ_TRAINING_PRODUCTS_COMPLETE.json
echo "24764983f5424593240469c68cee01222174d2af1e261efc8ef05b79083c45ee  $manifest" | sha256sum -c -
root=$(jq -er --arg key "omit_$phase" '.contracts[$key].path' "$manifest")
echo "$(jq -er --arg key "omit_$phase" '.contracts[$key].sha256' "$manifest")  $root/TRAINING_LOADER_READY.json" | sha256sum -c -
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
rc=0
srun -N1 -n1 -c32 --gpus=1 --cpu-bind=cores \
  /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -um workflows.abacus_tweb.p10_train_arm_a \
  --contract-root "$root" --output-root "$out" --run-name oof --model unet --seed 42 \
  --ordered-prefetch --epochs 4 --min-epochs 4 --disable-early-stopping --auto-resume \
  --checkpoint-every 1000 --max-runtime-seconds 19800 --validation-reserve-seconds 3600 || rc=$?
if [[ $rc == 75 ]]; then exit 0; fi
exit "$rc"
