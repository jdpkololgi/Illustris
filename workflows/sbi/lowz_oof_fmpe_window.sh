#!/bin/bash
set -eu -o pipefail
side=$1
cd "$FMPE_RUN/src"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
trap 'touch "$FMPE_RUN/failed.$side"' ERR
srun --kill-on-bad-exit=1 -N4 -n10 -c16 --gpus-per-node=4 "$PY" -um workflows.sbi.lowz_oof_fmpe_table shard "$side"
touch "$FMPE_RUN/sharded.$side"
waitfor() {
    for ((i=0;i<240;i++)); do
        [[ ! -e "$FMPE_RUN/failed.0" && ! -e "$FMPE_RUN/failed.1" ]] || return 1
        [[ ! -e "$1" ]] || return 0
        sleep 5
    done
    return 1
}
if [[ $side == 0 ]]; then
    waitfor "$FMPE_RUN/sharded.1"
    srun -N1 -n1 -c16 --gpus=1 "$PY" -um workflows.sbi.lowz_oof_fmpe_table join
fi
waitfor "$FMPE_RUN/P12A_DATASET_READY.json"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29458
export FIT_SEED=$((42+side))
srun --kill-on-bad-exit=1 -N4 -n4 --ntasks-per-node=1 -c64 --gpus-per-node=4 bash -c 'exec "$PY" -m torch.distributed.run --nnodes=4 --nproc-per-node=4 --node-rank="$SLURM_NODEID" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --max-restarts=0 -m workflows.sbi.lowz_oof_fmpe_fit "$FIT_SEED"'
