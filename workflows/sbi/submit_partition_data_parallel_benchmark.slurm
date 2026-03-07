#!/bin/bash
#SBATCH --job-name=abacus_part_dp_bench
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_part_dp_bench_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_part_dp_bench_%j.err

source ~/.bashrc
conda activate cosmic_env

MANIFEST="${MANIFEST:-/pscratch/sd/d/dkololgi/abacus/partitions/partition_manifest.json}"
SPLIT="${SPLIT:-train}"
MAX_PARTITIONS="${MAX_PARTITIONS:-40}"
BENCH_STEPS="${BENCH_STEPS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
LOG_DIR="${LOG_DIR:-/pscratch/sd/d/dkololgi/logs}"

srun python -u "/global/homes/d/dkololgi/TNG/Illustris/workflows/sbi/benchmark_partition_data_parallel.py" \
  --partition-manifest "$MANIFEST" \
  --split "$SPLIT" \
  --max-partitions "$MAX_PARTITIONS" \
  --warmup-steps "$WARMUP_STEPS" \
  --bench-steps "$BENCH_STEPS" \
  --log-dir "$LOG_DIR"
