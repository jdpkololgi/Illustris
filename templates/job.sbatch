#!/bin/bash
# Minimal, robust Perlmutter sbatch template (adapt as needed)

#SBATCH --job-name=desi-job
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

# If you rely on modules, load them here.
# module load ...

# Run work under srun so it runs on allocated resources.
srun bash -lc 'python -V; echo "Hello from compute allocation"'

echo "End: $(date -Is)"

