#!/bin/bash
#SBATCH --job-name=p10-v2-catalogues
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --licenses=scratch
#SBATCH --array=0-10%1
#SBATCH --no-requeue
#SBATCH --output=/global/u2/d/dkololgi/TNG/Illustris/outputs/p10_expanded_catalogues_20261021/slurm-%A_%a.log
set -euo pipefail
campaign=/global/u2/d/dkololgi/TNG/Illustris/outputs/p10_expanded_catalogues_20261021
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python "$campaign/verify_source.py"
phases=(ph008 ph009 ph010 ph011 ph012 ph013 ph020 ph021 ph022 ph023 ph024)
exec bash "$campaign/source/workflows/abacus_tweb/run_expanded_phase_catalogue_20261021.sh" "${phases[$SLURM_ARRAY_TASK_ID]}"
