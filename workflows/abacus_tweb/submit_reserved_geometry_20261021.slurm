#!/bin/bash
#SBATCH --job-name=p10-reserved-inputs
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --licenses=scratch
#SBATCH --array=0-5%1
#SBATCH --no-requeue
#SBATCH --output=/global/u2/d/dkololgi/TNG/Illustris/outputs/p10_reserved_geometry_20261021/slurm-%A_%a.log
set -euo pipefail
campaign=/global/u2/d/dkololgi/TNG/Illustris/outputs/p10_reserved_geometry_20261021
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
"$py" "$campaign/verify_source.py"
cd "$campaign/source"
phases=(ph014 ph015 ph016 ph017 ph018 ph019)
phase=${phases[$SLURM_ARRAY_TASK_ID]}
run() { srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$@"; }
run bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh "$phase" p1
run bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh "$phase" graph-cap SGC
run bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh "$phase" graph-cap NGC
run "$py" workflows/abacus_tweb/p10_build_phase_graph.py --registry configs/p10_phase_registry_v2.json --phase "$phase" --stage merge
run bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh "$phase" p3
