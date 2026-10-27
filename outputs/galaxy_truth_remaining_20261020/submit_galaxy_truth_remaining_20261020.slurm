#!/bin/bash
#SBATCH --job-name=galaxy-truth-v2
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --licenses=scratch
#SBATCH --array=0-4%1
#SBATCH --no-requeue
#SBATCH --output=/global/u2/d/dkololgi/TNG/Illustris/outputs/galaxy_truth_remaining_20261020/slurm-%A_%a.log

set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PATH=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH
repo=/global/u2/d/dkololgi/TNG/Illustris
source_root=$repo/outputs/galaxy_truth_benchmark_20261020/source
campaign=$repo/outputs/galaxy_truth_remaining_20261020
output_root=/pscratch/sd/d/dkololgi/abacus/galaxy_truth_remaining_20261020
phases=(ph000 ph003 ph004 ph005 ph006)
phase=${phases[$SLURM_ARRAY_TASK_ID]}
python=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
qualification=$source_root/workflows/abacus_tweb/galaxy_position_qualification_allpop_v2_20261019.json
if [[ "$phase" == ph006 ]]; then qualification=$campaign/qualification_ph006.json; fi
cd "$source_root"
"$python" - "$repo" "$campaign" <<'PY'
import hashlib,json,sys
from pathlib import Path
repo,campaign=map(Path,sys.argv[1:])
freeze=json.loads((repo/'outputs/galaxy_truth_benchmark_20261020/SOURCE_FREEZE.json').read_text())
for name,sha in freeze['source_sha256'].items():
    if hashlib.sha256((Path(freeze['source_root'])/name).read_bytes()).hexdigest()!=sha:
        raise RuntimeError('Frozen source changed: '+name)
launch=json.loads((campaign/'LAUNCH.json').read_text())
for name,sha in launch['frozen_files'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:
        raise RuntimeError('Launch dependency changed: '+name)
PY
catalogues=$output_root/$phase/catalogues
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$python" -u workflows/abacus_tweb/migrate_p10_galaxy_position_truth.py --phase "$phase" --qualification "$qualification" --output-root "$catalogues" --execute
observed=$catalogues/${phase}_observed_galaxy_realspace_v2_tweb.fits
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$python" -u workflows/abacus_tweb/p10_build_phase_index.py --phase "$phase" --observed "$observed" --out-dir "$output_root/$phase/p1_canonical"
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$python" -u workflows/abacus_tweb/audit_galaxy_truth_replacement.py --phase "$phase" --legacy-root /pscratch/sd/d/dkololgi/abacus/p10_multiphase --replacement-root "$output_root" --observed "$observed"
printf 'CATALOGUE_REBUILD_FINISHED phase=%s job=%s output=%s\n' "$phase" "$SLURM_JOB_ID" "$output_root/$phase"
