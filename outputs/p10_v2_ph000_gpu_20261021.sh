#!/bin/bash
set -euo pipefail
cd /global/u2/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gpus=1 "$py" - <<'PY'
from pathlib import Path
base=Path('/pscratch/sd/d/dkololgi/abacus')
s=base/'p10_multiphase/ph000/targets';d=base/'p10_galaxy_v2_20261020/ph000/targets'
if d.exists() or d.is_symlink():
 if d.resolve()!=s.resolve():raise FileExistsError(d)
else:d.symlink_to(s,target_is_directory=True)
PY
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --cpu-bind=cores bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh ph000 p2-post
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --cpu-bind=cores bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh ph000 p3-p4
