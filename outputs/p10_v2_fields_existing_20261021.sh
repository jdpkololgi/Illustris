#!/bin/bash
set -euo pipefail
cd /global/u2/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
for phase in ph002 ph003 ph004 ph005; do
 "$py" - "$phase" <<'PY'
import sys
from pathlib import Path
p=sys.argv[1];base=Path('/pscratch/sd/d/dkololgi/abacus')
source=base/'p10_multiphase'/p/'targets';target=base/'p10_galaxy_v2_20261020'/p/'targets'
assert (source/'tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json').is_file()
if target.exists() or target.is_symlink():
 if target.resolve()!=source.resolve():raise FileExistsError(target)
else:target.symlink_to(source,target_is_directory=True)
PY
 bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh "$phase" p3-p4
done
