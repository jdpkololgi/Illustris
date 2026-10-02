#!/bin/bash
#SBATCH --job-name=loa_global_cap
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu&hbm80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=03:30:00
#SBATCH --licenses=scratch
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_global_canvas_20261002_v1/job_%j.log
#SBATCH --no-requeue

set -euo pipefail
CAP_SOURCE=${1:?Frozen source directory required}
CAP_NAME=${2:?NGC or SGC required}
case "$CAP_NAME" in NGC|SGC) ;; *) exit 64 ;; esac
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
cd "$CAP_SOURCE"
CAP_ROOT=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_global_canvas_20261002_v1
python - "$CAP_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);bench=json.loads((root/'full_canvas_benchmark/BENCHMARK.json').read_text())
assert bench['passed'] and bench['cap']=='NGC' and bench['nfe']==2 and not bench['posterior_output']
binding=json.loads((root/'full_canvas_benchmark/BINDING.json').read_text())
for name,h in binding['source'].items():
    assert hashlib.sha256((Path('workflows/sbi')/name).read_bytes()).hexdigest()==h,name
assert json.loads((root/'smoke/SMOKE.json').read_text())['passed']
print('Full-canvas benchmark source binding verified',flush=True)
PY
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --cpu-bind=cores \
 python -u -m workflows.sbi.e2e_global_cap \
 --root /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_field_atlas_20260930_v1 \
 --out "$CAP_ROOT/$CAP_NAME" --cap "$CAP_NAME" --smoke "$CAP_ROOT/smoke/SMOKE.json"
