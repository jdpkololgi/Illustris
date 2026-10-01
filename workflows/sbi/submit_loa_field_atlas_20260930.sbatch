#!/bin/bash
#SBATCH --job-name=loa_field_atlas
#SBATCH --account=desi_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --licenses=scratch
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_field_atlas_20260930_v1_batch_%j.log

set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/loa-atlas-mpl
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
ATLAS_ROOT=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_field_atlas_20260930_v1
ATLAS_SOURCE=${1:?Pass the frozen source snapshot directory}
cd "$ATLAS_SOURCE"
python - "$ATLAS_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);record=json.loads((root/'SMOKE_COMPLETE.json').read_text())
assert record['passed'] and record['draws']==8
binding=record['binding']
paths={'conditions':'e2e_loa_field_conditions.py','sampling':'e2e_loa_field_sampling.py',
       'tensor':'e2e_loa_field_tensor.py','operators':'e2e_coupled_operators.py'}
for key,name in paths.items():
    assert hashlib.sha256((Path('workflows/sbi')/name).read_bytes()).hexdigest()==binding['source'][key]
assert hashlib.sha256(Path('workflows/sbi/e2e_loa_field_atlas.py').read_bytes()).hexdigest()==binding['runner']
print('Frozen source matches successful scientific smoke binding',flush=True)
PY
export ATLAS_ROOT
srun --nodes=1 --ntasks=4 --cpus-per-task=32 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores \
  bash -c 'python -c "import torch; assert torch.cuda.device_count()==1" && python -u -m workflows.sbi.e2e_loa_field_atlas run --root "$ATLAS_ROOT" --worker "$SLURM_PROCID" --workers 4 --draws 8 --seconds 10200'
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_loa_field_atlas merge --root "$ATLAS_ROOT"
