#!/bin/bash
# Isolated official LitePT-S A100 development gate; no existing environment writes.
set -euo pipefail
: "${SLURM_STEP_ID:?compute step required}" "${SLURM_JOB_ID:?}"
umask 077
root=/pscratch/sd/d/dkololgi/abacus
out="$root/lowz_litept_preflight_$SLURM_JOB_ID"
mkdir "$out"
exec >"$out/preflight.log" 2>&1
base=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" TORCH_HOME="$out/cache/torch"
export CUDA_CACHE_PATH="$out/cache/cuda" CUMM_CUDA_ARCH_LIST=8.0
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd /global/u2/d/dkololgi/TNG/Illustris
git diff --exit-code 87812ce -- workflows/abacus_tweb/{lowz_litept,lowz_point_loader,p8_train_unet_patch}.py
up="$root/litept_source_436d048"
test "$(git -C "$up" rev-parse HEAD)" = 436d04801c8151faebe66a1b2d368a9711e7e6aa
git -C "$up" diff --quiet HEAD
"$base" -m venv "$out/venv"
py="$out/venv/bin/python"
export PATH="$out/venv/bin:$PATH"
t=https://download.pytorch.org/whl/cu124
fa=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1
sc=https://data.pyg.org/whl/torch-2.5.0%2Bcu124
"$py" -m pip --isolated --disable-pip-version-check install --cache-dir "$out/cache/pip" --index-url https://pypi.org/simple \
  "$t/torch-2.5.1%2Bcu124-cp311-cp311-linux_x86_64.whl" \
  "$t/torchvision-0.20.1%2Bcu124-cp311-cp311-linux_x86_64.whl" \
  "$fa/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
  "$sc/torch_scatter-2.1.2%2Bpt25cu124-cp311-cp311-linux_x86_64.whl" \
  'spconv-cu124==2.3.8' 'numpy==1.26.4' scipy scikit-learn h5py addict timm colorhash einops
"$py" -m pip freeze > "$out/packages.txt"
"$py" -c 'import torch,torch_scatter,spconv.pytorch,flash_attn; print(torch.__version__,torch.cuda.get_device_name(),flush=True); assert torch.cuda.get_device_capability()==(8,0)'
"$py" -u -m workflows.abacus_tweb.lowz_litept "$root/lowz_points_61662493" "$up" | tee "$out/updates.log"
"$py" - "$out" <<'PY'
import json,math,sys
from pathlib import Path
p=Path(sys.argv[1])
rows=[json.loads(s) for s in (p/"updates.log").read_text().splitlines() if s.startswith("{")]
assert len(rows)==len({r["core"] for r in rows})==8
assert all(r["n"]>0 and r["n"]<=r["context"] and math.isfinite(r["grad_norm"]) and math.isfinite(r["loss"]) for r in rows)
(p/"LITEPT_PREFLIGHT_PASS.json").write_text(json.dumps(dict(pass_=True,phase="ph002",updates=rows)))
PY
