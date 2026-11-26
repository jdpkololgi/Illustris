#!/bin/bash
set -eu
cd /pscratch/sd/d/dkololgi/abacus/coarse_source_61551798
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=16
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_lowz_20261029.json CUBLAS_WORKSPACE_CONFIG=:4096:8
r=/global/u2/d/dkololgi/TNG/Illustris/workflows/sbi/lowz_loa_draw.py
m=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/lowz_loa_response_61911856/RESPONSE_PREPARED.json
h=2b82368046841046d1dddfb7e0bcb74401f63ca50adcf21b7f6e48a2db0ba451
echo "01a4ed3372075d7486586e275377e6b20587c6c1f2ee4a4489f150adc66d2844  $r" | sha256sum -c -
[[ "$SLURM_NTASKS" == 8 ]]
rank=$SLURM_PROCID
if ((rank<4)); then cap=NGC; draws=($((rank+5))); else cap=SGC; draws=($((rank+5)) $((rank+9))); fi
for draw in "${draws[@]}"; do
  if ((draw>12 && SECONDS>7800)); then echo 'Insufficient remaining walltime'; exit 1; fi
  for stage in density tides; do
    /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u "$r" "$stage" "$cap" "$draw" "$m" "$h"
  done
done
