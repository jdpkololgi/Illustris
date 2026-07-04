#!/bin/bash
# G4-PROPER run E — attentional DGCNN: dynamic feature-space kNN (recomputed per
# layer; layer 0 = coordinate kNN) + GAT/GAPNet-style attention aggregation,
# positions+LOS only. The one wave-1 model whose graph is NOT fixed.
# E - D isolates learned dynamic candidate selection vs fixed radius candidates.
set -euo pipefail

OUT=/pscratch/sd/d/dkololgi/abacus/sbi_runs/g4_p1e_dgcnn_attn
LOG=/pscratch/sd/d/dkololgi/logs/g4_p1e_dgcnn_attn_$(date +%Y%m%d_%H%M%S).log
GC=/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign
mkdir -p "$OUT" /pscratch/sd/d/dkololgi/logs
echo "=== G4 run E (attentional DGCNN): salloc 1xA100 4h at $(date); log $LOG ==="

salloc --nodes=1 --gpus-per-node=1 --cpus-per-task=32 --constraint=gpu \
       --qos=interactive --time=04:00:00 --account=desi_g \
  srun -n 1 bash -lc '
    set -euo pipefail
    unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cosmic_env
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # cache unpickle initialises JAX (jnp arrays) -> would prealloc 75% of GPU
    export JAX_PLATFORMS=cpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
    cd /global/u2/d/dkololgi/TNG/Illustris
    python -u workflows/sbi/gate_g4_p1e_dgcnn_attn.py \
      --cache /pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl \
      --points-xyz '"$GC"'/path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_points_xyz.npy \
      --out-dir '"$OUT"' \
      --k 20 --dim 128 --layers 4 --heads 4 --minutes 200 --seed 42
  ' 2>&1 | tee -a "$LOG"
echo "=== G4 run E EXITED $(date) ==="
