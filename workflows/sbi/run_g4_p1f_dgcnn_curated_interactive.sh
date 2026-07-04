#!/bin/bash
# G4-PROPER run F — attentional DGCNN (dynamic feature-space kNN) WITH curated
# Delaunay/cuGraph node features (JDPK 2026-07-04). Tests whether dynamic
# candidate selection helps when the model ALSO has the curated field estimators
# (vs run E's positions-only, which lost). Comparisons:
#   F vs E     = curated vs positions-only, both dynamic graph
#   F vs A     = dynamic vs radius, both curated features
#   F vs base  = dynamic vs Delaunay, both curated features
# Uncapped dynamic graph (canonical DGCNN) to match E; --knn-radius-cap is the
# available follow-up knob if F (or E's void slice) motivates a capped variant.
set -euo pipefail

OUT=/pscratch/sd/d/dkololgi/abacus/sbi_runs/g4_p1f_dgcnn_curated
LOG=/pscratch/sd/d/dkololgi/logs/g4_p1f_dgcnn_curated_$(date +%Y%m%d_%H%M%S).log
GC=/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign
mkdir -p "$OUT" /pscratch/sd/d/dkololgi/logs
echo "=== G4 run F (DGCNN + curated features): salloc 1xA100 4h at $(date); log $LOG ==="

salloc --nodes=1 --gpus-per-node=1 --cpus-per-task=32 --constraint=gpu \
       --qos=interactive --time=04:00:00 --account=desi_g \
  srun -n 1 bash -lc '
    set -euo pipefail
    unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cosmic_env
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export JAX_PLATFORMS=cpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
    cd /global/u2/d/dkololgi/TNG/Illustris
    python -u workflows/sbi/gate_g4_p1e_dgcnn_attn.py \
      --cache /pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl \
      --points-xyz '"$GC"'/path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_points_xyz.npy \
      --out-dir '"$OUT"' \
      --curated-features --k 20 --dim 128 --layers 4 --heads 4 --minutes 200 --seed 42
  ' 2>&1 | tee -a "$LOG"
echo "=== G4 run F EXITED $(date) ==="
