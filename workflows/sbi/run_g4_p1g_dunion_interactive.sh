#!/bin/bash
# G4-PROPER run G ("D-union") — the SAME point-attention model as run D
# (non-equivariant, attention, MSE, POSITIONS-ONLY node features), but on the
# prebuilt UNION edge set (Delaunay ∪ radius) instead of the self-built radius
# graph. Completes the union-vs-radius row for our best positions-only model:
#   G vs D  = union vs radius, positions-only point-attention  (the missing cell)
#   G vs G3 = positions-only vs curated features, both union graph
#   G vs B  = non-equivariant vs steerable, both positions-only + union
# NOTE: the union graph REQUIRES the Delaunay triangulation, so G is NOT a pure
# "self-built from positions" model — it is positions-only FEATURES on a prebuilt
# union graph. That is the intended test (does union connectivity help point-
# attention?), not a no-preconstructed-graph claim.
set -euo pipefail

OUT=/pscratch/sd/d/dkololgi/abacus/sbi_runs/g4_p1g_dunion
LOG=/pscratch/sd/d/dkololgi/logs/g4_p1g_dunion_$(date +%Y%m%d_%H%M%S).log
GC=/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign
mkdir -p "$OUT" /pscratch/sd/d/dkololgi/logs
echo "=== G4 run G (D-union, positions-only + union graph): salloc 1xA100 4h at $(date); log $LOG ==="

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
    python -u workflows/sbi/gate_g4_egnn_smoke.py \
      --cache /pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl \
      --gnn-arrays '"$GC"'/path1_wedge_union_r10hmpc_gnn_arrays.npz \
      --points-xyz '"$GC"'/path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_points_xyz.npy \
      --positions-only \
      --aggregation attention --heads 4 --steps 6000 --seed 42 \
      --out-file '"$OUT"'/p1g_dunion_results.txt
  ' 2>&1 | tee -a "$LOG"
echo "=== G4 run G EXITED $(date) ==="
