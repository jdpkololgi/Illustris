#!/bin/bash
# G4-PROPER P1a-ii (run D) — point-cloud control: non-equivariant attention MPNN,
# positions + LOS-derived scalars ONLY, neighbourhoods built at LOAD TIME from the
# point distribution (cKDTree radius = 14.78 Mpc; no prebuilt graph artifact).
# Point-Transformer-class in the sense that matters: learned attention over local
# point neighbourhoods with no hand-crafted features. Fills the
# (positions-only, non-equivariant) cell: D vs A isolates curated features vs raw
# geometry; C vs D isolates equivariance at matched inputs+graph.
set -euo pipefail

OUT=/pscratch/sd/d/dkololgi/abacus/sbi_runs/g4_p1d_pointattn_radius
LOG=/pscratch/sd/d/dkololgi/logs/g4_p1d_pointattn_$(date +%Y%m%d_%H%M%S).log
GC=/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign
mkdir -p "$OUT" /pscratch/sd/d/dkololgi/logs
echo "=== G4 P1a-ii point-cloud control: salloc 1xA100 4h at $(date); log $LOG ==="

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
    python -u workflows/sbi/gate_g4_egnn_smoke.py \
      --cache /pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl \
      --points-xyz '"$GC"'/path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_points_xyz.npy \
      --positions-only --build-radius-mpc 14.78 \
      --aggregation attention --heads 4 --steps 6000 --seed 42
  ' 2>&1 | tee -a "$LOG"
echo "=== G4 P1a-ii EXITED $(date) ==="
