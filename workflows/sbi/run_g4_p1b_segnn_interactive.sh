#!/bin/bash
# G4-PROPER P1b — SEGNN steerable + invariant-logit attention (first equivariant
# candidate, plan §5A #2). Usage: run_g4_p1b_segnn_interactive.sh {union|radius}
# Inputs are positions + LOS ONLY (§0 purity); the npz supplies edge_index only.
set -euo pipefail

VARIANT="${1:?usage: $0 union|radius}"
GC=/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign
case "$VARIANT" in
  union)  NPZ=$GC/path1_wedge_union_r10hmpc_gnn_arrays.npz ;;
  radius) NPZ=$GC/path1_wedge_radius_r10hmpc_gnn_arrays.npz ;;
  *) echo "unknown variant: $VARIANT"; exit 1 ;;
esac

OUT=/pscratch/sd/d/dkololgi/abacus/sbi_runs/g4_p1b_segnn_${VARIANT}
LOG=/pscratch/sd/d/dkololgi/logs/g4_p1b_segnn_${VARIANT}_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$OUT" /pscratch/sd/d/dkololgi/logs
echo "=== G4 P1b SEGNN ($VARIANT): salloc 1xA100 4h at $(date); log $LOG ==="

salloc --nodes=1 --gpus-per-node=1 --cpus-per-task=32 --constraint=gpu \
       --qos=interactive --time=04:00:00 --account=desi_g \
  srun -n 1 bash -lc '
    set -euo pipefail
    unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cosmic_env
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # the SI cache pickle contains jnp arrays -> unpickling initialises JAX,
    # which by default PREALLOCATES ~75% of the GPU under PyTorch. Pin JAX to CPU.
    export JAX_PLATFORMS=cpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
    cd /global/u2/d/dkololgi/TNG/Illustris
    python -u workflows/sbi/gate_g4_p1b_segnn.py \
      --cache /pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl \
      --gnn-arrays '"$NPZ"' \
      --points-xyz '"$GC"'/path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_points_xyz.npy \
      --out-dir '"$OUT"' \
      --minutes 200 --seed 42 \
      --hidden "16x0e+8x1o+4x2e" --layers 3 --heads 4 \
      --edge-sample 0.5 --val-every 50
  ' 2>&1 | tee -a "$LOG"
echo "=== G4 P1b SEGNN ($VARIANT) EXITED $(date) ==="
