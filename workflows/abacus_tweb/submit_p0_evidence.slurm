#!/bin/bash
#SBATCH --job-name=graphweb_p0
#SBATCH --account=desi_g
#SBATCH --constraint="gpu&hbm80g"
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/graphweb_p0_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/graphweb_p0_%j.err

set -euo pipefail
source ~/.bashrc
conda activate cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

REPO=/global/homes/d/dkololgi/TNG/Illustris
ROOT=/pscratch/sd/d/dkololgi/abacus
OUT=${ROOT}/p0_evidence_freeze
CACHE=${ROOT}/sbi_caches/s3c_cnn_fullrange/cnn_fullrange_cache.pkl
POINTS=${ROOT}/sbi_caches/s3c_cnn_fullrange/cnn_fullrange_points.npy
mkdir -p "${OUT}"
cd "${REPO}"

python workflows/abacus_tweb/p0_export_graphnet_predictions.py \
  --model R0=/pscratch/sd/d/dkololgi/abacus/R0_valid_corrected/sbi_output/flowjax_sbi_model_seed_42_bestL1_20260714_111148.pkl \
  --model A1_sqrt=/pscratch/sd/d/dkololgi/abacus/A1_sqrt/sbi_output/flowjax_sbi_model_seed_42_bestL1_20260714_174330.pkl \
  --tiles-dir "${ROOT}/sbi_caches/s3b_tiled_valid_v2" \
  --canonical-cache "${CACHE}" --out-dir "${OUT}" --n-samples 128

python workflows/abacus_tweb/p0_evidence.py \
  --cache "${CACHE}" --points "${POINTS}" \
  --graphnet "R0=${OUT}/R0_canonical_predictions.npz" \
  --graphnet "A1_sqrt=${OUT}/A1_sqrt_canonical_predictions.npz" \
  --point "UNet=${ROOT}/C_unet_fullrange/scores.pred_eigs.npy:none" \
  --point "DTFE_raw=${ROOT}/classical_baseline/fullrange_holdout/pred_eigs_dtfe.npy:none" \
  --point "DTFE_train_affine=${ROOT}/classical_baseline/fullrange_holdout/pred_eigs_dtfe.npy:affine_train" \
  --point "CIC_raw=${ROOT}/classical_baseline/fullrange_holdout/pred_eigs_cic.npy:none" \
  --point "CIC_train_affine=${ROOT}/classical_baseline/fullrange_holdout/pred_eigs_cic.npy:affine_train" \
  --out "${OUT}/evidence_freeze.json" --block-mpc 100 --n-bootstrap 1000

python workflows/abacus_tweb/p0_inventory_assets.py \
  --canonical-cache "${CACHE}" --canonical-points "${POINTS}" \
  --out "${OUT}/asset_inventory.json"

sha256sum \
  "${OUT}/R0_canonical_predictions.npz" \
  "${OUT}/A1_sqrt_canonical_predictions.npz" \
  "${OUT}/evidence_freeze.json" \
  "${OUT}/asset_inventory.json" \
  > "${OUT}/SHA256SUMS"
echo "P0_COMPLETE job=${SLURM_JOB_ID}" > "${OUT}/P0_COMPLETE"
