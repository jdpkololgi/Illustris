#!/bin/bash
#SBATCH --job-name=p12a_p1_annot
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --licenses=scratch,cfs
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/p12a_ph001_annotation_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/p12a_ph001_annotation_%j.err

set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=${P12A_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
TRUTH_ROOT=${P12A_TRUTH_ROOT:-/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1}
BLIND_PARENT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph001/catalogues/blind_parent/ph001_bgs_bright_parent_linkage.fits
HALO_INFO=/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph001/halos/z0.200/halo_info
cd "$REPO"

timeout 120 "$PY" -c 'import asdf,fitsio,numpy; print("P12A_PH001_ANNOTATION_RUNTIME_OK")'
"$PY" -u -m workflows.sbi.p12a_authorized_truth guard \
  --stage annotation --truth-root "$TRUTH_ROOT"
if "$PY" -u -m workflows.sbi.p12a_authorized_truth status \
  --stage annotation --truth-root "$TRUTH_ROOT"; then
  echo "Authorized annotation stage is already complete and unchanged."
  exit 0
else
  status=$?
  [[ "$status" -eq 3 ]] || exit "$status"
fi

tweb_complete=$("$PY" -m workflows.sbi.p12a_authorized_truth artifact \
  --stage tweb --name p10_tweb_complete --truth-root "$TRUTH_ROOT")
tweb_dir=${tweb_complete%/*}
attempt="$TRUTH_ROOT/attempts/annotation/$SLURM_JOB_ID"
output_name=ph001_bgs_bright_parent_linkage_with_tweb_eigs.fits
mkdir -p "$attempt/output" "$attempt/tmp"
"$PY" -u workflows/abacus_tweb/annotate_cutsky_with_tweb_eigs.py \
  --cutsky "$BLIND_PARENT" \
  --tweb-dir "$tweb_dir" \
  --halo-info-dir "$HALO_INFO" \
  --halo-pos-field x_com \
  --output-dir "$attempt/output" \
  --output-name "$output_name" \
  --temp-dir "$attempt/tmp"
"$PY" -u -m workflows.sbi.p12a_authorized_truth complete-annotation \
  --truth-root "$TRUTH_ROOT" \
  --annotated-parent "$attempt/output/$output_name"
