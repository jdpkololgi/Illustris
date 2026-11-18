#!/usr/bin/env bash
# One compute step; deterministic qualification, annotation and P1 preparation.
set -euo pipefail
phase=${1:?Pass a newly admitted training phase}
case "$phase" in ph001|ph014|ph015|ph016|ph017) ;; *) printf '%s\n' 'Unsupported phase' >&2; exit 2;; esac
: "${SLURM_JOB_ID:?Approved allocation required}"
: "${SLURM_STEP_ID:?Compute step required}"
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PATH=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
registry=configs/p10_phase_registry_v3.json
root=/pscratch/sd/d/dkololgi/abacus/p12a_r7_20phase_20261027_v1/$phase
[[ -s "$root/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json" ]] || { printf '%s\n' 'T-Web stage incomplete' >&2; exit 2; }
exec > >(tee "$root/catalogue_preparation_${SLURM_JOB_ID}.log") 2>&1
"$py" -u workflows/abacus_tweb/validate_galaxy_position_mapping.py --registry "$registry" --phases "$phase" --output "$root/qualification_audit"
"$py" -u workflows/abacus_tweb/certify_galaxy_position_scope.py --audit "$root/qualification_audit/report.json" --samples "$root/qualification_audit" --output "$root/qualification.json"
"$py" -u workflows/abacus_tweb/smoke_galaxy_position_annotation.py --certificate "$root/qualification.json" --samples "$root/qualification_audit" --output "$root/annotation_smoke"
"$py" -u workflows/abacus_tweb/migrate_p10_galaxy_position_truth.py --registry "$registry" --phase "$phase" --qualification "$root/qualification.json" --output-root "$root/catalogues" --field-in-memory --execute
"$py" -u workflows/abacus_tweb/p10_build_phase_index.py --registry "$registry" --phase "$phase" --observed "$root/catalogues/${phase}_observed_galaxy_realspace_v2_tweb.fits" --out-dir "$root/p1_canonical"
