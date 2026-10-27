#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PATH=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH
python=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
phase=${1:?registered development phase required}
case "$phase" in ph008|ph009|ph010|ph011|ph012|ph013|ph020|ph021|ph022|ph023|ph024) ;; *) echo 'Unsupported or reserved phase' >&2; exit 2;; esac
root=/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/$phase
[[ -s "$root/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json" ]] || { echo 'Targets incomplete' >&2; exit 2; }
registry=$repo/configs/p10_phase_registry_v2.json
run() { srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$python" -u "$@"; }
run workflows/abacus_tweb/validate_galaxy_position_mapping.py --registry "$registry" --phases "$phase" --output "$root/qualification_audit_equivalence"
run workflows/abacus_tweb/certify_galaxy_position_scope.py --audit "$root/qualification_audit_equivalence/report.json" --samples "$root/qualification_audit_equivalence" --output "$root/qualification_equivalence.json"
run workflows/abacus_tweb/smoke_galaxy_position_annotation.py --certificate "$root/qualification_equivalence.json" --samples "$root/qualification_audit_equivalence" --output "$root/annotation_smoke_equivalence"
run workflows/abacus_tweb/migrate_p10_galaxy_position_truth.py --registry "$registry" --phase "$phase" --qualification "$root/qualification_equivalence.json" --output-root "$root/catalogues" --execute
run workflows/abacus_tweb/p10_build_phase_index.py --registry "$registry" --phase "$phase" --observed "$root/catalogues/${phase}_observed_galaxy_realspace_v2_tweb.fits" --out-dir "$root/p1_canonical"
