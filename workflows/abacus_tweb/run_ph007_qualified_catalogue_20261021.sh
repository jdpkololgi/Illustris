#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PATH=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH
python=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
root=/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/ph007
registry=$repo/configs/p10_phase_registry_v2.json
run() { srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$python" -u "$@"; }
run workflows/abacus_tweb/run_galaxy_position_tests.py
run workflows/abacus_tweb/validate_galaxy_position_mapping.py --registry "$registry" --phases ph007 --output "$root/qualification_audit_equivalence"
run workflows/abacus_tweb/certify_galaxy_position_scope.py --audit "$root/qualification_audit_equivalence/report.json" --samples "$root/qualification_audit_equivalence" --output "$root/qualification_equivalence.json"
run workflows/abacus_tweb/smoke_galaxy_position_annotation.py --certificate "$root/qualification_equivalence.json" --samples "$root/qualification_audit_equivalence" --output "$root/annotation_smoke_equivalence"
run workflows/abacus_tweb/migrate_p10_galaxy_position_truth.py --registry "$registry" --phase ph007 --qualification "$root/qualification_equivalence.json" --output-root "$root/catalogues" --execute
run workflows/abacus_tweb/p10_build_phase_index.py --registry "$registry" --phase ph007 --observed "$root/catalogues/ph007_observed_galaxy_realspace_v2_tweb.fits" --out-dir "$root/p1_canonical"
run workflows/abacus_tweb/refresh_p10_expansion_graph.py
cd /global/u2/d/dkololgi/TNG/Illustris
srun --nodes=1 --ntasks=1 --cpus-per-task=8 graphify global add graphify-out/graph.json --as Illustris
