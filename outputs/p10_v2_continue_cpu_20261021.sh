#!/bin/bash
set -euo pipefail
cd /global/u2/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
run() { srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$py" -u "$@"; }
for phase in ph002 ph003 ph004 ph005; do
 run workflows/abacus_tweb/adopt_corrected_p1_20261021.py --phase "$phase"
done
run workflows/abacus_tweb/p10_materialize_phase_schemas.py --registry configs/p10_phase_registry_v2.json --phase ph000
run -m unittest discover -s tests/phase4 -p test_p10_phase_products.py
run workflows/abacus_tweb/p10_build_phase_graph.py --registry configs/p10_phase_registry_v2.json --phase ph000 --stage cap --cap NGC
run workflows/abacus_tweb/p10_build_phase_graph.py --registry configs/p10_phase_registry_v2.json --phase ph000 --stage merge
