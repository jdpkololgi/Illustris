#!/bin/bash
set -uo pipefail
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PATH="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH"
python -m outputs.field_restart_repair_20261026.verify_endpoint_entrypoint
a=$?
python -m unittest tests.test_field_restart_revision tests.test_e2e_wholefield_pilot_data tests.test_e2e_wholefield_pilot_models tests.phase4.test_p10_arm_a_training tests.phase4.test_p10_training_contract tests.phase4.test_p12_export_unet_summaries tests.phase4.test_p12_base_response tests.phase4.test_p12_production_contract tests.phase4.test_deterministic_interpolation tests.phase4.test_p12_expanded_crossfit tests.phase4.test_p12_expanded_inputs
b=$?
graphify update .
c=$?
if [ "$c" -eq 0 ]; then graphify global add graphify-out/graph.json --as Illustris; c=$?; fi
printf 'FINAL_EXIT_CODES entrypoint=%s tests=%s index=%s\n' "$a" "$b" "$c"
exit $((a || b || c))
