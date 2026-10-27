#!/usr/bin/env bash
# Invoke only inside separately approved CPU allocations. No submission/retry.
set -euo pipefail
stage=${1:?usage: run_p10_expansion_pilot.sh targets-or-catalogues}
: "${SLURM_JOB_ID:?Approved CPU allocation required}"
repo=$(cd "$(dirname "$0")/../.." && pwd)
registry=$repo/configs/p10_phase_registry_v2.json
root=/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/ph007
python=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH
cd "$repo"
field=$root/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7
density_manifest=$root/targets/density/AbacusSummit_base_c000_ph007_z0.200_ngrid2048_ab10_tsc_counts.job${SLURM_JOB_ID}.manifest.json
case "$stage" in
 targets)
  # Four CPU nodes, 16 MPI ranks, 32 CPUs per rank, 30 minute allocation.
  srun --nodes=1 --ntasks=1 --cpus-per-task=4 --cpu-bind=cores "$python" -u workflows/abacus_tweb/p10_expansion_preflight.py \
    --registry "$registry" --adapt-density ph007 --output "$density_manifest"
  srun --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --kill-on-bad-exit=1 "$python" -u workflows/abacus_tweb/p10_run_tweb.py \
    --registry "$registry" --phase ph007 \
    --density /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1/matter/ph007/counts_tsc_2048.npy \
    --density-manifest "$density_manifest" --output-dir "$field"
  ;;
 catalogues)
  # One CPU node, 64 CPUs, 90 minute allocation; dependent on passed target stage.
  [[ -s "$field/TWEB_COMPLETE.json" ]] || { echo 'Target stage incomplete' >&2; exit 2; }
  export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  run() { srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$python" -u "$@"; }
  run -m unittest tests.test_p10_phase_expansion tests.phase4.test_p10_multiphase_assets tests.phase4.test_p10_tweb_runner
  run workflows/abacus_tweb/validate_galaxy_position_mapping.py --registry "$registry" --phases ph007 --output "$root/qualification_audit"
  run workflows/abacus_tweb/certify_galaxy_position_scope.py --audit "$root/qualification_audit/report.json" --samples "$root/qualification_audit" --output "$root/qualification.json"
  run workflows/abacus_tweb/smoke_galaxy_position_annotation.py --certificate "$root/qualification.json" --samples "$root/qualification_audit" --output "$root/annotation_smoke"
  run workflows/abacus_tweb/migrate_p10_galaxy_position_truth.py --registry "$registry" --phase ph007 --qualification "$root/qualification.json" --output-root "$root/catalogues" --execute
  run workflows/abacus_tweb/p10_build_phase_index.py --registry "$registry" --phase ph007 --observed "$root/catalogues/ph007_observed_galaxy_realspace_v2_tweb.fits" --out-dir "$root/p1_canonical"
  srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores "$python" workflows/abacus_tweb/refresh_p10_expansion_graph.py
  cd /global/u2/d/dkololgi/TNG/Illustris
  srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores graphify global add graphify-out/graph.json --as Illustris
  ;;
 *) echo 'Choose targets or catalogues' >&2; exit 2;;
esac
