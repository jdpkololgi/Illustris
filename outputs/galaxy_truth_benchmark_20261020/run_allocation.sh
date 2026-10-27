#!/bin/bash
set -uo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PATH=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH
repo=/global/u2/d/dkololgi/TNG/Illustris
run=$repo/outputs/galaxy_truth_benchmark_20261020
scratch_run=/pscratch/sd/d/dkololgi/abacus/galaxy_truth_benchmark_20261020/job${SLURM_JOB_ID}
python=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd "$run/source" || exit 2
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$python" -u workflows/abacus_tweb/run_galaxy_truth_benchmark.py --run-root "$scratch_run"
benchmark_rc=$?
cd "$repo" || exit 2
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$python" "$run/refresh_graph.py"
graph_rc=$?
if [ "$graph_rc" -eq 0 ]; then
  srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores graphify global add graphify-out/graph.json --as Illustris
  graph_rc=$?
fi
printf 'BENCHMARK_EXIT=%s GRAPH_EXIT=%s JOB=%s OUTPUT=%s\n' "$benchmark_rc" "$graph_rc" "$SLURM_JOB_ID" "$scratch_run"
if [ "$benchmark_rc" -ne 0 ]; then exit "$benchmark_rc"; fi
exit "$graph_rc"
