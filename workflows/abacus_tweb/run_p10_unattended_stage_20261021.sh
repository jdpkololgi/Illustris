#!/bin/bash
set -euo pipefail
mode=${1:?stage};phase_list=${2:?explicit comma-separated phases}
IFS=',' read -r -a phases <<< "$phase_list"
phase=${phases[${SLURM_ARRAY_TASK_ID:-0}]}
case "$phase" in ph00[0-9]|ph01[0-9]|ph02[0-4]) ;; *) exit 2;; esac
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cpu=${SLURM_CPUS_PER_TASK:-32}
run() {
 if [[ "$mode" == gpu ]]; then
  srun --nodes=1 --ntasks=1 --cpus-per-task="$cpu" --gpus=1 --cpu-bind=cores "$@"
 else
  srun --nodes=1 --ntasks=1 --cpus-per-task="$cpu" --cpu-bind=cores "$@"
 fi
}
case "$mode" in
 graph)
  run bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh "$phase" graph-cap SGC
  run bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh "$phase" graph-cap NGC
  run "$py" workflows/abacus_tweb/p10_build_phase_graph.py --registry configs/p10_phase_registry_v2.json --phase "$phase" --stage merge
  run bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh "$phase" p3
  ;;
 gpu)
  run bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh "$phase" p2-post
  run bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh "$phase" p3-p4
  ;;
 reuse)
  [[ "$phase" == ph006 ]] || exit 2
  run "$py" workflows/abacus_tweb/adopt_corrected_p1_20261021.py --phase "$phase"
  run "$py" workflows/abacus_tweb/reuse_corrected_p2_20261021.py --phase "$phase"
  run "$py" workflows/abacus_tweb/adopt_p10_existing_inputs_20261021.py
  run bash workflows/abacus_tweb/run_p10_v2_phase_stage.sh "$phase" p3-p4
  ;;
 *) exit 2;;
esac
