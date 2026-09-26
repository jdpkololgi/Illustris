#!/usr/bin/env bash
# Named-job adapter for the immutable, previously validated array launcher.
set -euo pipefail
candidate_root=$1
phase=$2
case "$phase" in
  ph003) export SLURM_ARRAY_TASK_ID=2 ;;
  ph004) export SLURM_ARRAY_TASK_ID=3 ;;
  ph005) export SLURM_ARRAY_TASK_ID=4 ;;
  ph006) export SLURM_ARRAY_TASK_ID=5 ;;
  *) echo "Unsupported replacement phase: $phase" >&2; exit 2 ;;
esac
# This compatibility variable selects a phase; no Slurm array is submitted.
exec bash "$candidate_root/source/workflows/sbi/run_p12a_halo48_stage.sh" export "$candidate_root"
