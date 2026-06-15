#!/usr/bin/env bash
set -euo pipefail

echo "### Host"
hostname

echo
echo "### Slurm context"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-<none>}"
echo "SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT:-<none>}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-<none>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<none>}"

echo
echo "### Filesystem hints"
echo "HOME=$HOME"
echo "PSCRATCH=${PSCRATCH:-<unset>}"
echo "SCRATCH=${SCRATCH:-<unset>}"

echo
echo "### Simple node classification (best-effort)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Looks like you're inside a Slurm job allocation."
else
  echo "No Slurm job detected (likely login node or unallocated shell)."
fi

