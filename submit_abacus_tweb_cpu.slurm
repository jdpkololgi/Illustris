#!/bin/bash
#SBATCH --job-name=abacus_tweb_3414
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --time=08:00:00
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_tweb_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_tweb_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

set -euo pipefail

# DEPRECATION SHIM WARNING
echo "DEPRECATION WARNING: submit_abacus_tweb_cpu.slurm is a compatibility shim and may be removed in a future cycle. Use workflows entrypoint `workflows/abacus_tweb/submit_abacus_tweb_cpu.slurm` instead." >&2

# Compatibility wrapper for migrated SLURM entrypoint.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/workflows/abacus_tweb/submit_abacus_tweb_cpu.slurm" "$@"
