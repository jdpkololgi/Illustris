#!/usr/bin/env bash
# Run inside an explicitly approved one-node CPU srun step.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an approved Slurm allocation}"
: "${SLURM_STEP_ID:?Run inside an approved Slurm step}"
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
out=/pscratch/sd/d/dkololgi/abacus/p12a_r7_20phase_20261027_v1/preparation_smoke_${SLURM_JOB_ID}
mkdir -p "$out"
exec > >(tee "$out/run.log") 2>&1
"$py" -m workflows.sbi.shared_phase_migration inventory --output "$out/inventory.json"
"$py" -m workflows.sbi.shared_phase_migration verify --phase ph001 --kind ph001-truth --output "$out/ph001_truth_verified.json"
"$py" -m workflows.sbi.shared_phase_migration verify --phase ph000 --kind p12-phase --output "$out/ph000_phase_receipts_verified.json"
"$py" workflows/abacus_tweb/p10_expansion_preflight.py --registry configs/p10_phase_registry_v3.json --adapt-density ph014 --output "$out/ph014_density_adapter.json"
"$py" workflows/abacus_tweb/p10_run_tweb.py --registry configs/p10_phase_registry_v3.json --phase ph014 --density /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1/matter/ph014/counts_tsc_2048.npy --density-manifest "$out/ph014_density_adapter.json" --preflight-only
printf '%s\n' 'Reuse verification and density preflight completed; no T-Web build or training performed.'
