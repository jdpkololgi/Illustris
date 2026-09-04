#!/bin/bash
# Submit the immutable authorized ph001 truth-construction dependency chain.
set -euo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=${P12A_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
TRUTH_ROOT=${P12A_TRUTH_ROOT:-/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1}
cd "$REPO"

# This script is intentionally unusable before the exclusive authorization
# marker exists.  It does not create that marker and it never edits the phase
# registry.
"$PY" -u -m workflows.sbi.p12a_authorized_truth guard \
  --stage particle_b --truth-root "$TRUTH_ROOT"

particle=$(sbatch --parsable workflows/sbi/submit_p12a_ph001_particle_b.slurm)
particle=${particle%%;*}
density=$(sbatch --parsable --dependency="afterok:$particle" workflows/sbi/submit_p12a_ph001_density.slurm)
density=${density%%;*}
tweb=$(sbatch --parsable --dependency="afterok:$density" workflows/sbi/submit_p12a_ph001_tweb.slurm)
tweb=${tweb%%;*}
annotation=$(sbatch --parsable --dependency="afterok:$tweb" workflows/sbi/submit_p12a_ph001_annotation.slurm)
annotation=${annotation%%;*}
compact=$(sbatch --parsable --dependency="afterok:$annotation" workflows/sbi/submit_p12a_ph001_compact_truth.slurm)
compact=${compact%%;*}

printf 'particle_b=%s\ndensity=%s\ntweb=%s\nannotation=%s\ncompact_truth=%s\n' \
  "$particle" "$density" "$tweb" "$annotation" "$compact"
