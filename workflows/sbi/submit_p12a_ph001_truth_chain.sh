#!/bin/bash
# Submit the immutable authorized ph001 truth-construction dependency chain.
set -euo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
TRUTH_ROOT=/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1
cd "$REPO"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 PATH=/usr/bin:/bin

# This script is intentionally unusable before the exclusive authorization
# marker exists.  It does not create that marker and it never edits the phase
# registry.
"$PY" -u -m workflows.sbi.p12a_authorized_truth guard \
  --stage particle_b --truth-root "$TRUTH_ROOT"
submission_id="truth-$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$PY" -u -m workflows.sbi.p12a_authorized_truth claim-chain \
  --kind truth --submission-id "$submission_id"

particle=$(sbatch --export=NONE --parsable --comment="p12a:$submission_id:particle_b" workflows/sbi/submit_p12a_ph001_particle_b.slurm)
particle=${particle%%;*}
"$PY" -u -m workflows.sbi.p12a_authorized_truth record-chain-job \
  --kind truth --submission-id "$submission_id" --job particle_b --job-id "$particle"
density=$(sbatch --export=NONE --parsable --comment="p12a:$submission_id:density" --dependency="afterok:$particle" workflows/sbi/submit_p12a_ph001_density.slurm)
density=${density%%;*}
"$PY" -u -m workflows.sbi.p12a_authorized_truth record-chain-job \
  --kind truth --submission-id "$submission_id" --job density --job-id "$density" \
  --dependency-job-id "$particle"
tweb=$(sbatch --export=NONE --parsable --comment="p12a:$submission_id:tweb" --dependency="afterok:$density" workflows/sbi/submit_p12a_ph001_tweb.slurm)
tweb=${tweb%%;*}
"$PY" -u -m workflows.sbi.p12a_authorized_truth record-chain-job \
  --kind truth --submission-id "$submission_id" --job tweb --job-id "$tweb" \
  --dependency-job-id "$density"
annotation=$(sbatch --export=NONE --parsable --comment="p12a:$submission_id:annotation" --dependency="afterok:$tweb" workflows/sbi/submit_p12a_ph001_annotation.slurm)
annotation=${annotation%%;*}
"$PY" -u -m workflows.sbi.p12a_authorized_truth record-chain-job \
  --kind truth --submission-id "$submission_id" --job annotation --job-id "$annotation" \
  --dependency-job-id "$tweb"
compact=$(sbatch --export=NONE --parsable --comment="p12a:$submission_id:compact" --dependency="afterok:$annotation" workflows/sbi/submit_p12a_ph001_compact_truth.slurm)
compact=${compact%%;*}
"$PY" -u -m workflows.sbi.p12a_authorized_truth record-chain-job \
  --kind truth --submission-id "$submission_id" --job compact --job-id "$compact" \
  --dependency-job-id "$annotation"

"$PY" -u -m workflows.sbi.p12a_authorized_truth record-chain \
  --kind truth --submission-id "$submission_id"

printf 'particle_b=%s\ndensity=%s\ntweb=%s\nannotation=%s\ncompact_truth=%s\n' \
  "$particle" "$density" "$tweb" "$annotation" "$compact"
