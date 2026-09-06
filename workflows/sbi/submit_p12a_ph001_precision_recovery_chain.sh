#!/bin/bash
# Submit the authorized precision-only compact recovery and frozen evaluation dispatcher.
set -euo pipefail

P12_REPO=/global/homes/d/dkololgi/TNG/Illustris
P12_PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd "$P12_REPO"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 PATH=/usr/bin:/bin

P12_SUBMISSION_ID="precision-$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$P12_PY" -u -m workflows.sbi.p12a_compact_precision_recovery claim-resume \
  --submission-id "$P12_SUBMISSION_ID" --failed-job 57928446 \
  --blocked-dispatcher 57928546

P12_RECOVERY_JOB=$(sbatch --export=NONE --parsable \
  --comment="p12a:$P12_SUBMISSION_ID:compact_precision" \
  workflows/sbi/submit_p12a_ph001_compact_precision_recovery.slurm)
P12_RECOVERY_JOB=${P12_RECOVERY_JOB%%;*}
"$P12_PY" -u -m workflows.sbi.p12a_compact_precision_recovery record-resume-job \
  --submission-id "$P12_SUBMISSION_ID" --job compact_precision_recovery \
  --job-id "$P12_RECOVERY_JOB"

P12_DISPATCH_JOB=$(sbatch --parsable --account=desi --qos=cron --constraint=cron \
  --cpus-per-task=1 --mem=4G --time=00:10:00 --licenses=scratch --export=NONE \
  --job-name=p12a_p1_postchain --dependency="afterok:$P12_RECOVERY_JOB" \
  --output=/pscratch/sd/d/dkololgi/logs/p12a_postopen_dispatch_%j.out \
  --error=/pscratch/sd/d/dkololgi/logs/p12a_postopen_dispatch_%j.err \
  --wrap="exec /bin/bash $P12_REPO/workflows/sbi/submit_p12a_ph001_postopen_chain.sh")
P12_DISPATCH_JOB=${P12_DISPATCH_JOB%%;*}
"$P12_PY" -u -m workflows.sbi.p12a_compact_precision_recovery record-resume-job \
  --submission-id "$P12_SUBMISSION_ID" --job postopen_dispatch \
  --job-id "$P12_DISPATCH_JOB" --dependency-job-id "$P12_RECOVERY_JOB"
"$P12_PY" -u -m workflows.sbi.p12a_compact_precision_recovery record-resume \
  --submission-id "$P12_SUBMISSION_ID"

printf 'compact_precision_recovery=%s\npostopen_dispatch=%s\n' \
  "$P12_RECOVERY_JOB" "$P12_DISPATCH_JOB"
