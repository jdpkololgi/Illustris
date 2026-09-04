#!/bin/bash
# Submit the immutable finalize -> energy score -> evaluate -> plot chain.
set -euo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd "$REPO"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 PATH=/usr/bin:/bin

# The compact truth job already freezes and deeply validates the terminal marker.
# Chain claiming performs a lightweight validation of that immutable marker and
# compact truth array; it deliberately does not re-hash multi-hundred-GB density
# and T-web intermediates on a login node.
submission_id="postopen-$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$PY" -u -m workflows.sbi.p12a_authorized_truth claim-chain \
  --kind postopen --submission-id "$submission_id"

finalize=$(sbatch --export=NONE --parsable --comment="p12a:$submission_id:finalize" workflows/sbi/submit_p12a_ph001_finalize.slurm)
finalize=${finalize%%;*}
"$PY" -u -m workflows.sbi.p12a_authorized_truth record-chain-job \
  --kind postopen --submission-id "$submission_id" --job finalize --job-id "$finalize"
energy=$(sbatch --export=NONE --parsable --comment="p12a:$submission_id:energy_score" --dependency="afterok:$finalize" workflows/sbi/submit_p12a_ph001_energy_score.slurm)
energy=${energy%%;*}
"$PY" -u -m workflows.sbi.p12a_authorized_truth record-chain-job \
  --kind postopen --submission-id "$submission_id" --job energy_score --job-id "$energy" \
  --dependency-job-id "$finalize"
evaluate=$(sbatch --export=NONE --parsable --comment="p12a:$submission_id:evaluate" --dependency="afterok:$energy" workflows/sbi/submit_p12a_ph001_evaluate.slurm)
evaluate=${evaluate%%;*}
"$PY" -u -m workflows.sbi.p12a_authorized_truth record-chain-job \
  --kind postopen --submission-id "$submission_id" --job evaluate --job-id "$evaluate" \
  --dependency-job-id "$energy"
plot=$(sbatch --export=NONE --parsable --comment="p12a:$submission_id:plot" --dependency="afterok:$evaluate" workflows/sbi/submit_p12a_ph001_plot.slurm)
plot=${plot%%;*}
"$PY" -u -m workflows.sbi.p12a_authorized_truth record-chain-job \
  --kind postopen --submission-id "$submission_id" --job plot --job-id "$plot" \
  --dependency-job-id "$evaluate"

"$PY" -u -m workflows.sbi.p12a_authorized_truth record-chain \
  --kind postopen --submission-id "$submission_id"
printf 'finalize=%s\nenergy_score=%s\nevaluate=%s\nplot=%s\n' \
  "$finalize" "$energy" "$evaluate" "$plot"
