#!/usr/bin/env bash
# Resume after passed PHOTSYS catalogue/field/selection products; do not rebuild them.
set -euo pipefail

repo="/global/homes/d/dkololgi/TNG/Illustris"
root="/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1"
python="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python"

cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run inside an interactive allocation" >&2
  exit 2
fi

"$python" -c '
import json
from pathlib import Path
root = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
product = "bf_proxy_response_v1"
catalogue = json.loads((root / "catalogues" / product / "manifest.json").read_text())
oracle = json.loads(
    (root / "catalogues" / "bf_oracle_assigned_v1" / "manifest.json").read_text()
)
field = json.loads((root / "fields" / product / "manifest.json").read_text())
selection = json.loads(
    (root / "selection" / product / "multitracer_selection_manifest.json").read_text()
)
audit = catalogue["response"]["application_audit"]
assert catalogue["pass"] and field["pass"] and selection["pass"]
assert catalogue["response"]["calibration_basis"] == "DESI LOA PHOTSYS"
assert audit["ambiguous_rows"] == 0
assert audit["overall_fallback_rows"] == oracle["counts"]["FAINT"]["rows"]
assert catalogue["counts"]["FAINT"]["rows"] <= audit["overall_fallback_rows"]
assert audit["north_rows"] + audit["south_rows"] == 0
assert "mock has no PHOTSYS" in audit["mapping"]
for marker in (
    root / "catalogues" / product / "CATALOGUE_COMPLETE",
    root / "fields" / product / "FIELD_OVERLAY_COMPLETE",
    root / "selection" / product / "MULTITRACER_SELECTION_COMPLETE",
):
    assert marker.exists(), marker
'

printf 'job_id=%s\ncommit=%s\n' \
  "$SLURM_JOB_ID" "$(git rev-parse HEAD)" > "$root/MT_PHOTSYS_MARGINAL_PROXY_PRODUCTS_READY"

exec bash workflows/abacus_tweb/run_p8_multitracer_graph_cpu_worker.sh
