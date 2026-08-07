#!/usr/bin/env bash
# Repair the Proxy response and rebuild only products derived from it.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 JOB_ID" >&2
  exit 2
fi

job_id="$1"
repo="/global/homes/d/dkololgi/TNG/Illustris"
root="/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1"
logs="/pscratch/sd/d/dkololgi/logs"
python="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python"
product="bf_proxy_response_v1"

cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1

run_stage() {
  local name="$1"
  shift
  srun --jobid="$job_id" --overlap --exact --nodes=1 --ntasks=1 \
    --cpus-per-task=32 env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE \
    -u LD_PRELOAD "$@" 2>&1 | tee "$logs/p8_mt_proxy_${name}_${job_id}.log"
}

run_stage catalogue_repair \
  "$python" -u -m workflows.abacus_tweb.p8_build_multitracer_catalogues \
  --repair-proxy-from-oracle --force

run_stage field_rebuild \
  "$python" -u -m workflows.abacus_tweb.p8_build_multitracer_fields \
  --products "$product" --force

run_stage selection_rebuild \
  "$python" -u -m workflows.abacus_tweb.p8_refit_multitracer_selection \
  --products "$product" --force

run_stage grid_support_audit \
  "$python" -u -m workflows.abacus_tweb.p8_audit_multitracer_grid_support \
  --products bf_oracle_assigned_v1 "$product"

run_stage validation "$python" -c '
import json
from pathlib import Path
root = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
product = "bf_proxy_response_v1"
catalogue = json.loads((root / "catalogues" / product / "manifest.json").read_text())
field = json.loads((root / "fields" / product / "manifest.json").read_text())
selection = json.loads(
    (root / "selection" / product / "multitracer_selection_manifest.json").read_text()
)
audit = catalogue["response"]["application_audit"]
assert catalogue["pass"] and field["pass"] and selection["pass"]
assert "target-selection bits" in audit["mapping"]
assert audit["ambiguous_rows"] == 0
assert Path(root / "catalogues" / product / "CATALOGUE_COMPLETE").exists()
assert Path(root / "fields" / product / "FIELD_OVERLAY_COMPLETE").exists()
assert Path(root / "selection" / product / "MULTITRACER_SELECTION_COMPLETE").exists()
print(json.dumps({"catalogue_rows": catalogue["total_rows"], "audit": audit}, indent=2))
'

printf 'job_id=%s\ncommit=%s\n' \
  "$job_id" "$(git rev-parse HEAD)" > "$root/MT_TARGETBIT_PROXY_PRODUCTS_READY"
