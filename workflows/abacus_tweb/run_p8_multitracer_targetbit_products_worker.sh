#!/usr/bin/env bash
# Rebuild both products with PHOTSYS calibration and assignment-time response bits.
set -euo pipefail

repo="/global/homes/d/dkololgi/TNG/Illustris"
root="/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1"
logs="/pscratch/sd/d/dkololgi/logs"
python="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python"

cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This worker must run inside an interactive allocation" >&2
  exit 2
fi

run_stage() {
  local name="$1"
  shift
  srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=32 \
    env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
    "$@" 2>&1 | tee "$logs/p8_mt_targetbit_${name}_${SLURM_JOB_ID}.log"
}

run_stage catalogue_rebuild \
  "$python" -u -m workflows.abacus_tweb.p8_build_multitracer_catalogues --force

run_stage field_rebuild \
  "$python" -u -m workflows.abacus_tweb.p8_build_multitracer_fields --force

run_stage selection_rebuild \
  "$python" -u -m workflows.abacus_tweb.p8_refit_multitracer_selection --force

run_stage grid_support_audit \
  "$python" -u -m workflows.abacus_tweb.p8_audit_multitracer_grid_support

run_stage validation "$python" -c '
import json
from pathlib import Path
root = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
products = ("bf_oracle_assigned_v1", "bf_proxy_response_v1")
for product in products:
    catalogue = json.loads((root / "catalogues" / product / "manifest.json").read_text())
    field = json.loads((root / "fields" / product / "manifest.json").read_text())
    selection = json.loads(
        (root / "selection" / product / "multitracer_selection_manifest.json").read_text()
    )
    assert catalogue["pass"] and field["pass"] and selection["pass"]
    bits = catalogue["inputs"]["faint_deduplication_audit"]["response_target_bits"]
    assert bits["ambiguous_rows"] == 0
    assert bits["overall_fallback_rows"] == 0
    assert "target-selection bits" in bits["mapping"]
    for marker in (
        root / "catalogues" / product / "CATALOGUE_COMPLETE",
        root / "fields" / product / "FIELD_OVERLAY_COMPLETE",
        root / "selection" / product / "MULTITRACER_SELECTION_COMPLETE",
    ):
        assert marker.exists(), marker
proxy = json.loads((root / "catalogues/bf_proxy_response_v1/manifest.json").read_text())
application = proxy["response"]["application_audit"]
assert proxy["response"]["calibration_basis"] == "DESI LOA PHOTSYS"
assert application["ambiguous_rows"] == 0
assert application["overall_fallback_rows"] == 0
print(json.dumps({"proxy_rows": proxy["total_rows"], "application": application}, indent=2))
'

printf 'job_id=%s\ncommit=%s\n' \
  "$SLURM_JOB_ID" "$(git rev-parse HEAD)" > "$root/MT_TARGETBIT_PROXY_PRODUCTS_READY"
