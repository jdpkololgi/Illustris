#!/usr/bin/env bash
# Resume from a verified metadata parent; interrupted private partials stay intact.
set -euo pipefail
phase=${1:?phase}
case "$phase" in ph001|ph014|ph015|ph016|ph017) ;; *) exit 2;; esac
: "${SLURM_JOB_ID:?Approved allocation required}"
: "${SLURM_STEP_ID:?Compute step required}"
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
root=/pscratch/sd/d/dkololgi/abacus/p12a_r7_20phase_20261027_v1/$phase
exec > >(tee "$root/catalogue_resident_${SLURM_JOB_ID}.log") 2>&1
"$py" - "$root" "$phase" <<'PY'
import json,sys
from pathlib import Path
from workflows.abacus_tweb.p10_phase_assets import sha256_file
root=Path(sys.argv[1]);phase=sys.argv[2]
p=root/'catalogues'/f'{phase}_bright_parent_metadata_v2.fits'
r=json.loads(Path(str(p)+'.complete.json').read_text())
if r['phase']!=phase or r['target_truth_present'] or r['output']['sha256']!=sha256_file(p):
 raise ValueError('Parent receipt verification failed')
print('Metadata parent fully hash verified',flush=True)
PY
"$py" -u workflows/abacus_tweb/annotate_cutsky_with_tweb_eigs.py --phase "$phase" --qualification "$root/qualification.json" --cutsky "$root/catalogues/${phase}_bright_parent_metadata_v2.fits" --tweb-dir "$root/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7" --output-dir "$root/catalogues" --output-name "${phase}_bright_parent_galaxy_realspace_v2_tweb.fits" --field-in-memory
"$py" -u workflows/abacus_tweb/p10_build_observed_truth.py --registry configs/p10_phase_registry_v3.json --phase "$phase" --annotated-parent "$root/catalogues/${phase}_bright_parent_galaxy_realspace_v2_tweb.fits" --output "$root/catalogues/${phase}_observed_galaxy_realspace_v2_tweb.fits"
"$py" -u workflows/abacus_tweb/p10_build_phase_index.py --registry configs/p10_phase_registry_v3.json --phase "$phase" --observed "$root/catalogues/${phase}_observed_galaxy_realspace_v2_tweb.fits" --out-dir "$root/p1_canonical"
