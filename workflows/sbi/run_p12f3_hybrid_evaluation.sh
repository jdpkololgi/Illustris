#!/bin/bash
set -euo pipefail

repo=${P12F3_REPO:-/global/homes/d/dkololgi/TNG/Illustris}
python=${P12F3_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
method=${P12F3_METHOD:?set P12F3_METHOD}
run_name=${P12F3_RUN_NAME:-f3l_seed42_rngfix_v2}
training_root=${P12F3_TRAINING_ROOT:-/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_hierarchical_lowmode_v1/${run_name}}
output=${P12F3_EVALUATION_ROOT:-/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_hierarchical_lowmode_v1/${run_name}_evaluation}
draw_batch=${P12F3_DRAW_BATCH:-16}
config=${repo}/configs/p12f3_hierarchical_lowmode_v1.json
evaluation_config=${repo}/configs/p12f3_hybrid_evaluation_v1.json

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
cd "${repo}"

"${python}" -m unittest -v \
  tests.phase4.test_p12f3_hierarchical_lowmode \
  tests.phase4.test_p12f3_hybrid_archive

arguments=(
  --config "${config}"
  --method "${method}"
  --output-root "${output}/archives"
  --device cuda
  --draw-batch "${draw_batch}"
  --resume
)
case "${method}" in
  hybrid_local_h8)
    arguments+=(--low-checkpoint "${training_root}/local_h8/checkpoint.pt")
    ;;
  hybrid_wide_h24)
    arguments+=(--low-checkpoint "${training_root}/wide_h24/checkpoint.pt")
    ;;
  g1_wide_crop_h8|g1_wide_h24)
    ;;
  *)
    echo "unsupported P12-F3 evaluation method: ${method}" >&2
    exit 2
    ;;
esac

"${python}" -u -m workflows.sbi.p12f3_export_hybrid_archive "${arguments[@]}"
mkdir -p "${output}/reports"
"${python}" -u -m workflows.sbi.p12f_common_evaluator \
  --config "${evaluation_config}" \
  --archive-manifest "${output}/archives/${method}/P12F_SAMPLE_ARCHIVE.json" \
  --panel-marker "${output}/archives/P12F3_PH006_PANEL_256.json" \
  --output "${output}/reports/${method}.json" \
  --device cuda
