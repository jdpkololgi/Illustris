# TNG/Illustris Runbook

This runbook lists verified workflow entrypoints, launch commands, and common
operational constraints for the TNG/Illustris and Abacus cosmic web pipelines.
For a concise status index, see `ACTIVE_WORKFLOWS.md`.

## Environment Setup

Activate an environment before running repository scripts or tests. The default
for this codebase is `cosmic_env`:

```bash
source ~/.bashrc
conda activate cosmic_env
```

Use `cosmic_env` for T-Web annotation, graph construction/subsetting, cache
building, Jraph/SBI training, GCN workflows, plotting, tests, and normal
diagnostics.

Use the RAPIDS/cuGraph `rapids-gnn` environment whenever calculating graph
metrics/features:

```bash
source ~/.bashrc
unset PYTHONPATH PYTHONHOME LD_PRELOAD
source /global/homes/d/dkololgi/miniforge3/bin/activate "${ABACUS_RAPIDS_ENV_PATH:-/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn}"
```

This applies to `workflows/abacus_tweb/abacus_graph_features_cugraph.py`,
`workflows/abacus_tweb/abacus_graph_features.py`, and any graph-metric
recomputation. The cuGraph SLURM launcher uses the same
`ABACUS_RAPIDS_ENV_PATH` default in
`workflows/abacus_tweb/submit_abacus_graph_features_cugraph.slurm`.

Other setup notes:

- DESI table/catalog tools may require `desienv`.
- JAX GPU jobs usually set:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

## Path Configuration

Shared defaults live in `shared/config_paths.py` and
`shared/tng_pipeline_paths.py`. Override them with environment variables instead
of editing scripts.

| Variable | Default / purpose |
| --- | --- |
| `TNG_ILLUSTRIS_PROJECT_DIR` | Repository path; SLURM defaults to `/global/homes/d/dkololgi/TNG/Illustris`. |
| `DK_SCRATCH_ROOT` | Scratch root, default `/pscratch/sd/d/dkololgi`. |
| `TNG_SCRATCH_ROOT` | Canonical workflow scratch root under `DK_SCRATCH_ROOT`. |
| `TNG_CANONICAL_CACHE_ROOT` | Canonical cache root, used by TNG/Jraph/SBI helpers. |
| `TNG_CANONICAL_OUTPUT_ROOT` | Canonical output root for model artifacts and logs. |
| `TNG_LOG_DIR` | SLURM log directory. |
| `TNG_ABACUS_BASE` | AbacusSummit simulation base path. |
| `TNG_MOCKS_BASE` | DESI SecondGenMocks base path. |
| `TNG_ABACUS_TWEB_OUTPUT_DIR` | Slabwise Abacus T-Web output directory. |
| `TNG_ABACUS_MOCKS_WITH_EIGS_DIR` | Annotated CutSky output directory. |
| `TNG_CUTSKY_Z0200_PATH` | Default z=0.200 BGS CutSky FITS input. |
| `TNG_JRAPH_CACHE_DIR`, `TNG_JRAPH_OUTPUT_DIR` | Jraph-specific cache/output overrides. |
| `TNG_SBI_CACHE_DIR`, `TNG_SBI_OUTPUT_DIR` | Full-graph SBI cache/output overrides. |

## Abacus T-Web And Mock Annotation

The Abacus path builds T-Web labels in the simulation cube, then annotates
CutSky mock galaxies using host-halo linkage.

Batch launch for slabwise T-Web:

```bash
sbatch workflows/abacus_tweb/submit_abacus_tweb_cpu.slurm
```

Direct entrypoints:

```bash
python workflows/abacus_tweb/abacus_cactus_tweb.py --help
python workflows/abacus_tweb/annotate_cutsky_with_tweb_eigs.py --help
python workflows/abacus_tweb/abacus_process_particles2.py --show-workflow
```

Important constraints:

- `annotate_cutsky_with_tweb_eigs.py` maps `(FILE_NUM, HALO_INDEX)` to host-halo
  box-frame positions and then to T-Web voxels. This avoids assigning labels by
  naive sky-coordinate inversion or periodic modulo into one cube.
- The annotated FITS should contain `CWEB`, `LAMBDA1`, `LAMBDA2`, and `LAMBDA3`
  before downstream graph/cache builders consume it.
- Alignment and leakage diagnostics are documented in
  `workflows/abacus_tweb/ABACUS_TWEB_AUDIT_FINDINGS.md`.

## Abacus Mock Graph Construction

The graph builder uses observed CutSky coordinates (`RA`, `DEC`, observed `Z`)
converted to Planck18 comoving Cartesian coordinates. In catalog mode it applies
the DESI BGS mock selection `(IN_Y1 | IN_Y5)` and `R_MAG_APP < 19.5`, excludes
`BOX_INDEX == -1` by default, and builds separate north/south Galactic
hemisphere alpha complexes to avoid long edges across the survey mask.

Batch launch:

```bash
sbatch workflows/abacus_tweb/submit_abacus_graph_cpu.slurm
```

Inspect options:

```bash
python workflows/abacus_tweb/build_abacus_graph.py --help
```

Build alpha-pruned graph artifacts:

```bash
python workflows/abacus_tweb/build_abacus_graph.py \
  --catalog-path "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb.fits" \
  --mode alpha \
  --boxsize-mpc 2000.0 \
  --output-dir "/pscratch/sd/d/dkololgi/abacus/graph_constructions" \
  --output-prefix abacus_alpha
```

Build Delaunay-equivalent artifacts:

```bash
python workflows/abacus_tweb/build_abacus_graph.py \
  --catalog-path "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb.fits" \
  --mode delaunay \
  --output-dir "/pscratch/sd/d/dkololgi/abacus/graph_constructions" \
  --output-prefix abacus_delaunay
```

The builder writes a metadata manifest plus arrays such as:

- `<prefix>_points.npy`
- `<prefix>_points_xyz.npy`
- `<prefix>_edges_combined_idx.npy`
- `<prefix>_tetrahedra_idx.npy`
- `<prefix>_tetrahedra_volumes.npy`

Operational constraints:

- `build_abacus_graph.py` enforces a CPU SLURM allocation unless explicitly
  configured for tiny smoke tests.
- Full-run Gudhi alpha/Delaunay construction is memory heavy. Login-node
  execution is expected to fail for all-points catalogs.

## Abacus Graph Features

CPU feature extraction uses Networkit-style metrics:

```bash
sbatch workflows/abacus_tweb/submit_abacus_graph_features_cpu.slurm
python workflows/abacus_tweb/abacus_graph_features.py --help
```

GPU/cuGraph feature extraction writes parquet tables and GNN-ready arrays:

```bash
sbatch workflows/abacus_tweb/submit_abacus_graph_features_cugraph.slurm
python workflows/abacus_tweb/abacus_graph_features_cugraph.py --help
```

The cuGraph path defaults to the RAPIDS environment at
`/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn`, overrideable with
`ABACUS_RAPIDS_ENV_PATH`.

## Generalisable GraphWeb Canonical Fields (P3a)

Run development/preprocessing inside a reusable CPU `salloc`; do not use `sbatch`
for this one-off development build. Use the absolute `cosmic_env` Python after
clearing inherited Python variables.

```bash
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python

$PY workflows/abacus_tweb/p3a_audit_units.py \
  --out /pscratch/sd/d/dkololgi/abacus/p3_full_footprint/unit_audit.json
$PY workflows/abacus_tweb/p3a_canary_parity.py
$PY workflows/abacus_tweb/p3a_build_canonical_fields.py --probe-only
$PY workflows/abacus_tweb/p3a_build_canonical_fields.py
$PY workflows/abacus_tweb/p3a_postbuild_validate.py
```

The unit audit is mandatory. Observer-frame graph/U-Net coordinates and lattice
lengths are comoving Mpc. The historical matched cell is 5 Mpc (3.383 Mpc/h for
Planck18), not 5 Mpc/h. The T-Web target smoothing remains 7 Mpc/h.

Authoritative products are under
`/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/`. NGC and SGC use separate
HDF5 lattices. A valid run has passing `unit_audit.json`, `field_manifest.json`,
`validation_report.json`, `postbuild_validation.json`, and `FIELD_COMPLETE`.
Consumers must load the checksummed manifest/schema contract; they must not infer
units or channel order from an unaccompanied HDF5 file.

## Abacus SBI Cache And Wedges (older graph-NPE stack)

The older Abacus-scale graph NPE chain is:

```text
annotated CutSky FITS
  -> graph artifacts
  -> wedge graph artifacts + wedge targets FITS
  -> wedge cuGraph GNN arrays and metadata
  -> SBI cache pickle
  -> FlowJAX NPE on one wedge graph
```

This is **not** the current VAC posterior. New Abacus posterior work uses
P12-A FMPE (next section).

Build a survey-space wedge from a parent graph:

```bash
python workflows/abacus_tweb/subset_abacus_graph_wedge_for_sbi.py \
  --graph-metadata "/pscratch/sd/d/dkololgi/abacus/graph_constructions/abacus_delaunay_metadata.json" \
  --annotated-fits "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb.fits" \
  --out-prefix abacus_delaunay_wedge_ra120_140_dec16p5_26p7_z0p2_0p3 \
  --ra-min 120 --ra-max 140 --dec-min 16.5 --dec-max 26.7 --z-min 0.2 --z-max 0.3
```

Project full-graph cuGraph features onto the induced wedge:

```bash
python workflows/abacus_tweb/subset_cugraph_metrics_for_wedge.py \
  --artifacts-dir "/pscratch/sd/d/dkololgi/abacus/graph_constructions" \
  --full-prefix abacus_delaunay \
  --wedge-prefix abacus_delaunay_wedge_ra120_140_dec16p5_26p7_z0p2_0p3
```

Build an SBI-ready cache from the wedge metadata and targets:

```bash
python workflows/abacus_tweb/build_abacus_sbi_cache.py \
  --gnn-metadata-path "/pscratch/sd/d/dkololgi/abacus/graph_constructions/abacus_delaunay_wedge_ra120_140_dec16p5_26p7_z0p2_0p3_cugraph_gnn_metadata.json" \
  --targets-catalog-path "/pscratch/sd/d/dkololgi/abacus/graph_constructions/abacus_delaunay_wedge_ra120_140_dec16p5_26p7_z0p2_0p3_wedge_targets.fits" \
  --output-cache-path "/pscratch/sd/d/dkololgi/abacus/sbi_caches/processed_jraph_data_mc1e+09_v2_scaled_3_transformed_eig.pkl" \
  --no-apply-y1y5-filter \
  --no-exclude-invalid-box-index \
  --three-targets-only
```

Cache constraints:

- Targets can come from `--targets-catalog-path` FITS or `--targets-npz-path`
  wedge truth arrays.
- `--apply-y1y5-filter` is enabled by default to match graph construction.
- `BOX_INDEX == -1` rows are excluded by default to preserve node/target
  alignment.
- For wedge-target FITS produced by `subset_abacus_graph_wedge_for_sbi.py`, pass
  `--no-apply-y1y5-filter --no-exclude-invalid-box-index`. Those rows are
  already aligned to wedge node order and the compact FITS does not carry the
  full graph-build selection columns.
- The default target mode is ordered softplus eigenvalue increments. Use
  `--no-transformed-eig` only for explicit raw-eigenvalue ablations.
- The output pickle schema includes `graph`, `regression_targets`,
  `regression_targets_raw`, `masks`, `target_scaler`, `eigenvalues_raw`, and
  optional classification labels.

The older graph-partition path (`submit_build_partitions_adaptive.slurm`,
`build_abacus_partition_batches.py`, and `PARTITION_ARTIFACT_SCHEMA.md`) is
legacy. Keep it for reproducing partitioned FlowJAX diagnostics, but do not use
it for new Abacus SBI runs. The current Abacus VAC posterior is P12-A FMPE
(below), not this wedge-cache NPE stack.

## P12-A VAC Posterior (current Abacus production candidate)

P12-A estimates

```text
q(lambda_ordered | U_PATCH_R0_epoch20_OOF_prediction, P3b-R response, H_fid)
```

in ordered-softplus coordinates. The frozen untempered FMPE is
`docs/evidence/p12/P12A_PRODUCTION_CANDIDATE_FROZEN.json`. The scientific
`P12A_CALIBRATION_PASS.json` marker is still absent. Do not start new Abacus
VAC posterior work from `jraph_sbi_flowjax.py`.

### Conditioning contract

Features in `p12_prepare_base_response_dataset.py` (`FEATURE_NAMES`):

| Feature | Meaning |
| --- | --- |
| `base_lambda1/2/3` | Physical OOF U-PATCH eigenvalue predictions |
| `redshift` | Galaxy redshift |
| `log_ntilde_mpc3` | Frozen BRIGHT radial selection density at z (not fibre completeness) |
| `cap_ngc` | NGC vs SGC cap |
| `log1p_random_support_boundary_distance_mpc` | P3b-R random-support boundary distance in comoving Mpc |

Fold ID, superblock ID, phase ID, and independently trained 32-d latents are
**not** model features. Superblock/fold are retained only to keep ph006 width
calibration (folds 0–1) spatially disjoint from the selection report (folds
2–4). `ph001` is refused during dataset/fit/OOF export.

Distance quality bits use 10.3458469 and 20.6916938 Mpc, equal to 7 and 14
Mpc/h at Planck18 `h=0.6766`. The legacy bit name
`response_outside_training_range` is actually `log(ntilde_mpc3)` OOD, not a
full survey-response flag.

### Fit and supervisor (already complete)

```bash
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python

$PY workflows/sbi/p12_prepare_crossfit_contracts.py --help
$PY workflows/sbi/p12_export_unet_summaries.py --help
$PY workflows/sbi/p12_prepare_base_response_dataset.py --help
$PY workflows/sbi/p12_train_base_response_fmpe.py --help
```

The persistent supervisor `workflows/sbi/run_p12a_posterior_interactive.sh`
waits for all six OOF summaries (`ph000`, `ph002`–`ph006`) and a free
interactive GPU slot, then runs a canary plus the full dataset/fit. It stops
when `P12A_COMPLETE.json` exists and does **not** run the later audit, affine
canary, or width diagnostic.

Constraints:

- Requires CUDA. `--dataloader-workers` must stay `0` because the training
  `TensorDataset` lives on GPU.
- If `P12A_COMPLETE.json` is missing, the supervisor retries `salloc` at most
  8 times.
- The trainer's `calibration_pass` field is a weaker in-script gate than the
  later physical-eigenvalue audit. Absence of
  `docs/evidence/p12/P12A_CALIBRATION_PASS.json` is the scientific status.
- Paths default to `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/...`.

Post-fit diagnostics (ph006 only; do not promote a new map from them):

```bash
$PY workflows/sbi/p12_calibration_diagnostics.py --help
$PY workflows/sbi/p12_affine_calibration_canary.py --help
$PY workflows/sbi/p12_width_information_diagnostics.py --help
$PY workflows/sbi/p12a_physical_dependence_diagnostic.py --help
```

The affine location-scale canary was rejected: both crossfit scores worsened
and the folds-2–4 physical log-score delta was negative with a spatial 95%
interval below zero. Rank flattening alone is not adoption. Evidence:
`docs/evidence/p12/P12A_AFFINE_CALIBRATION_CANARY.json`.

### Blind opening (one shared ph001 opening)

The opening is two exclusive state changes in `p12a_open_blind.py`:

1. `authorize` — revalidates frozen truth-free predictions and
   `docs/evidence/p12/P12A_BLIND_EVALUATION_CONTRACT.json`, then consumes
   `open_count=1` by creating `P12_BLIND_OPEN_AUTHORIZED.json` **before** any
   ph001 truth is read.
2. `finalize` — runs only after `P12A_PH001_TRUTH_COMPLETE.json` exists, then
   writes `P12_BLIND_OPENED.json`.

Both markers use `O_EXCL`. A failed truth build cannot obtain a second
opening. Isolated truth is written under
`/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1`, not the
ordinary `p10_multiphase/ph001` product tree.

Authorized truth chain (requires the exclusive authorization marker):

```bash
bash workflows/sbi/submit_p12a_ph001_truth_chain.sh
# particle_b -> density -> tweb -> annotation -> compact
```

Post-open evaluation chain (after compact/terminal truth):

```bash
bash workflows/sbi/submit_p12a_ph001_postopen_chain.sh
# finalize -> energy_score -> evaluate -> plot
```

Truth-free four-GPU export (before opening) uses
`submit_p12a_blind_export.slurm`. It requires `sbi==0.26.1`, CUDA, and the
frozen candidate/plan/context/checkpoint. Clear `PYTHONPATH` /
`PYTHONHOME` / `PYTHONUSERBASE` / `LD_PRELOAD` first.

Stop after the immutable pass/fail report. P13/Loa is not authorized by a
passing or failing P12-A report.

### Compact-truth precision recovery

Job `57928446` failed compact join because one of 4,897,905 rows has an
eigenvalue equal to `float32(0.2)` (stored value 0.20000000298023224). NumPy
`float32` comparison rounds the 0.2 threshold to the same value; the native
CACTUS `CWEB` label was computed in higher precision. This is a validation
bug, not a posterior failure.

The authorized exception permits **only** that one-row boundary ambiguity.
It keeps stored eigenvalues and native `CWEB` labels unchanged, does not
refit or rescore, and preserves `open_count=1`. Implementation:
`p12a_compact_precision_recovery.py` plus
`stored_class_consistency()` in `p10_target_contract.py`.

```bash
$PY -m workflows.sbi.diagnose_p12a_compact_closure --output /path/to/diagnostic.json
$PY -m workflows.sbi.p12a_compact_precision_recovery validate
# Production resume (exclusive claim; no automatic retry):
bash workflows/sbi/submit_p12a_ph001_precision_recovery_chain.sh
```

Do not edit the frozen original compact builder to “fix” this. Receipts live
under `docs/evidence/p12/p12a_blind_opening_20260905/`.

## P12-F3-D2 Field Diffusion (parallel, non-blocking)

D2 is a bounded ph006 field-diffusion experiment. It must not read `ph001`,
delay P12-A, or change P12-A gates. The frozen canary selected `modern_base4`;
`modern_base8` improved paired energy by ~0.44% (below the 1% materiality
threshold) and attention remains unlicensed.

Run stages from an existing one-GPU allocation:

```bash
bash workflows/sbi/run_p12f3_d2_in_allocation.sh
# usage: test|contract|matched-references|transform-roundtrip|gpu-smoke|
#        a0|a1|select-capacity|a2|select-final|confirm|
#        science ARM|export ROLE NFE ARM|evaluate ROLE NFE ARM|...
```

The official science wrapper
`workflows/sbi/submit_p12f3_d2_training_fullwall.slurm` uses the pinned
worktree `/global/u2/d/dkololgi/TNG/Illustris_d2_467f442` at commit
`467f442` and a 13,800 s operational soft stop (the interactive launcher
stops at 6,500 s). Changing that wrapper hash fails
`dispatch_p12f3_d2_after_primary.py`. Confirmation needs more than one hour;
the first 1 h allocation timed out without a scientific failure.

Post-primary dispatch submits only already-licensed second-seed or sampler
diagnostics. Exclusive claims forbid duplicate or automatic retries.

## TNG / Wedge FlowJAX Training (older graph-NPE stack)

Use `workflows/sbi/jraph_sbi_flowjax.py` for the TNG/full-graph cache path and
for older Abacus wedge-subvolume caches. This is not the current VAC
posterior (that is P12-A FMPE above).

```bash
python workflows/sbi/jraph_sbi_flowjax.py --help
export TNG_SBI_CACHE_DIR="/pscratch/sd/d/dkololgi/abacus/sbi_caches"
python workflows/sbi/jraph_sbi_flowjax.py --epochs 1000 --output_dir "/pscratch/sd/d/dkololgi/outputs/sbi_wedge"
```

There is not yet a tracked production `sbatch` launcher for wedge NPE. Run it
inside an appropriate GPU allocation until one is added.

Legacy partitioned FlowJAX entrypoints are still available for diagnostics:

```bash
python workflows/sbi/jraph_sbi_flowjax_partitioned.py --help
python workflows/sbi/benchmark_partition_data_parallel.py --help
```

`workflows/sbi/ABACUS_SBI_DEBUG_STRATEGY.md` records partition alignment checks,
tiny-overfit diagnostics, and the legacy learning diagnostics that motivated the
wedge path.

## Jraph Regression And Classification

Batch launch:

```bash
sbatch workflows/jraph/submit_jraph.slurm
```

Other launchers:

```bash
sbatch workflows/jraph/debug_jraph.slurm
sbatch workflows/jraph/submit_tuning.slurm
sbatch workflows/jraph/train_ensemble.slurm
```

Direct entrypoints:

```bash
python workflows/jraph/jraph_pipeline.py --help
python workflows/jraph/jraph_regression_eval_from_checkpoint.py --help
python workflows/jraph/jraph_classification_eval_from_checkpoint.py --help
```

The regression pipeline trains on ordered softplus eigenvalue increments by
default and converts back to physical eigenvalues only for evaluation/plotting.
Raw-eigenvalue modes and shape/invariant conversions are retained for legacy
caches and controlled ablations.

## GCN Paper Workflow

Batch launch:

```bash
sbatch workflows/gcn_paper/submit_gcn.slurm
```

Direct:

```bash
python workflows/gcn_paper/gcn_pipeline.py --help
python workflows/gcn_paper/gcn_pipeline_postprocess.py --help
python workflows/gcn_paper/postprocessing.py --help
```

This workflow is retained for paper reproduction and uses PyTorch/Torch
Geometric classification utilities under `workflows/gcn_paper/`.

## Smoke Tests

Run the lightweight Phase 4 tests from the repository root:

```bash
python -m unittest discover -s tests/phase4
```

These tests cover import compatibility, cache-schema helpers, and help output
for selected entrypoints. Full scientific validation still requires the
Perlmutter data products and SLURM workflows above.

## Compatibility Notes

- Prefer canonical `workflows/...` and `shared/...` paths for new commands.
- Some root-level Python shims remain for back-compat imports, but root-level
  SLURM wrappers have been archived or removed.
- SLURM scripts may call absolute NERSC paths such as
  `/global/homes/d/dkololgi/TNG/Illustris/...`; override
  `TNG_ILLUSTRIS_PROJECT_DIR` where the script supports it.
