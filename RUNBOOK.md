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

## Generalisable GraphWeb P12-A Posterior

P12-A is the current Abacus VAC uncertainty model: a per-galaxy FMPE posterior
over ordered tidal eigenvalues, conditioned on the three-dimensional OOF
U-PATCH base prediction plus deployable response covariates. It is **not** a
joint tidal-field posterior and **not** the older wedge-graph FlowJAX NPE.

Estimand:

```text
q(λ | λ̂_U-PATCH^OOF, z, ñ(z), cap, random-support boundary distance, H_fid)
```

Targets are ordered softplus increments (`λ1` plus `gap12`/`gap23`); invert to
physical `(λ1 ≤ λ2 ≤ λ3)` only at evaluation. Phase, fold, superblock, and
artificial fold-boundary distance are never features. `ph001` is sealed.

Default scratch root (hardcoded in the P12 scripts):

```text
/pscratch/sd/d/dkololgi/abacus/p10_multiphase
```

### Chain

OOF export and FMPE fit (what the supervisor runs):

```text
leave-one-phase-out U-PATCH contracts
  -> p12_export_unet_summaries.py          # OOF summaries; GPU
  -> p12_prepare_base_response_dataset.py  # 2e6 train / 6e5 ph006 rows
  -> p12_train_base_response_fmpe.py       # GPU; writes P12A_COMPLETE.json
```

Post-fit diagnostics (not in the supervisor; GPU; frozen uncorrected posterior):

```text
p12_calibration_diagnostics.py            # physical ranks + TARP
p12_affine_calibration_canary.py          # challenger; REJECTED 2026-08-30
p12_width_information_diagnostics.py      # uses frozen audit draws
```

Persistent supervisor (login-safe watcher; heavy work only on interactive GPU):

```bash
# tmux session p12a_posterior; flock-locked against duplicate watchers
bash workflows/sbi/run_p12a_posterior_interactive.sh
```

It waits until `p12_oof_summaries/{ph000,ph002–ph006}/OOF_SUMMARY_COMPLETE.json`
exist and fewer than two user allocations are submitted, then `salloc`s one
HBM80 GPU node. It does **not** run the calibration audit, affine canary, or
width diagnostic; those are separate GPU commands after `P12A_COMPLETE.json`
exists. Direct commands (inside `cosmic_env`, GPU allocation):

```bash
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
OUT=$ROOT/p12a_base_response_v1

$PY workflows/sbi/p12_prepare_base_response_dataset.py --output-root "$OUT"
$PY workflows/sbi/p12_train_base_response_fmpe.py \
  --dataset-root "$OUT" --output-root "$OUT/fmpe_seed42"
$PY workflows/sbi/p12_calibration_diagnostics.py \
  --dataset-root "$OUT" --output-root "$OUT/fmpe_seed42/calibration_audit_v1"
$PY workflows/sbi/p12_width_information_diagnostics.py \
  --dataset-root "$OUT" \
  --audit-root "$OUT/fmpe_seed42/calibration_audit_v1"
```

`--dataloader-workers` must stay `0`: FMPE keeps the training TensorDataset on
CUDA, and a forked DataLoader re-initialises CUDA.

### Markers (do not confuse)

| Marker | Meaning |
| --- | --- |
| `P12A_DATASET_READY.json` | Dataset contract passed; sealed phase not opened. |
| `P12A_COMPLETE.json` | Fit + evaluation ran (`technical_complete`). |
| `P12A_CALIBRATION_PASS.json` | Trainer's own coverage/SBC/TARP gates. **Absent** for the frozen fit. |
| `P12A_CALIBRATION_AUDIT.json` | Authoritative physical-eigenvalue audit. |
| `P12A_AFFINE_CORRECTION_REJECTED.json` | Affine challenger failed proper-score/crossfit gates. |

Scientific status (2026-08-30): keep the **uncorrected** posterior; widths
adapt to information; sparse-shell λ2/λ3 residual is mild miscentring, not a
reason to revive the affine map. Repository evidence:
`docs/evidence/p12/`. Figures: `docs/figures/p12_calibration_audit_20260830/`
and `docs/figures/p12_width_information_20260830/`.

Fold contract on ph006: folds 0–1 calibration-only, folds 2–4 selection-only.
Natural-volume weights undo sqrt-count shell sampling.

### Pitfalls

- GPU required for FMPE fit, posterior sampling, calibration audit, and affine
  canary. The scripts `raise RuntimeError` without CUDA.
- Do not open `ph001`. Dataset/train/audit/canary/width all refuse it.
- Do not treat the trainer's `calibration_pass` field as the scientific
  calibration verdict. The audit + affine canary programme is authoritative.
- Do not promote the affine map because rank histograms look flatter. Promotion
  requires every gate in `selection_gates()`, including out-of-fit proper log
  score with spatial-block 95% CI above zero.
- The supervisor re-runs canary then full fit whenever `P12A_COMPLETE.json` is
  missing (bounded to 8 `salloc` attempts). If a partial fit died after the
  canary, expect that cost again unless you invoke the Python entrypoints
  directly.
- Realised BRIGHT neighbour counts are an **external** width diagnostic, not
  P12-A features. P1/P3 coordinates are comoving Mpc; the width script converts
  to Mpc/h with Planck18 `h=0.6766` before the 7/10/20 Mpc/h queries.
- Tests: `tests/phase4/test_p12_base_response.py`,
  `test_p12_calibration_diagnostics.py`, `test_p12_affine_calibration.py`,
  `test_p12_width_information_diagnostics.py`. Full sampling still needs NERSC
  products and a GPU.

## P11 Factorial Observation Views

P11 materialises truth-free dense/assigned/final count fields on canonical P3
grids. It is a JEPA/observation-operator substrate, not a posterior and not a
replacement for P12-A calibration.

```text
configs/p11_factorial_views_v1.json
  -> p11_prepare_factorial_view_sources.py   # login-safe manifests only
  -> p11_build_factorial_view_counts.py      # CPU interactive; heavy FITS/HDF5
```

Axes: observation stage `V_dense` / `V_assign` / `V_final`; tracer BRIGHT-only
(production default) vs BRIGHT+FAINT context; stochastic response
(`tileloc_correlated_thinning` is the held-out degradation recipe). `V_final`
FAINT is an identity reference to supported `V_assign` FAINT under the current
mock `C_z=1` contract — not a Loa pointwise product.

`ph001` is sealed. `ph000` is excluded from this branch only (canonical final
BRIGHT is legacy `path1_fiberassign` while dense/assigned sources are
`altmtl0`; not TARGETID-nested). P10/P12 still use `ph000`.

The CPU supervisor waits for `P12A_COMPLETE.json` and an allocation slot:

```bash
bash workflows/abacus_tweb/run_p11_factorial_view_counts_interactive.sh
```

Login-safe source freeze, then heavy build inside a CPU `salloc`:

```bash
$PY workflows/abacus_tweb/p11_prepare_factorial_view_sources.py
$PY workflows/abacus_tweb/p11_build_factorial_view_counts.py
```

`--phase ph00N` builds one phase and does **not** write the all-phase
`FACTORIAL_VIEW_PRODUCTS_READY.json` marker. Omit `--phase` for the full
`ph002–ph006` product. `--force` rebuilds an existing phase.

Nesting gate: voxelwise `V_final Bright ≤ V_assign Bright ≤ V_dense Bright`
(and the assigned/dense FAINT analogue) on the common random-derived support.
The builder reads no T-Web targets.

## Abacus SBI Cache And Wedges

This is the older Abacus **wedge-graph FlowJAX NPE** chain (one RA/Dec/z
wedge graph). It is not the P12-A VAC posterior above.

The wedge-graph SBI chain is:

```text
annotated CutSky FITS
  -> graph artifacts
  -> wedge graph artifacts + wedge targets FITS
  -> wedge cuGraph GNN arrays and metadata
  -> SBI cache pickle
  -> FlowJAX NPE on one wedge graph
```

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
it for new Abacus SBI runs.

## SBI FlowJAX Training

For the current Abacus VAC posterior, use P12-A
(`workflows/sbi/p12_train_base_response_fmpe.py`), not this trainer.

`workflows/sbi/jraph_sbi_flowjax.py` remains the TNG/full-graph and older
Abacus wedge-graph NPE trainer:

```bash
python workflows/sbi/jraph_sbi_flowjax.py --help
```

Use the same trainer for current Abacus wedge-subvolume caches. The trainer
resolves its input through `TNG_SBI_CACHE_DIR` and expects the cache filename
shown in the cache example above:

```bash
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

These tests cover import compatibility, cache-schema helpers, P11/P12
contracts, and help output for selected entrypoints. Full scientific validation
still requires the Perlmutter data products and SLURM workflows above. P12
posterior sampling tests that need CUDA will fail in CPU-only environments.

## Compatibility Notes

- Prefer canonical `workflows/...` and `shared/...` paths for new commands.
- Some root-level Python shims remain for back-compat imports, but root-level
  SLURM wrappers have been archived or removed.
- SLURM scripts may call absolute NERSC paths such as
  `/global/homes/d/dkololgi/TNG/Illustris/...`; override
  `TNG_ILLUSTRIS_PROJECT_DIR` where the script supports it.
