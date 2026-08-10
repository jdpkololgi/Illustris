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

## P8 Deterministic Spatial-Transfer Screens And Recovery

P8 compares finite-context graph (G-PATCH) and field (U-PATCH) encoders against
matched full-cap classical reconstruction on blocked sky folds. The controlling
metric is the equal-weight four-shell macro R² of λ₁ over every authoritative
validation-core galaxy. The first-three-shell score is diagnostic only: a
four-shell macro lead caused solely by classical collapse in the sparsest shell
is not a learned-model adoption win.

There are **three distinct run classes**. Do not mix them:

| Class | Scripts | Sampling | Status |
| --- | --- | --- | --- |
| Short screen | `p8_train_{graph,unet}_patch.py` | Replacement-sampled fixed steps | Frozen optimization-smoke evidence (~15% core exposure) |
| Exposure-aware recovery | `p8_train_patch_recovery.py` (`--run-name recovery_v1`) | Every eligible training core once per epoch | Two-rotation primary complete; artifacts immutable |
| Convergence extension | same trainer (`--run-name convergence_extension_v1`) | Warm-start weights + fresh AdamW/cosine | Rotations 0 and 2 complete with `NOT_CONVERGED_EXTENSION_CAP` |

Use `cosmic_env` in an interactive CUDA allocation. There is no P8 SLURM submit
script; G-PATCH, U-PATCH, CIC FFT, and recovery all reject an unavailable CUDA
device. Clear inherited DESI Python variables before launch:

```bash
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
conda activate cosmic_env
export P8_ROOT=/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1
export P8_RECOVERY_ROOT=/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1
```

### Preparation and short screens

```text
P4 blocked-fold assignment + P5 graph adapter + P6 field adapter
  -> p8_prepare_deterministic.py
  -> p8_prepare_parent_diagnostics.py
  -> p8_prepare_graph_features.py (rotation 0 and 2)
  -> p8_classical_fullcap.py
  -> p8_train_graph_patch.py / p8_train_unet_patch.py
  -> p8_audit_predictions.py
  -> p8_audit_training_adequacy.py
  -> p8_summarize_screens.py
```

```bash
python workflows/abacus_tweb/p8_prepare_deterministic.py --output-root "$P8_ROOT"
python workflows/abacus_tweb/p8_prepare_parent_diagnostics.py --p8-root "$P8_ROOT"
python workflows/abacus_tweb/p8_prepare_graph_features.py --rotation 0 --p8-root "$P8_ROOT"
python workflows/abacus_tweb/p8_prepare_graph_features.py --rotation 2 --p8-root "$P8_ROOT"
python workflows/abacus_tweb/p8_classical_fullcap.py --screen-rotations 0 2 --p8-root "$P8_ROOT"
```

Reproduce the frozen short-screen schedule (not a recovery run):

```bash
python workflows/abacus_tweb/p8_train_graph_patch.py \
  --rotation 0 --steps 2000 --eval-every 2000 --patience 1 \
  --loss-log-every 25 --p8-root "$P8_ROOT"
python workflows/abacus_tweb/p8_train_unet_patch.py \
  --rotation 0 --steps 2000 --eval-every 2000 --patience 1 \
  --loss-log-every 25 --p8-root "$P8_ROOT"
```

Repeat explicitly for rotation 2. Do not rely on model defaults for matched
comparisons: U-PATCH defaults to 4000 steps / eval every 500 / patience 5,
whereas G-PATCH defaults to 2000 / 2000 / 1.

### Exposure-aware recovery (`recovery_v1`)

A recovery epoch visits every eligible P4 training core exactly once
(deterministic weight-proportional without-replacement order). Patch losses keep
the frozen square-root shell objective and are scaled so the arithmetic epoch
mean equals the globally row-weighted MSE (`p8_epoch_training.py`). Scientific
defaults: 5–20 epochs, no early stop before epoch 5, patience 3, min Δmacro-R²
0.005, complete-fold validation after every epoch. Pass
`--run-name recovery_v1` explicitly — the trainer default is `control_v1`.

```bash
python workflows/abacus_tweb/p8_train_patch_recovery.py \
  --model graph --rotation 0 --run-name recovery_v1 \
  --epochs 20 --min-epochs 5 --patience 3 --min-delta 0.005 \
  --p8-root "$P8_ROOT" --output-root "$P8_RECOVERY_ROOT"
python workflows/abacus_tweb/p8_train_patch_recovery.py \
  --model unet --rotation 0 --run-name recovery_v1 \
  --epochs 20 --min-epochs 5 --patience 3 --min-delta 0.005 \
  --p8-root "$P8_ROOT" --output-root "$P8_RECOVERY_ROOT"
```

Outputs land under
`$P8_RECOVERY_ROOT/recovery_v1/{graph,unet}/rotation_<r>/seed_42/`.
Resume after allocation expiry with the **same arguments** plus `--resume`. The
checkpoint freezes Git revision and a wide CLI contract (epochs, lr, architecture,
warm-start paths, roots, etc.): resume from a detached worktree at that SHA
rather than changing code under a live run. Mid-epoch resumes restore sampler
order, cursor, optimizer/scheduler, and partial-epoch loss accumulators;
`reconcile_loss_trace` drops abandoned post-checkpoint loss rows. Audit a
finished or live directory with:

```bash
python workflows/abacus_tweb/p8_audit_recovery_run.py \
  "$P8_RECOVERY_ROOT/recovery_v1/unet/rotation_0/seed_42"
```

Rotations are fold-role rotations, not physical sky rotations. Rotation 0 trains
folds `{2,3,4}` / validates fold 1; rotation 2 trains `{0,1,4}` / validates fold
3. That is same-phase spatial transfer, not a fresh-phase P10 test.

### Convergence extension (`convergence_extension_v1`)

Warm-start from an immutable parent `best_checkpoint.pt`, then reset AdamW and
the cosine schedule. The CLI enforces the registered contract:

- `--run-name convergence_extension_v1`
- `--epochs 20 --lr 2e-4 --disable-early-stopping`
- `--epoch-seed-offset` equal to the parent best epoch
- matching `--model` / `--rotation` / `--seed`
- parent directory must also contain `run_manifest.json` and
  `recovery_checkpoint.pt` (provenance check)

```bash
# Example: U-PATCH rotation 0 (parent best epoch was 20)
python workflows/abacus_tweb/p8_train_patch_recovery.py \
  --model unet --rotation 0 --seed 42 \
  --run-name convergence_extension_v1 \
  --warm-start-checkpoint \
    "$P8_RECOVERY_ROOT/recovery_v1/unet/rotation_0/seed_42/best_checkpoint.pt" \
  --epoch-seed-offset 20 \
  --epochs 20 --lr 2e-4 --disable-early-stopping \
  --p8-root "$P8_ROOT" --output-root "$P8_RECOVERY_ROOT"
```

For G-PATCH rotation 0 use `--epoch-seed-offset 15` against that parent's best
checkpoint. Never overwrite, relabel, or resume-with-altered-args the primary
`recovery_v1` trees. Extension summary statuses include
`NOT_CONVERGED_EXTENSION_CAP` (best in the final three epochs),
`EXTENSION_COMPLETE_NO_MATERIAL_GAIN` (Δ < `--min-delta`), and
`EXTENSION_COMPLETE`. Both registered rotations finished under
`NOT_CONVERGED_EXTENSION_CAP`.

### Evaluation plots and P9 residual audit

Recovery evidence figures (fixed scratch output locations; not portable CLIs):

- `plot_p8_recovery_curves.py` — live/finished learning curves vs frozen bars
- `plot_p8_recovery_parity.py` — eigenvalue parity + sparse-shell diagnosis
- `plot_p8_recovery_visuals.py` — λ fields and T-Web class maps
- `plot_p8_rotation2_eval.py` — rot0/rot2 replication (fig15–17)
- `plot_all_models_parity.py` — pooled parity ladder across completed P8 models

`p9_residual_complementarity_audit.py` fits least-squares blends of CIC /
G-PATCH / U-PATCH on half the validation super-blocks and scores the other half
(both directions). It is a one-shot diagnostic with hardcoded NERSC repo and
recovery paths, not a general CLI. Early same-fold result: CIC+learned hybrid
is a NO-GO (sparse shell destroyed); U+G ensemble gain is small (~+0.012).

### Operational and scientific constraints

- P8 trains on **linear increments**
  `(λ₁, λ₂−λ₁, λ₃−λ₂)` with a cumulative-sum inverse and no post-hoc sorting.
  This is different from the ordered-softplus target used by wedge NPE; see
  `docs/evidence/contracts/p8_target_metric_contract_v1.json`.
- Most scripts accept `--p8-root` / `--output-root`, but several upstream
  P4/P5/P6 inputs still default to project-specific scratch paths. Inspect
  `--help` when reproducing elsewhere.
- G-PATCH short screens write `pre_evaluation_checkpoint.pt` before expensive
  complete-fold assembly; recover with `p8_eval_graph_checkpoint.py` rather than
  restarting training.
- F-tier / U-Physics remains scientifically untested under P8: frozen v2_A was
  `NO_GO_FROZEN_V2_A_RESOURCE_INFEASIBLE` (≥91.6 GiB before autograd).
- Same-phase blocked folds establish spatial-transfer evidence only. P10
  cross-phase testing and multi-seed replication remain required before any
  finalist freeze.

Authoritative numbers live in the newest P8/P9 entries of `SCIENCE_LOG.md`; the
protocol is `docs/plan_generalisable_graphweb_vac.md` §P8/P9; machine-readable
short-screen reports are under `docs/evidence/p8/`.

## P8.8 Multitracer (BGS_FAINT Context) And MT4 Closeout

P8.8 asks whether faint tracers add recoverable spatial information beyond Bright
under the same P4/P8 protocol. Runtime root:

```bash
export P8_MT_ROOT=/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1
```

Typical chain (CPU graph handoff before GPU training):

```text
p8_build_multitracer_catalogues.py
  -> p8_build_multitracer_fields.py / p8_refit_multitracer_selection.py
  -> p8_build_multitracer_graph_adapter.py
  -> p8_prepare_multitracer_graph_features.py
  -> p8_validate_multitracer_global_graph.py
  -> p8_train_multitracer_unet_patch.py / p8_smoke_multitracer_null.py
  -> p8_close_multitracer_mt4.py
```

Interactive supervisors under `workflows/abacus_tweb/run_p8_multitracer_*.sh`
clear DESI `PYTHONPATH` and use the absolute `cosmic_env` Python. Graph-metric
recomputes still need `rapids-gnn`. Do not launch GPU training until the
validated graph handoff markers exist.

Frozen MT4 decision (`docs/evidence/p8/multitracer_mt4_decision.json`):

- multitracer information: `PASS_SAME_PHASE_TWO_ROTATIONS`
- current encoder adoption: `NO_GO_FOLD_INSTABILITY_AND_ROTATION_2_SAFEGUARDS`
- fresh-phase / production claim: `NOT_TESTED`
- next registered model: `U-DENSITY-PHYS-v1` (not MT5 union-graph training)

Scrambled-Faint nulls use `p8_smoke_multitracer_null.py`. Matched classical /
control fields are built by `p8_build_multitracer_control_fields.py` and scored
by `p8_evaluate_multitracer_controls.py` / `p8_multitracer_classical.py`.

## P8.9 Density-First Target Pipeline (`U-DENSITY-PHYS-v1`)

P8.9 registers a Bright-only density objective: predict R=7 Mpc/h smoothed matter
contrast `delta_R7` on P3 cores, stitch owned voxels, then apply one global fixed
FFT tidal solve per cap. The matter field is **privileged supervision only** and
must never enter a DESI model input. There is **no density trainer entrypoint
yet**; current code closes target coordinates, builds targets, and freezes
exact-owner tiling.

```text
p8_density_target_alignment.py          # coordinate gate
  -> p8_build_density_targets.py        # cap-aligned delta_R7 + science_support
  -> p8_build_field_output_tiling.py    # exact half-open owners + inference-only cores
  -> (blocked) train U-DENSITY-PHYS-v1  # after corrected rebuild + trace/tensor closure
```

Environment and interactive supervisors:

```bash
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export P8_DENSITY_ROOT=/pscratch/sd/d/dkololgi/abacus/p8_density_phys_v1

# CPU interactive supervisors (salloc + cosmic_env absolute Python)
bash workflows/abacus_tweb/run_p8_density_target_alignment.sh
bash workflows/abacus_tweb/run_p8_build_density_targets.sh
# tiling is a lighter module invoke (also clears DESI Python vars)
bash workflows/abacus_tweb/run_p8_build_field_output_tiling.sh
```

Or call modules directly after activating `cosmic_env`:

```bash
python -m workflows.abacus_tweb.p8_density_target_alignment
python -m workflows.abacus_tweb.p8_build_density_targets
python -m workflows.abacus_tweb.p8_build_field_output_tiling
```

### Coordinate and unit contract

- P3 lattices are observer-frame **comoving Mpc** (5 Mpc cells). Periodic T-web
  slabs and the observer origin are **Mpc/h**.
- Passing sky mapping: cosmological redshift `Z_COSMO` with origin
  `(-1000,-1000,-1000) Mpc/h`. Host-halo `x_com` reproduces frozen eigenvalues
  exactly and closes label/slab lineage.
- Periodic index mapping:
  `periodic_xyz_Mpc/h = (observer_xyz_Mpc * h + origin_Mpc/h) modulo 2000 Mpc/h`.
- Radial science support uses Planck18 comoving distances in **Mpc**, matching
  P3, not Mpc/h shell limits.
- Observed/RSD `Z` at the same origin is materially worse (especially λ₁). Report
  oracle (`Z_COSMO`) and deployable (observed-`Z`) evaluation rows separately;
  never quote the oracle row as a DESI VAC score.
- Tracked preflight: `docs/evidence/p8/density_target_alignment.json`
  (`pass: true`). Runtime copy under `$P8_DENSITY_ROOT/preflight/`.

### Target fields and support

`p8_build_density_targets.py` samples the already-smoothed R=7 eigenvalue-slab
**trace** onto P3 centres (no second smoothing). It refuses a failing alignment
gate, a non-R=7 slab directory, and an origin other than `-1000`. Outputs:

- `$P8_DENSITY_ROOT/targets/{ngc,sgc}_delta_r7.h5` — `delta_r7`,
  `science_support`, `core_coverage`, `density_loss_support`
- `$P8_DENSITY_ROOT/targets/target_manifest.json`

`density_loss_support = science_support & core_coverage`. Uncovered supported
voxels are counted explicitly and must not be zero-filled or replaced by a
classical field.

**Supersession:** the first runtime target products mixed observer Mpc with
periodic Mpc/h and compared P3 radii in Mpc to shell bounds in Mpc/h. Those
HDF5 files, support counts, and the first tracked
`docs/evidence/p8/density_target_manifest.json` are **not** training-safe.
Rebuild under the unit-corrected source (`9500924` or later) before proceeding.
Unit tests in `tests/phase4/test_p8_build_density_targets.py` guard the mapping
and shell bounds.

### Exact-owner field output tiling

`p8_build_field_output_tiling.py` freezes voxel ownership by P3 cell centre in
exactly one half-open 64 Mpc/h P4 lattice core. It keeps every nominal P4 core
and adds **inference-only** owner cores (fold sentinel 255) wherever
science-supported voxels would otherwise lack an owner. Those rows never own
density loss, labels, galaxy metrics, or model selection; they only let a frozen
model write every supported voxel before the global FFT. The script requires a
passing target manifest and exits non-zero if coverage is incomplete.

Tracked tiling evidence: `docs/evidence/p8/field_output_tiling_manifest.json`
(also provisional until regenerated against the corrected support mask).

### What is still blocked

1. Corrected target rebuild + regenerated tiling/support manifests.
2. Trace/tensor closure at host-`x_com`, `Z_COSMO`, and observed-`Z` locations.
3. Only then may `U-DENSITY-PHYS-v1` train (Bright-only density MSE; no direct
   eigenvalue/tensor loss in D0). Protocol and adoption gates remain in
   `docs/plan_generalisable_graphweb_vac.md` §P8.9.

## Abacus SBI Cache And Wedges

The active Abacus-scale SBI chain is:

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

Use `workflows/sbi/jraph_sbi_flowjax.py` for the TNG/full-graph cache path:

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
