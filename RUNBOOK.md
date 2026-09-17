# TNG/Illustris Runbook

## Controlled VDM diversity/context/multiscale experiment

Live2026-09-17: `tmux attach -t vdm-context-matrix` on login04. Frozen source
1639851; full GPU smoke/restart passed on58472788. First scientific allocation
58473428/nid008513,4GPUs,4h max; subsequent bounded IDs live in resources/*START.
Inspect logs/matrix_controller.log, logs/resource_*.log, and per-task logs/01_*.
Do not start another controller or rerun smoke into this root. Scientific fits
and draws are incomplete; source snapshot must stay immutable throughout.

Full execution: commit tested code, then `context_control stage-run --root ROOT`
to freeze source/data/draw ledger. From ROOT/source, run module
`workflows.sbi.e2e_vdm_context_interactive --mode smoke-controller --root ROOT`
for the one-GPU<=1h technical allocation. Only after SMOKE and RESTART_TEST pass,
use `--mode controller` for the fixed train/freeze/refinement/sampler/main/report
chain. Run the bounded controller in a named tmux shell under the user's explicit
interactive/disconnect preference; all compute is launched with srun. Never use
tmux as a substitute for Slurm limits. Resources/*REQUEST/START/END/ACCOUNTING
and logs/resource_* plus per-worker/task logs are authoritative; no automatic
retry after unexpected exit, scientific gate failure or budget exhaustion.
EXPERIMENT_COMPLETE requires all registered draws, final RESULTS/REPORT and
terminal resource accounting, not merely zero scheduler exit status.

Latest2026-09-17: corrected physical gate58472098 COMPLETED0:0 in44s; v2 passes
all three unchanged25%gates. PHYSICS_V2_RETURN exit0; REPRESENTATION_RELEASE
and A32 NORMALIZATION exist and hashes verify. No allocation remains running.
Do not rerun either physics launcher. Failedv1 remains immutable evidence,
superseded for launch decisions only by the explicit v2 release. Next is full
GPU replay/throughput preflight and bounded controller completion, not an
unconditional training launch. No new fitted models/draws exist yet.

STOP2026-09-17: `data/REPRESENTATION_GATE.json` is an immutable FAILED scientific
gate from58470989. No GPU launch or normalization release. All5phase product
builders completed; no tmux allocation remains running. Do not rerun the single-
use physics launcher or overwrite its gate to resume. See
`docs/e2e_vdm_context_representation_failure_20260917.md`; a separately authorized
operator correction has now been approved: one<=30minCPU test, unchanged gate.
After committed focused tests, stage using context_control `stage-physics-v2`;
launch the frozen context_cpu `--mode physics_v2` in a named tmux shell with
cosmic_env activated and Python paths cleared. This user-directed bounded test
retains the earlier interactive preference; tmux does not extend Slurm walltime.
PHYSICS_V2_REQUEST/RETURN and logs/physics_v2.log record the attempt, with no
automatic retry. Require REPRESENTATION_GATE_V2 and hash-bound
REPRESENTATION_RELEASE before proceeding; original failed receipt stays intact.

Approved contract: `docs/e2e_vdm_context_diversity_v1.md`; current root
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1`.
Geometry job58469111 completed0:0,416 primary anchors, no quota relaxation.
`GEOMETRY_SOURCE.json`, `SCREEN_REQUEST.json`, `SCREEN_RETURN.json` and
`data/GEOMETRY.json` bind its source and outcome. Never rerun its single-use launch.

Training-phase product build58469318 on nid004194, bounded2CPU-nodehours,
launched from `tmux attach -t vdm-context-products` on login04. Fixed source in
`ROOT/source_products`; `BUILD_SOURCE.json` lists all hashes. Inspect
`ROOT/logs/products.log`, `BUILD_RETURN.json` after exit, and per-phase
`data/phNNN/COMPLETE.json` with payload hashes. Existing partial files fail closed;
no overwrite or restart solely because an observation times out. Completion of
this build is not model readiness: require the separately frozen physical gate,
A32 normalization, full-size GPU smoke/replay and revised cost forecast first.

New training entrypoint `workflows.sbi.e2e_vdm_context_train` requires a full-run
MANIFEST and passing SMOKE/REPRESENTATION_GATE, so cannot launch from the current
geometry/product-only snapshots. Durable generation checkpoints commit via
LATEST.json, with optimizer/objective/global RNG state; clean signal stop75.
Full GPU controller/evaluation/report integration is still being implemented.

## Restartable four-GPU VDM assessment

Active launch2026-09-17: `tmux attach -t vdm-assessment-4gpu` on login04;
first allocation58465656/nid001133, four one-GPU steps. Controller log:
`ROOT/logs/interactive_controller.log`. Launcher revisionbc81d1b. The next
allocation ID, if needed, is recorded dynamically in INTERACTIVE_SEGMENT_1_START.

Contract `docs/e2e_vdm_assessment_v1.md`. Root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_assessment_20260917_v2`.
After matching SMOKE and RESTART_TEST receipts, stage the standalone launcher
using `python -m workflows.sbi.e2e_vdm_assessment_interactive --mode stage --root ROOT`.
Run the frozen `ROOT/interactive_launcher.py --mode controller --root ROOT` from
a clean cosmic_env tmux shell on the same login host. This is an explicitly
authorized exception to the usual finalized-workload batch handoff: two75-minute
salloc requests maximum, four GPUs and four independent exclusive srun workers.
Each worker signals a clean chunk stop at70minutes. Only exit75 plus the matching
pause receipt permits the next allocation. No retry on failure/unavailable nodes;
no batch fallback; no new training. tmux preserves the shell, not Slurm wall time.

INTERACTIVE_INTENT/REQUEST/SEGMENT/RETURN receipts record resources and status;
logs/interactive_SEGMENT_JOB_BRANCH.log contains chunk progress. All four verified
ALL_COMPLETE receipts gate the report; `analysis/RESULTS.json` and
INTERACTIVE_COMPLETE.json are the success artifacts. No completed-report claim
from allocation exit alone. Do not rerun stage/controller into an existing launch;
inspect receipts and obtain a new bounded resource decision after budget exhaustion.

## Direct-density conditional VDM pilot

Contract: `docs/e2e_direct_vdm_v1.md`. From frozen source run
`python -m workflows.sbi.e2e_direct_experiment smoke --root ROOT` on approved GPU.
After matching SMOKE.json passes, `submit --root ROOT` submits one shared90-minute
GPU job for the four-fit matrix. Launch that command in a named tmux terminal;
Slurm, not tmux, provides disconnect persistence. SUBMISSION.json records job ID;
logs/train_JOBID.out and per-branch update_*.pt/COMPLETE.json record progress.
No automatic retry/resubmit into a partial branch. Exact checkpoint replay is a
smoke gate; saved interrupted states require an explicitly reviewed resume path.

Submitted58445857 from `tmux attach -t e2e-direct-vdm` on login09, root
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/direct_vdm_20260916_v3`.
Maintenance-pending at launch. Always activate cosmic_env AND unset PYTHONPATH,
PYTHONHOME, PYTHONUSERBASE, LD_PRELOAD inside the tmux shell: its startup can
reintroduce DESI Python3.13 paths even when tmux inherited a clean environment.
Check `squeue -j 58445857`; after termination use sacct plus MATRIX_COMPLETE.json,
four branch COMPLETE.json files and saved draw receipts, not scheduler status alone.

## Frozen neural sampler / first-moment controls

Completed58442539, contract/results `docs/e2e_frozen_controls_v1.md`. Stage with
oracle-conflict CLI; from frozen source run `e2e_frozen_controls --mode optimizer`
or `--mode sampler`, always with --root and approved GPU. Reporting module
`e2e_frozen_controls_report --root ROOT` validates hashes and draws. Saved CUDA
objective RNG state requires CUDA generator during full checkpoint reload;
do not pass CPU generator to base.restore. No source/weights/defaults overwritten.

## Oracle / loss-conflict diagnostics

Activate cosmic_env before graphify and Python. Contract/results:
`docs/e2e_oracle_conflict_v1.md`. Stage committed source with
`python -m workflows.sbi.e2e_oracle_conflict stage --root <new-root>`; on an
approved GPU, from <root>/source use `run --root <root>`. Analytic/reference and
gradient results are separate from trained-model promotion. Follow-on modules
`e2e_oracle_solver` and `e2e_gradient_step_probe` accept the same --root and were
run in a separate frozen snapshot. No original checkpoints or sampler defaults
are modified. Completed diagnostic allocation58439522 is released.

## Preservation-objective pilot

Contract: `docs/e2e_preservation_objective_v1.md`. Commit source, then stage a
new preservation_* Scratch run with `e2e_preservation_experiment stage`; run its
`smoke` command from frozen source on one allocated GPU. `train` and `report`
use source/parent/data checks, durable checkpoints and paired gate evaluation.
The reviewed `submit_e2e_preservation.slurm` expects run root and train/report;
it checks matching SMOKE.json. No batch submission occurs during stage/smoke.
Full matrix: eight one-GPU tasks, throttle2, followed by an afterok CPU report.
Keep logs on Scratch and record actual job IDs if/when a full run is requested.

Authorized launch2026-09-16: array58407655, report58407656. On login39,
`tmux attach -t e2e-preservation` opens the retained launch terminal. To check
without attaching: `squeue -j 58407655,58407656`; after exit use `sacct` plus
branch COMPLETE receipts and analysis/SUMMARY.json. Do not rerun the single-use
launch script to monitor. Jobs survive loss of the tmux session itself.

## Disconnect-safe clean-limit experiments (2026-09-15)

Frozen-source batch launcher: `workflows/sbi/e2e_clean_limit_launch.py`; reviewed
Slurm entrypoint: `workflows/sbi/submit_e2e_clean_limit.slurm`. Follow the gated
stage/smoke/signal-replay/submit sequence in `docs/e2e_clean_limit_20260915.md`.
Check SUBMITTED.json for job IDs, logs/ for durable stdout/stderr, per-branch
LATEST.json/COMPLETE.json for restart/completion, analysis/SUMMARY.json for results.
Do not blindly repeat submission after interruption: inspect the exclusive intent
and Slurm first. Jobs do not require a live SSH connection, but failures/walltime
need explicit recovery; no automatic requeue or E2E restart.

This runbook lists verified workflow entrypoints, launch commands, and common
operational constraints for the TNG/Illustris and Abacus cosmic web pipelines.
For a concise status index, see `ACTIVE_WORKFLOWS.md`.

## Diversity / normalization diagnostic

Training-only controlled experiment: `docs/e2e_diversity_norm_20260915.md`.
In an approved allocation and `cosmic_env`, use the module subcommands:
`python -m workflows.sbi.e2e_diversity_norm prepare --root <new-scratch-root>`,
then `frozen --root <root> --output <root>/FROZEN.json`, and
`train --root <root> --replica 0 --output <root>/replica_0` (likewise replica1).
Each process needs one GPU; use exclusive one-GPU steps for concurrent replicas.
The prepare step verifies source inputs, caches the fixed panel, freezes train-only
moments and performs a full-size gradient smoke. Do not edit hash-bound sources
between preparation and completion. Outputs are exclusive-create, not resumable.
After both replicas and frozen controls finish, run
`python -m workflows.sbi.e2e_diversity_norm_report --root <root> --output <new-report-dir>`.
It verifies all32 checkpoint hashes, field/noise schedules and endpoint probes.
Small tests: `python -m unittest tests.test_e2e_diversity_norm`.
No full E2E/held-out evaluation is included.

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
