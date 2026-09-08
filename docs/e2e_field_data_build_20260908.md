# E2E parent-data build and physical-reference audit — 2026-09-08

## Scope and status

The user authorized one concurrent interactive slot for E2E data preparation,
with D2 untouched and model fits deferred. This is a balanced internal canary
dataset, not a production VAC or fresh external evidence. Observation/target
arrays and all independent native labels are complete; full-size numerical
validation passes. `data_engineering_ready=true`, but `training_ready=false`:
the scientific release gates below remain open.

Data root: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v1/data_20260908`.
Contract: `configs/e2e_field_build_v1.json`.
Archived receipts: `docs/evidence/e2e_field_v1/build_20260908/BUILD_REPORT.json`
and its adjacent native-reference, numerical-smoke and scheduler records.
Ten parent shards total 10,495,784,088 bytes; five independent-truth shards total
5,102,246,520 bytes. Partial/diagnostic files are excluded from these totals.

Existing P12-A, D2, P13, September-5 preparation products and all source arrays
are unchanged. Neither ph001 nor ph006 payloads are used in this build.

## Dataset and experimental unit

| Role | Simulation phases | Distinct anchors | Nested parent views |
| --- | --- | ---: | ---: |
| Training | ph000, ph002, ph003 | 96 | 192 |
| Internal selection | ph004 | 32 | 64 |
| Internal confirmation | ph005 | 32 | 64 |
| Total | Five historically used phases | 160 | 320 |

Each cap/phase has two anchors in each of four redshift shells and two support
strata. Deterministic aligned-lattice proposals are accepted using only radius
and random support in the central 32-cell region. Interior means at least 95%
support; boundary means 25--95%. No matter target or galaxy-count value selects
an anchor. All 80 cap/phase/shell/support strata filled.

Both sizes use the same anchor: 320 views are not 320 independent realizations.
Entire simulation phases, including all caps and box replicas, have one role.
The inherited source-box mapping finds 366 overlapping anchor pairs, none
crossing roles. Whole-phase grouping also guards aliases missed by rectangular
footprints. Nearby parents remain correlated; parent/voxel counts are not
effective independent-phase counts.

All five phases have historical programme exposure. ph004/ph005 are held out
from new E2E fitting/normalization, not fresh cosmological evidence. External
confirmation remains unassigned. Final training exposure and population weighting
must not be inferred from this strata-balanced canary panel alone.

## Array layout and access

One HDF5 shard per phase/cap stores, for each anchor:

- a complete 96-cubed `delta_r7_trace` target, without multiplying by survey support;
- a 128-cubed condition: eleven response-overlay channels plus observer redshift;
- separate latent, truth-available, loss, observed-support, science-geometry and
  science-supported masks; and
- global voxel origins/centers, units and role metadata.

The 64-cell target and 96-cell condition are exact central crops, not independently
reconstructed or normalized fields. Child cores have unique ownership, with both
layouts in `DATASET_INDEX.json`. The central science region is 32 cells across.
Density dependence can be tested across all children; derived tidal science
outside the audited central region is not automatically qualified.

Five observer-Mpc cells are 3.383 Mpc/h at h=0.6766. Parent sides are 216.512 and
324.768 Mpc/h; the central region is 108.256 Mpc/h. The snapshot remains z=0.2:
redshift-varying observed selection is not redshift-evolving truth.

`TRAIN_NORMALIZATION.json` uses ph000/ph002/ph003 only. Equal-volume stored
parents give equal weight per anchor. Count/expectation channels use log1p;
ntilde uses a declared log floor. Other transformations and affine parameters are
recorded. One field-space target scaler preserves the parent mean; old fitted
normalizers are not inherited.

`workflows.sbi.e2e_field_dataset.ParentDataset` rejects unreleased science-training
products by default. `allow_unreleased=True` is for explicit engineering tests,
not a training license. Confirmation access additionally requires
`allow_confirmation=True`. No scheduled model trainer enables either exception.

Full hashes are checked for targets, response overlays and virtual-dataset base
fields. The ph000 manifest retains pre-import paths: imported fields are accepted
only when byte-identical to the registered cap checksum. File size/name matching
is insufficient. Source mutation during extraction is rejected.

## Target-convention finding

The inherited scalar is the native R7 tensor trace sampled at nearest native
cells. Native T-web solves a Gaussian-smoothed Fourier Poisson problem, then
applies a centred first difference twice; it does not use spectral second
derivatives. On the native periodic lattice, before observer-grid sampling:

```text
delta_R7,trace(k) = [sum_i sin(k_i dx_native)^2 / (dx_native^2 k^2)]
                    W_R7(k) delta(k),             k != 0.
```

It approaches, but is not exactly, continuum Gaussian-smoothed density. A
spectral tensor solve on that trace does not exactly reproduce the native tensor
even on a full box. Nearest-cell sampling and parent boundaries add other effects.
Trace closure of a parent solve alone cannot establish native physical fidelity.

This clarifies the source definition without changing P12-A/D2 or their frozen
evaluations. New arrays are named `delta_r7_trace`. A continuum/spectral or
differently interpolated target would require a separate E2E target contract.

The independent reference builder reads the native 2048-cubed count field,
reconstructs the full-box smoothed potential, and samples six tensor components
with the native stencil. It does not infer orientations from eigenvalues or use
the same parent scalar as its own truth. Frozen eigenvalues and traces supply
numerical replay checks. The stencil was tested against installed vendored
FIESTA, used by `workflows/abacus_tweb/abacus_process_particles2.py`.

Full-box work uses about 199 GiB at peak on a CPU compute node. Full potentials
and tensor volumes are not retained: only parent reference arrays are saved.

## First domain comparison

Eight ph000 anchors were selected by shell/cap and random support before reading
truth. Reconstructed eigenvalues agree with frozen native values to within
2.75e-7; the inherited scalar trace matches exactly at all reference positions.
Across all 160 complete 96-cubed parent references, the worst eigenvalue replay
difference is 5.36e-7 and source-trace differences are zero. Selection and
confirmation packaging checks numerical replay only, not scientific scores.

At the same central 32-cubed region, a periodic parent spectral solve with the
declared isotropic parent-mean completion gives these ranges across anchors:

| Parent side | lambda1 RMSE / truth scatter | lambda2 | lambda3 |
| --- | ---: | ---: | ---: |
| 64 cells | 0.093--0.196 | 0.068--0.134 | 0.051--0.142 |
| 96 cells | 0.067--0.090 | 0.047--0.068 | 0.038--0.061 |

The larger parent roughly halves the error. The regional `lambda2 > 0.2` filling
fraction changes by at most 0.0019 in magnitude for 96 cells. These are
training-phase truth-only diagnostics, not posterior coverage or calibration.

The result supports retaining 96 cells for development. It does not separate all
discretization/boundary effects, express errors in conditional-posterior units,
qualify a final non-periodic physics convention, or bound errors in pair and
connectivity probabilities. R0-PHYSICS remains open. No retrospective science
threshold was chosen to make these numbers pass.

## Numerical checks and release boundary

The numerical suite tests low/high projection, parent DC, Haar depth-1/depth-2
round trips, globally addressed noise and synchronized Heun evolution under
different tilings. Haar uses local pairs without installing dependencies into
the environment shared with D2. It is not a learned-wavelet-arm license or a
test of the eventual neural sampler.

The numerical split preserves the inherited dimensionless Fourier cutoff;
its physical label uses 3.383 Mpc/h rather than the older 5-Mpc/h label.
Neither a science-training cutoff nor a wavelet depth is frozen by this test.

Readiness layers are:

1. source-verified arrays and independent labels;
2. geometry, scaler, transform and stochastic-identity checks;
3. a frozen R0-PHYSICS error budget and qualified operator/science region;
4. matched conditioner/transform, diagnostic-power and training contracts;
5. explicit model-launch authorization.

Packaging may finish while scientific release stays closed. This quarantined
input build does not relax the physical-adequacy requirement before science
training. MIRA remains supplementary; no full power study or learned canary is
performed here.

## Operational record

Allocation 58068916 (nid004186) completed screening, the independent ph000 audit
and all parent-array packaging. Its idle owner shell auto-logged out at 30 minutes,
releasing the allocation and terminating the active label step. This was not
out-of-memory. ph000 labels remain complete; `ph002_partial_58068916.h5` preserves
the interrupted shard and is excluded from manifests.

Replacement 58069599 completed labels and validation in the foreground through
`workflows/sbi/run_e2e_data_completion.sh`, then released the allocation after
15m22s. Both steps and the allocation have exit 0:0. Only one E2E allocation was
active at a time. No D2 job was submitted, cancelled, reprioritized or modified.

The first packaging probe rejected the ph000 path alias before writing any
parent arrays. Requiring its exact registered cap checksum resolved that issue.
Initial probes found FIESTA under `cactus.ext`; PyWavelets is absent and was not
installed. Attempt logs and the partial file are preserved.

Twenty-eight focused tests pass, including the eleven original tests. The final
compute-node validation took 21.2 seconds and verified all parent/native shard
hashes, unique phase sources, release/confirmation guards and six training
phase/cap nested-array and normalization checks. Target-scaler relative error is
at most 1.47e-8. At both 64 and 96 cells, Fourier/Haar relative errors are below
8e-16, tensor trace residual below 2.7e-15, and globally addressed noise and both
synchronized Heun tilings agree exactly. These are numerical fixtures, not tests
of an actual neural sampler or a full R0-WAVELET scientific pass.

The source AST and global knowledge graphs were refreshed. No learned model,
GPU training, calibration pass or scientific release is claimed. The next
decision is to freeze an R0-PHYSICS error budget and qualify the target/operator
and science region using these independent references, then complete the matched
transform/conditioner and diagnostic-power contracts before requesting fits.
