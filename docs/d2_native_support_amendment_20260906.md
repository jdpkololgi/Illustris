# D2 native-galaxy support amendment

## Execution status — reconciled 2026-09-08

The affected-core smoke and both seed42 evaluations completed successfully.
The unchanged decision code reports seed42 science and sampler convergence
passes under this explicit amendment, retaining all 133,698 galaxies. The first
evaluation attempt exhausted host memory; its partial directory is preserved,
and the identical evaluator completed with a 110-GiB host-memory allocation.
See [the seed42 decision](evidence/p12/p12f3_d2_20260906/D2_SEED42_PH006_DECISION.json)
and [memory recovery](evidence/p12/p12f3_d2_20260906/D2_MEMORY_RECOVERY.md).

The [licensed replication chain](evidence/p12/p12f3_d2_20260906/D2_REPLICATION_CHAIN_SUBMITTED.json)
is submitted, not yet a two-seed promotion. Training job 58065554 was explicitly
approved to accept any A100 node; all other resources and science settings stay
unchanged. Active polling was stopped at the user's request; this document does
not assert a freshly checked scheduler state. Frozen source/config/amendment
files must remain unchanged while that chain is outstanding.

2026-09-06: user explicitly approved retaining all galaxies with a verified
native-footprint check after the geometry/construction audit. This replaces the
earlier proposed supported-nearest-voxel subset amendment. No galaxies are dropped.

## Evidence and corrected explanation

All 133,698 galaxies are inside the native random-map angular/cap footprint and
the registered radial interval/sentinel rule. Exactly 736 have a nearest voxel
centre outside that footprint; 132,962 have a supported nearest centre. The
stored voxel mask matches an independent reconstruction at every sampled centre.
None of the 736 is near a context-patch edge (minimum 24.14 voxels); core-face
distance median is 2.39 voxels versus 2.45 for the supported population. Thus
patch truncation is excluded. The 736 are all within 1.341 voxels of a supported
voxel centre, with 722 within one voxel. This is native-footprint/voxel-centre
discretization at mask edges/holes on a 5 Mpc grid, not archive corruption or
galaxies genuinely outside the survey. Full row-level evidence and the sky plot
are archived under docs/evidence/p12/p12f3_d2_20260906/.

## Sole scientific change

Replace the derived-galaxy-calibration guard requiring nearest-voxel M=1 with
native angular/cap/radial membership evaluated at each authoritative galaxy's
canonical position. Preserve exact coordinate/order checks and rejection of
coordinates outside the patch. Apply the identity and native-support checks to
all candidate and reference archives. Retain all galaxies. Do not alter any voxel
mask, target, field sample, interpolation, FFT, normalization, loss, spectral
metric, conditional binning, uncertainty unit, threshold, training or sampler.
The existing references already use the same all-galaxy sample, so no reference
score refit/subset is introduced. Derived physical conditional gates retain all
133,698 galaxies; common field metrics and model-selection gates are unchanged.

## Additive contract and implementation

Do not edit the original D2_CONTRACT_FROZEN.json or immutable 467f442 worktree.
Create D2_NATIVE_SUPPORT_AMENDMENT.json alongside the base contract. It hashes the
base contract, geometry/construction audits, full-panel membership table, angular
map, geometry arrays, native verifier, amended entrypoint and Slurm launcher.
The amended entrypoint imports every original science module from the unchanged
467f442 worktree; its original Git revision and source/data guards still execute.
AST tests verify that all metric/plot functions are identical and the derived
function differs solely at the explicit support guard. The new main adds only
membership checks, smoke control and additive provenance, plus a new report path.

Reports go to reports_native_support_v1, preserving failed original report paths.
Both the evaluation digest and terminal marker bind the amendment hash. A wrapper
requires the same amendment for the 50/100-step ladder before calling the original
unchanged decision code; decision input hashes transitively bind the amendment.
Do not call this an unamended frozen-evaluator pass.

## Verification and scheduler recovery

Four focused tests cover true native rejection, typed/shape-safe flags, empty
panels, same-row acceptance despite an M=0 nearest voxel, unchanged masks, identity
failure and AST equivalence of all scientific calculations. A real affected core
must reproduce the original error and pass the new derived calculation with no
dropped rows under the existing interactive allocation. Only after that smoke
passes, submit bounded native evaluation jobs, replace stranded dependencies,
and call the unchanged decision machinery. Keep the running export100 recovery
57985431 and all original exports/checkpoints unchanged. No automatic unamended
dispatcher may launch evaluation after this amendment; follow any frozen science
licence with the native evaluator if a further seed/control is actually licensed.
